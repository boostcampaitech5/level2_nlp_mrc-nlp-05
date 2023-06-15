import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class RerankSparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.tokenizer = tokenize_fn
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000,
        )
        self.bm25 = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> None:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        bm25_pickle_name = f"bm25_sparse_embedding.bin"
        bm25_emd_path = os.path.join(self.data_path, bm25_pickle_name)
        
        tfidf_pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        tfidf_emd_path = os.path.join(self.data_path, tfidf_pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(bm25_emd_path) and os.path.isfile(tfidf_emd_path):
            with open(bm25_emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
            with open(tfidf_emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            
            tokenized_corpus = [self.tokenizer(doc) for doc in self.contexts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            with open(bm25_emd_path, "wb") as file:
                pickle.dump(self.bm25, file)
            
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            
            print(self.p_embedding.shape)
            
            with open(tfidf_emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
                
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> None:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.bm25.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, split=False,
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            bm25_doc_scores, bm25_doc_indices = self.bm25_get_relevant_doc(query_or_dataset, k=topk)
            tfidf_doc_scores, tfidf_doc_indices = self.tfidf_get_relevant_doc(query_or_dataset, k=topk)
            
            doc_indices = []
            doc_scores = []
            
            for i in tqdm(range(len(bm25_doc_indices)), total=len(bm25_doc_indices)):
                doc_indice, doc_score = [], []
                bm25_doc_scores[i] = bm25_doc_scores[i] / np.max(bm25_doc_scores[i])
                tfidf_doc_scores[i] = tfidf_doc_scores[i] / np.max(tfidf_doc_scores[i])  
                 
                for j in range(len(bm25_doc_indices[i])):
                    if bm25_doc_indices[i][j] == tfidf_doc_indices[i][j]:
                        doc_score.append((bm25_doc_scores[i][j] + tfidf_doc_scores[i][j])/2)
                        doc_indice.append(bm25_doc_indices[i][j])
                    else:
                        doc_score.append(bm25_doc_scores[i][j])
                        doc_indice.append(bm25_doc_indices[i][j])
                        doc_score.append(tfidf_doc_scores[i][j])
                        doc_indice.append(tfidf_doc_indices[i][j])
                     
                doc_score, doc_indice = np.array(doc_score), np.array(doc_indice)

                # doc_scores를 기준으로 내림차순으로 정렬한 인덱스 배열 생성
                sorted_indices = np.argsort(-doc_score)

                # doc_scores와 doc_indices를 정렬된 인덱스를 기준으로 정렬
                doc_scores.append(doc_score[sorted_indices][:topk])
                doc_indices.append(doc_indice[sorted_indices][:topk])
                
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            with timer("query exhaustive search"):
                bm25_doc_scores, bm25_doc_indices = self.bm25_get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
                tfidf_doc_scores, tfidf_doc_indices = self.tfidf_get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            
            doc_indices = []
            doc_scores = []
            
            for i in tqdm(range(len(bm25_doc_indices)), total=len(bm25_doc_indices)):
                doc_indice, doc_score = [], []
                bm25_doc_scores[i] = bm25_doc_scores[i] / np.max(bm25_doc_scores[i])
                tfidf_doc_scores[i] = tfidf_doc_scores[i] / np.max(tfidf_doc_scores[i])  
                 
                for j in range(len(bm25_doc_indices[i])):
                    if bm25_doc_indices[i][j] == tfidf_doc_indices[i][j]:
                        doc_score.append((bm25_doc_scores[i][j] + tfidf_doc_scores[i][j])/2)
                        doc_indice.append(bm25_doc_indices[i][j])
                    else:
                        doc_score.append(bm25_doc_scores[i][j])
                        doc_indice.append(bm25_doc_indices[i][j])
                        doc_score.append(tfidf_doc_scores[i][j])
                        doc_indice.append(tfidf_doc_indices[i][j])
                     
                doc_score, doc_indice = np.array(doc_score), np.array(doc_indice)

                # doc_scores를 기준으로 내림차순으로 정렬한 인덱스 배열 생성
                sorted_indices = np.argsort(-doc_score)

                # doc_scores와 doc_indices를 정렬된 인덱스를 기준으로 정렬
                doc_scores.append(doc_score[sorted_indices][:topk])
                doc_indices.append(doc_indice[sorted_indices][:topk])  
            
            if split:
                cqas_lst = [] 
                for i in range(topk):
                    total = []
                    for idx, example in enumerate(
                        tqdm(query_or_dataset, desc="Sparse retrieval: ")
                    ):
                        tmp = {
                            # Query와 해당 id를 반환합니다.
                            "question": example["question"],
                            "id": example["id"],
                            # Retrieve한 Passage의 id, context를 반환합니다.
                            "context": self.contexts[doc_indices[idx][i]],
                        }
                        if "context" in example.keys() and "answers" in example.keys():
                            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                            tmp["original_context"] = example["context"]
                            tmp["answers"] = example["answers"]
                        total.append(tmp)
                    cqas = pd.DataFrame(total)
                    cqas_lst.append(cqas)    
                return doc_scores, cqas_lst
            else:
                total = []        
                for idx, example in enumerate(
                    tqdm(query_or_dataset, desc="Sparse retrieval: ")
                ):
                    tmp = {
                        # Query와 해당 id를 반환합니다.
                        "question": example["question"],
                        "id": example["id"],
                        # Retrieve한 Passage의 id, context를 반환합니다.
                        "context": " ".join(
                            [self.contexts[pid] for pid in doc_indices[idx]]
                        ),
                    }
                    if "context" in example.keys() and "answers" in example.keys():
                        # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    total.append(tmp)

                cqas = pd.DataFrame(total)
                return cqas

    def bm25_get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        tokenized_query = [self.tokenizer(query) for query in list(query)]
        result = np.array([self.bm25.get_scores(query) for query in tokenized_query])
        doc_scores = []
        doc_indices = []
        for scores in tqdm(result, total=len(result)):
            sorted_result = np.argsort(scores)[-2*k:][::-1]
            doc_scores.append(scores[sorted_result])
            doc_indices.append(sorted_result.tolist())
        
        return doc_scores, doc_indices

    def bm25_get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        tokenized_query = [self.tokenizer(query) for query in list(queries)]
        result = np.array([self.bm25.get_scores(query) for query in tqdm(tokenized_query, total=len(tokenized_query))])
        doc_scores = []
        doc_indices = []
        for scores in tqdm(result, total=len(result)):
            sorted_result = np.argsort(scores)[-2*k:][::-1]
            doc_scores.append(scores[sorted_result])
            doc_indices.append(sorted_result.tolist())
        
        return doc_scores, doc_indices

    def tfidf_get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:2*k]
        doc_indices = sorted_result.tolist()[:2*k]
        return doc_score, doc_indices
    
    def tfidf_get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:2*k])
            doc_indices.append(sorted_result.tolist()[:2*k])
            
        return doc_scores, doc_indices
    
    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="/opt/ml/input/data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="/opt/ml/input/data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument("--top_k_retrieval", default=10, type=int, help="")

    args = parser.parse_args()
    
    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    #full_ds = concatenate_datasets(
    #    [
    #        org_dataset["train"].flatten_indices(),
    #        org_dataset["validation"].flatten_indices(),
    #    ]
    #)  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    full_ds = org_dataset["validation"]
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)

    retriever = RerankSparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:
        retriever.get_sparse_embedding()

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df.apply(lambda row: row['original_context'] in row['context'], axis=1)
            
            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        retriever.get_sparse_embedding()
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=args.top_k_retrieval)
            df["correct"] = df.apply(lambda row: row['original_context'] in row['context'], axis=1)
            
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)