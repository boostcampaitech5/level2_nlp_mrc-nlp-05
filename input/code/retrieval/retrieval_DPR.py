import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from DenseModel.DenseModel import *

from transformers import AutoTokenizer

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrieval:
    """ DenseRetrieval을 하기위한 Class
    """
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """ DenseRetrieval 생성자

        Args:
            tokenize_fn (Tokenizer.tokenize): Query나 Passage를 Tokenize하기 위한 함수
            data_path (Optional[str], optional): Data의 Path. Defaults to "../data/".
            context_path (Optional[str], optional): Context의 Path. Defaults to "wikipedia_documents.json".
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        MODEL_NAME = "klue/roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.Model = ColBERTModel.from_pretrained(MODEL_NAME)
        self.Model.load_state_dict(torch.load("/opt/ml/input/code/retrieval/Model.pt"))
        self.Model.to("cuda")

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> None:
        """ Context를 미리 Embedding 하기 위한 함수
        """
        pickle_name = f"p_DenseEmbedding.bin"

        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")

            p_emb = list()

            for p in tqdm(self.contexts):
                context_emb = self.tokenizer(p, stride=128, truncation=True, padding="max_length", return_tensors='pt').to('cuda')
                p_emb.append(self.p_encoder(**context_emb).pooler_output.to('cpu').detach().numpy())

            self.p_embedding = np.array(p_emb).squeeze()

            print(self.p_embedding.shape)

            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """ Retrieval 동작 함수

        Args:
            query_or_dataset (Union[str, Dataset]): Query나 Queries 인자
            topk (Optional[int], optional): Topk를 설정해주는 인자

        Returns:
            Union[Tuple[List, List], pd.DataFrame]: 유사도가 높은 Topk개의 Passage
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
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

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """ 단일 Query에 대해서 유사도가 높은 Topk개의 Passage를 찾는 함수

        Args:
            query (str): Query
            k (Optional[int], optional): Topk의 값. Defaults to 1.

        Returns:
            Tuple[List, List]: 유사도가 높은 Topk개의 Passage
        """
        with timer("transform"):
            query_emb = self.tokenizer([query], stride=128, truncation=True, padding="max_length", return_tensors='pt').to('cuda')
            query_vec = self.Model.query(**query_emb).to('cpu').detach().numpy()
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = self.Model.get_score(query_vec, self.p_embedding).numpy()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """ 다중 Query에 대해서 각각 유사도가 높은 Topk개의 Passage를 찾는 함수

        Args:
            query (str): Query
            k (Optional[int], optional): Topk의 값. Defaults to 1.

        Returns:
            Tuple[List, List]: 각각의 Query에 대해서 유사도가 높은 Topk개의 Passage
        """

        q_emb = list()

        for q in tqdm(queries):
            query_emb = self.tokenizer(q, stride=128, truncation=True, padding="max_length", return_tensors='pt').to('cuda')
            q_emb.append(self.Model.query(**query_emb).to('cpu').detach().numpy())
        
        query_vec = np.array(q_emb).squeeze()
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = self.Model.get_score(query_vec, self.p_embedding).numpy()

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

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
    full_ds = org_dataset["validation"]
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)

    retriever = DenseRetrieval(
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
