import json
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union
from utils.elastic_search import *
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
import pickle

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class ESSparseRetrieval:
    def __init__(
        self,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))     
        self.indexer = None  # build_faiss()로 생성합니다.              


    def retrieve(
            self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, independent=False,
        ) -> Union[Tuple[List, List], pd.DataFrame]:
        """Arguments:
            query_or_dataset (Union[str, Dataset]):
                Dataset으로 이루어진 Query를 받습니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다
        """ 
        
        pickle_name = f"es_embedding.bin"
        es_bm25_name = f"es_bm25.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        es_bm25_path = os.path.join(self.data_path, es_bm25_name)
        
        if os.path.isfile(emd_path) and os.path.isfile(es_bm25_path): # ES search를 통한 유사도 파일이 있다면 load
            with open(emd_path, "rb") as file:
                doc_indices = pickle.load(file)
            with open(es_bm25_path, "rb") as file:
                doc_scores = pickle.load(file)
            print("Embedding pickle load.")
            
        else: # search를 한 번도 진행하지 않은 경우 Indexing 후 Searching
            print("Build passage embedding")   
            ES = ElasticSearch()
            ES.indexing(self.contexts) # Indexing -> Embedding
            
            # dump scores, indices
            with timer("query exhaustive search"): 
                doc_scores, doc_indices = self.get_relevant_doc_bulk(ES, query_or_dataset["question"], k=topk)    
            with open(emd_path, "wb") as file:
                pickle.dump(doc_indices, file)
            with open(es_bm25_path, "wb") as file:
                pickle.dump(doc_scores, file)                        
            print("Embedding pickle saved.")

        total = []
                        
        if independent:
            doc_scores = np.array(doc_scores)
            doc_scores = doc_scores / np.max(doc_scores)
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
                    
                if independent:
                    doc_scores = np.array(doc_scores)
                    doc_scores = doc_scores / np.max(doc_scores)
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

    def get_relevant_doc_bulk(
        self, ES, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Return:
            topk개 문서의 점수와 인덱스 -> Tuple(List, List)            
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        
        doc_scores = []
        doc_indices = []
        
        for query in queries:
            tmp = ES.search(query, k)
            doc_scores.append(tmp[0])
            doc_indices.append(tmp[1])
            
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
    
    retriever = ESSparseRetrieval(
        data_path=args.data_path,
        context_path=args.context_path,
    )
