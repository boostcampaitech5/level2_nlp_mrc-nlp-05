import json
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union
from utils.es import *
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

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

        self.ES = ES()
        self.es = None
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        
        self.ids = list(range(len(self.contexts)))     
        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.


    def get_sparse_embedding(self) -> None:
            self.es = self.ES.indexing(self.contexts)
            self.p_embedding = self.ES.tv
            print("Embedding clear.")
            
            
    def retrieve(
            self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, split=False,
        ) -> Union[Tuple[List, List], pd.DataFrame]:

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
                    
                if split:
                    doc_scores = doc_scores
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
            
            
    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        
        doc_scores = []
        doc_indices = []
        
        for query in queries:
            tmp = self.ES.search(query, k)
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
