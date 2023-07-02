import re
from tqdm import tqdm
from elasticsearch import Elasticsearch

class ES:
    def __init__(self):
        self.INDEX_NAME = "wiki-sample"
        self.INDEX_SETTINGS = {
            "settings": {
                "analysis": {
                    "filter": {
                        "my_shingle": {
                            "type": "shingle"
                        }
                    },
                    "analyzer": {
                        "my_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer",
                            "decompound_mode": "mixed",
                            "filter": ["my_shingle"]
                        }
                    },
                    "similairty": {
                        "my_similarity": {
                            "type": "BM25"
                        }
                    }
                }
            },

            "mappings": {
                "properties": {
                    "document_text": {
                        "type": "text",
                        "analyzer": "my_analyzer"
                    }
                }
            }
        }
        
        try:
            self.es.transport.close()
        except:
            pass

        self.es = Elasticsearch("http://localhost:9200", timeout=30, max_retries=10, retry_on_timeout=True)
        self.tv = []
    
    
    def preprocess(self, text):
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\\n", " ", text)
        text = re.sub(r"#", " ", text)
        text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥サマーン]", "", text)  
        text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
        
        return text


    def indexing(self, contexts):
        if self.es.indices.exists(self.INDEX_NAME):
            self.es.indices.delete(index=self.INDEX_NAME)
            self.es.indices.create(index=self.INDEX_NAME, body=self.INDEX_SETTINGS)
            self.es.indices.exists("wiki-sample")
        
        wiki_contexts = [self.preprocess(text) for text in contexts]
        wiki_articles = [{"document_text": wiki_contexts[i]} for i in range(len(wiki_contexts))]
        
        for i, text in enumerate(tqdm(wiki_articles)):
            try:
                self.es.index(index=self.INDEX_NAME, id=i, body=text)
            except:
                print(f"Unable to load document {i}.")
                
            self.tv.append(self.es.termvectors(index=self.INDEX_NAME, id=i, body={"fields" : ["document_text"]}))


    def search(self, query, topk):
        body = {
                "query": {
                    "bool": {
                        "must": [{"match": {"document_text": query}}],
                    }
                }
            }

        res = self.es.search(index=self.INDEX_NAME, body=body, size=topk)    
        scores = []
        indices = []
        
        for hit in res['hits']['hits']:
            scores.append(hit['_score'])
            indices.append(int(hit['_id']))
        
        print(scores)
        print(indices)
        
        return scores, indices