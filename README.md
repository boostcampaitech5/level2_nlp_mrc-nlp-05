# level2_mrc-nlp-05

## 🐴Members

|<img src='https://avatars.githubusercontent.com/u/102334596?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/86002769?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/107304584?v=' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/60664644?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/126854237?v=4' height=100 width=100px></img>
| --- | --- | --- | --- | --- |
| [변성훈](https://github.com/DNA-B) | [서보성](https://github.com/Seoboseong) | [이상민](https://github.com/SangMini2) | [이승우](https://github.com/OLAOOT) | [이예원](https://github.com/aeongaewon) |

## 📎ODQA (Open-Domain Question Answering)

> 부스트캠프 AI-Tech 5기 NLP 트랙 Level2 3차 경진대회 프로젝트입니다. Question Answering 은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 특히 **Open-Domain Question Answering (ODQA)**은 주어지는 지문이 따로 존재하지 않고 사전에 구축된 Knowledge resource에서 질문에 답할 수 있는 문서를 찾는 과정이 추가됩니다. 본 대회에서 우리가 만들어야 했던 모델은 질문에 관련된 문서를 찾아주는 **retriever** - 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 **reader**의 two-stage로 구성되어 있습니다.
> 

### Data (Private)

![data](https://github.com/boostcampaitech5/level2_nlp_mrc-nlp-05/assets/60664644/f3f5f7fc-97b0-41d2-b5d2-22b064673609)

![data_example](https://github.com/boostcampaitech5/level2_nlp_mrc-nlp-05/assets/60664644/22debc45-cf75-44ef-9c17-0611d7d459cf)

### Metric

- Exact Match (EM), F1 Score (보조)
  
## ✔️Project

### Structure

```
root/input/code/
|
|-- BS_inference.py
|-- BS_train.py
|-- BSQuestionAnsweringModel.py
|-- curriculum.py
|-- evaluation.py
|-- find_answer_sentence.py
|-- inference.py
|-- prepare_dataset.py
|-- scores_voting.py
|-- train.py
|-- trainer_qa.py
|-- utils_qa.py
|
|-- ensemble/
|   |-- probs_voting_ensemble_n.py
|   |-- probs_voting_ensemble.py
|   |-- scores_voting_ensemble.py
|
|-- retrieval/
|   |-- DenseModel/
|   |   |-- DenseModel.py
|   |   |-- train.py
|   |-- cross_encoder.py
|   |-- retrieval_BM25.py
|   |-- retrieval_DPR.py
|   |-- retrieval_ES.py
|   |-- retrieval_reranking.py
|   |-- retrieval_reranking2.py
|   |-- retrieval_TFIDF.py
|
|-- utils/
|   |-- elastic_search.py
|   |-- es.py
|   |-- naming.py

```

### Data Analysis

- Data Augmentation
- Preprocessing
- Prompt Tuning

### Retriever

- Retriever Tokenizer
- Sparse Retriever
    - TF-IDF
    - BM25
    - BM25 + TF-IDF
    - Elastic Search
- Dense Retriever
    - Bi-Encoder
    - Cross-Encoder
    - Colbert
- Inference Method
    - [SEP] special token
    - Independent Documents Inference

### Reader

- Model, Tokenizer Tuning
- Curriculum Learning
- Transfer Learning
- Negative Sampling

### Sentence-level Approach (Bremen Special)

### Ensemble

- nbest-probs soft voting


💡 __*위에 관한 자세한 내용은 [Wrap-up Report](https://github.com/boostcampaitech5/level2_klue-nlp-05/blob/main/%5BNLP-05%5Dklue_wrapup_report.pdf)를 참고해주세요.*__

## 🐞Usage

```bash
# TRAIN
python3 input/code/train.py

# INFERENCE
python3 input/code/inference.py
```

## 🏆Result

- Public 3위

![public](https://github.com/boostcampaitech5/level2_nlp_mrc-nlp-05/assets/60664644/440542d9-bf3c-4594-9d29-952cb6e2c545)


- Private 2위

![private](https://github.com/boostcampaitech5/level2_nlp_mrc-nlp-05/assets/60664644/9c42b0db-7501-41b0-b222-23b61033fe6a)