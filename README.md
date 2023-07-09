# level2_MRC-nlp-05

<br>

## 🐴Members

|<img src='https://avatars.githubusercontent.com/u/102334596?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/86002769?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/107304584?v=' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/60664644?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/126854237?v=4' height=100 width=100px></img>
| --- | --- | --- | --- | --- |
| [변성훈](https://github.com/DNA-B) | [서보성](https://github.com/Seoboseong) | [이상민](https://github.com/SangMini2) | [이승우](https://github.com/OLAOOT) | [이예원](https://github.com/aeongaewon) |

<br>

## 📎ODQA (Open-Domain Question Answering)

> 부스트캠프 AI-Tech 5기 NLP 트랙 Level2 3차 경진대회 프로젝트입니다. Question Answering 은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 특히 ***Open-Domain Question Answering (ODQA)*** 은 주어지는 지문이 따로 존재하지 않고 사전에 구축된 Knowledge resource에서 질문에 답할 수 있는 문서를 찾는 과정이 추가됩니다. 본 대회에서 우리가 만들어야 했던 모델은 질문에 관련된 문서를 찾아주는 ***retriever*** - 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 ***reader***의 two-stage로 구성되어 있습니다.
> 

<br>

### Data

- 데이터 구성  
![image](https://github.com/DNA-B/Open-Domain-Question-Answering/assets/102334596/f1a23caf-a8d5-43fe-8eb5-94c53051b4b9)

<br>

- 데이터 예시  
![image](https://github.com/DNA-B/Open-Domain-Question-Answering/assets/102334596/7309b366-c047-4ad6-bdc1-8b8c7d38d189)
  
<br>

### Metric
- Exact Match (EM), F1 Score (보조)

<br><br>

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

<br>

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

<br>

### Reader
- Model, Tokenizer Tuning
- Curriculum Learning
- Transfer Learning
- Negative Sampling

<br>

### Sentence-level Approach (Bremen Special)

<br>

### Ensemble
- nbest-probs soft voting
  
<br>


💡 __*위에 관한 자세한 내용은 [Wrap-up Report](https://github.com/boostcampaitech5/level2_nlp_mrc-nlp-05/blob/dev/mrc_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(05).pdf)를 참고해주세요.*__

<br>

## 🐞Usage

```bash
# TRAIN
python3 input/code/train.py

# INFERENCE
python3 input/code/inference.py
```

<br>

## 🏆Result

- Public 3위
![image](https://github.com/DNA-B/Open-Domain-Question-Answering/assets/102334596/ec20af96-a502-4357-b14f-72957bf9ffd9)

- Private 2위
![image](https://github.com/DNA-B/Open-Domain-Question-Answering/assets/102334596/fcb03111-13c5-4ad6-b61c-a4307c736e1d)
