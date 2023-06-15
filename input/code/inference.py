"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
import pandas as pd

from typing import Callable, Dict, List, Tuple

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk
)
import evaluate
import argparse
import json
# import wandb

from retrieval.retrieval_TFIDF import TFIDFSparseRetrieval
from retrieval.retrieval_BM25 import BM25SparseRetrieval
from retrieval.retrieval_reranking import RerankSparseRetrieval
from retrieval.retrieval_reranking2 import RerankSparseRetrieval2
from scores_voting import post_process_voting
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
)
from utils_qa import set_seed, check_no_error, postprocess_qa_predictions
from omegaconf import OmegaConf
from omegaconf import DictConfig
import konlpy.tag as konlpy
import discord

logger = logging.getLogger(__name__)


def main(args):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    model_args, data_args = args.model, args.data
               
    training_args = TrainingArguments(
        output_dir=args.train.inference_output_dir,
        overwrite_output_dir = True,
        do_train=args.train.do_train,
        do_eval=args.train.do_eval,
        do_predict=args.train.do_predict,
        save_total_limit=3,
        num_train_epochs=args.train.max_epoch,
        learning_rate=args.train.learning_rate,
        per_device_train_batch_size=args.train.batch_size,
        per_device_eval_batch_size=args.train.batch_size,
        evaluation_strategy="steps",
        eval_steps=args.train.eval_step,
        logging_steps=args.train.logging_step,
        save_steps=args.train.save_step,
        warmup_steps=args.train.warmup_steps,
        weight_decay=args.train.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='exact_match'
    )
    
    # wandb.init(project=args.wandb.project, name=args.wandb.name)
    
    training_args.do_train = True

    print(f"model is from {model_args.saved_model_path}")
    print(f"data is from {data_args.test_dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    datasets = load_from_disk(data_args.test_dataset_name)
    test_df = pd.DataFrame(datasets['validation'])
    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.saved_model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.saved_model_path,
        use_fast=True,
    )
        
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.saved_model_path,
        from_tf=bool(".ckpt" in model_args.saved_model_path),
        config=config,
    )

    doc_scores = None
    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
<<<<<<< HEAD
        # konlpy 계열은 morphs, huggingface 계열은 tokenize를 사용하므로 구분
        if model_args.retrieval_tokenizer not in ['mecab', 'hannanum', 'kkma', 'komoran', 'okt']:
            retrieval_tokenizer = AutoTokenizer.from_pretrained(model_args.retrieval_tokenizer)
            tokenize_fn = retrieval_tokenizer.tokenize
        else:
            if model_args.retrieval_tokenizer == 'mecab':
                retrieval_tokenizer = konlpy.Mecab()
            elif model_args.retrieval_tokenizer == 'hannanum':
                retrieval_tokenizer = konlpy.Hannanum()
            elif model_args.retrieval_tokenizer == 'kkma':
                retrieval_tokenizer = konlpy.Kkma()
            elif model_args.retrieval_tokenizer == 'komoran':
                retrieval_tokenizer = konlpy.Komoran()
            else:
                retrieval_tokenizer = konlpy.Okt()
            tokenize_fn = retrieval_tokenizer.morphs
        print('retrieval_tokenizer:', retrieval_tokenizer)
        
        datasets = run_sparse_retrieval(
            tokenize_fn, datasets, training_args, data_args,
=======
        datasets, doc_scores = run_sparse_retrieval(
            retrieval_tokenizer.tokenize, datasets, training_args, data_args,
>>>>>>> 4af4a9aa0b253a3f2f65f7e9b3884fd4796530f0
        )
    

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model, test_df, doc_scores)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DictConfig,
    data_path: str = "/opt/ml/input/data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    if data_args.retrieval_type == 'tfidf':
        retriever = TFIDFSparseRetrieval
    elif data_args.retrieval_type == 'bm25':
        retriever = BM25SparseRetrieval
    elif data_args.retrieval_type == 'tfidf+bm25':
        retriever = RerankSparseRetrieval
    elif data_args.retrieval_type == 'tfidf+bm25_2':
        retriever = RerankSparseRetrieval2
        
    retriever = retriever(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    ) 
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        if data_args.split:
            doc_scores, df_list = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, split=True)
        else:
            df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, split=False)
            
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "original_context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    #if training_args.do_predict:
    #    datasets2 = load_from_disk('/opt/ml/input/data/test_dataset13')
    #elif training_args.do_eval:
    #    datasets2 = load_from_disk('/opt/ml/input/data/train_dataset7')

    #valid_df = datasets2['validation']
    #df['question'] = valid_df['question']
    
    if data_args.split:
        datasets2 = load_from_disk('/opt/ml/input/data/test_dataset13') ##
        dataset_list = []
        for i in range(data_args.top_k_retrieval):
            df_list[i]['question'] = datasets2['validation']['question'] ##
            dataset = DatasetDict({"validation": Dataset.from_pandas(df_list[i], features=f)})
            dataset_list.append(dataset)
        return dataset_list, doc_scores
    else:
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return [datasets]


def run_mrc(
    data_args: DictConfig,
    training_args: TrainingArguments,
    model_args: DictConfig,
    datasets: DatasetDict,
    tokenizer,
    model,
    test_df,
    doc_scores=None,
) -> None:

    if data_args.split:
        run = data_args.top_k_retrieval
    else:
        run = 1
    
    eval_dataset = []
        
    for i in range(run):
        # eval 혹은 prediction에서만 사용함
        column_names = datasets[i]["validation"].column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]
        
        # Padding에 대한 옵션을 설정합니다.
        # (question|context) 혹은 (context|question)로 세팅 가능합니다.
        pad_on_right = tokenizer.padding_side == "right"

        # 오류가 있는지 확인합니다.
        last_checkpoint, max_seq_length = check_no_error(
            data_args, training_args, datasets[i], tokenizer
        )

        # Validation preprocessing / 전처리를 진행합니다.
        def prepare_validation_features(examples):
            # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
            # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
                
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=False if 'roberta' in model_args.model_name else True, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
            # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # sequence id를 설정합니다 (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                # 하나의 example이 여러개의 span을 가질 수 있습니다.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            return tokenized_examples

        eval_dataset.append(datasets[i]["validation"])

        # Validation Feature 생성
        eval_dataset[i] = eval_dataset[i].map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Data collator
        # flag가 True이면 이미 max length로 padding된 상태입니다.
        # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )

        # Post-processing:
        def post_processing_function(
            examples,
            features,
            predictions: Tuple[np.ndarray, np.ndarray],
            training_args: TrainingArguments,
        ) -> EvalPrediction:
            # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=data_args.max_answer_length,
                output_dir=training_args.output_dir,
            )
            # Metric을 구할 수 있도록 Format을 맞춰줍니다.
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

            if training_args.do_predict:
                return formatted_predictions
            elif training_args.do_eval:
                references = [
                    {"id": ex["id"], "answers": ex[answer_column_name]}
                    for ex in datasets["validation"]
                ]

                return EvalPrediction(
                    predictions=formatted_predictions, label_ids=references
                )

        metric = evaluate.load("squad")

        def compute_metrics(p: EvalPrediction) -> Dict:
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        print("init trainer...")
        # Trainer 초기화
        
        if data_args.split:
            training_args.output_dir = f'{args.train.inference_output_dir}/split_prediction/{i}_pred'
            
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset[i],
            eval_examples=datasets[i]["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
        )

        logger.info("*** Evaluate ***")

        #### eval dataset & eval example - predictions.json 생성됨
        if training_args.do_predict:
            predictions = trainer.predict(
                test_dataset=eval_dataset[i], test_examples=datasets[i]["validation"]
            )

            # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
            print(
                "No metric can be presented because there is no correct answer given. Job done!"
            )
        elif training_args.do_eval:
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(eval_dataset)

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
        
    if data_args.split:
        post_process_voting(doc_scores, args.train.inference_output_dir, data_args.top_k_retrieval, test_df)
        
        if training_args.do_eval:
            predict_path = f'{args.train.inference_output_dir}/predictions.json'
            with open(predict_path, 'r') as json_file:
                prediction = json.load(json_file)
            ground_truth = {item["id"]: item['answesrs']["text"] for item in eval_dataset[0]}
            print(metric.compute(predictions=prediction, references=ground_truth))
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--args_path", default=f"/opt/ml/args.yaml", type=str, help=""
    )
    arg = parser.parse_args()
    
    args = OmegaConf.load(arg.args_path)
    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(args.train.seed)  
    
    main(args)
