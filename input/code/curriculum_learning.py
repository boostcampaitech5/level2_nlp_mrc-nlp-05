from datasets import load_from_disk
from transformers import AutoModelForQuestionAnswering, AutoConfig, AutoTokenizer, EvalPrediction
from omegaconf import OmegaConf
from custom import NewModelwithLinear, NewModelwithReverseLSTM
import torch
import evaluate
from tqdm import tqdm

args = OmegaConf.load("/opt/ml/args.yaml")

model_args, data_args = args.model, args.data
checkpoint = torch.load('/opt/ml/models/train_dataset/pytorch_model.bin')
model_args.model_name_or_path = model_args.saved_model_path
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

if model_args.Custom_model == False:
    config=AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
else :
    checkpoint = torch.load('/opt/ml/models/train_dataset/pytorch_model.bin')
    if model_args.Custom_model == "ReverseLSTM" :
        model=NewModelwithReverseLSTM(model_name=model_args.model_name)
        
    elif model_args.Custom_model == "Linear" :
        model=NewModelwithLinear(model_name=model_args.model_name)
    
    model.load_state_dict(checkpoint)

max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
pad_on_right=True

def prepare_train_features(examples):
    # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
    # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
    tokenized_examples = tokenizer(
        examples[question_column_name],
        examples[context_column_name],
        truncation="only_second",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False if 'roberta' in model_args.model_name else True, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
    # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 데이터셋에 "start position", "enc position" label을 부여합니다.
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        #cls_index = torch.where(input_ids == tokenizer.cls_token_id)[0][0] # tensor 버전
        cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i) # token type id랑 똑같은데, special token에는 None 값이 있음.

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]

        # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # text에서 정답의 Start/end character index
            start_char = answers["answer_start"][0] # 정답은 항상 1개 인가?
            end_char = start_char + len(answers["text"][0])

            # text에서 current span의 Start token index
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # text에서 current span의 End token index
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def post_processing_function(examples, features, predictions, training_args):
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

def compute_metrics(p: EvalPrediction):
    metrics = metric.compute(predictions=p.predictions, references=p.label_ids)
    exact_match = metrics['exact_match']
    f1 = metrics['f1']
        
    return {'eval_exact_match' : exact_match, 'eval_f1' : f1}

metric = evaluate.load("squad")

# 후처리를 해주고 , predictions = [{'prediction_text': 'sdfsdfasdf', "id":"~~~"}] 이렇게 준다
# references = [{'answers':{'answer_start':[number], 'text':['text_thing']}, 'id':"~~~"}]
# 그리고 metric.compute(predictions=predictions, references=references)


dataset = load_from_disk("/opt/ml/input/data/train_dataset")
train_dataset=dataset['train']
eval_dataset=dataset['validation']

column_names = train_dataset.column_names

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]


train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

# train_dataset = train_dataset.map(make_tensors)

breakpoint()