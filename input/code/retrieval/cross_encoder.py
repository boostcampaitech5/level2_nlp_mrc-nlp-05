import torch
import pickle
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_from_disk, load_metric
from tqdm import tqdm
import re

dataset = load_from_disk('/opt/ml/input/data/train_dataset')

with open('/opt/ml/input/data/bm25_sparse_embedding.bin', "rb") as f:
    bm25_embedding = pickle.load(f)
    
with open('/opt/ml/input/data/wikipedia_documents.json', 'r', encoding="utf-8") as f:
    wiki = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

def negative_sampling(dataset):
    contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))
    questions = dataset['question']
    passages = dataset['context']
    
    contexts = [re.sub(r'\n|\\n|\\\\n', ' ', c) for c in contexts]
    contexts = [re.sub(r"#", " ", c) for c in contexts]
    contexts = [re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”\":%&《》〈〉''㈜·\-\'+\s一-龥サマーン]", "", c) for c in contexts]
    contexts = [re.sub(r"\s+", " ", c).strip() for c in contexts]
    
    passages = [re.sub(r'\n|\\n|\\\\n', ' ', p) for p in passages]
    passages = [re.sub(r"#", " ", p) for p in passages]
    passages = [re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”\":%&《》〈〉''㈜·\-\'+\s一-龥サマーン]", "", p) for p in passages]
    passages = [re.sub(r"\s+", " ", p).strip() for p in passages]
    
    q_with_negs, p_with_negs = [], []

    bm25_acc_cnt = 0
    topk = 4

    for i in tqdm(range(len(questions)), desc='negative_sampling'):
        new_q = [questions[i]] * topk
        new_p = [passages[i]]
        
        top_contexts = bm25_embedding.get_top_n(tokenizer.tokenize(questions[i]), contexts, n=topk)
        
        if passages[i] in top_contexts:
            bm25_acc_cnt += 1
            top_contexts = [c for c in top_contexts if c != passages[i]]
            new_p.extend(top_contexts)
        else:
            new_p.extend(top_contexts[:-1])
        
        q_with_negs.extend(new_q)
        p_with_negs.extend(new_p)
        
    print('bm25_accu: ', bm25_acc_cnt / len(questions))

    return Dataset.from_dict({"question": q_with_negs, "passage": p_with_negs})

dataset['train'] = negative_sampling(dataset['train'])
dataset['validation'] = negative_sampling(dataset['validation'])
dataset.save_to_disk('/opt/ml/cross_encoder_data')

dataset = load_from_disk('/opt/ml/cross_encoder_data')

def preprocess_function(examples):
    inputs = tokenizer(
            examples['question'],
            examples['passage'],
            truncation=True,
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            padding="max_length"
        )
    
    labels = inputs['overflow_to_sample_mapping']
    labels = [1.0 if i % 4 == 0 else 0.0 for i in labels]
    inputs['labels'] = labels

    return inputs

dataset['train'] = dataset['train'].map(preprocess_function, batched=True, remove_columns=['question', 'passage'])
dataset['validation'] = dataset['validation'].map(preprocess_function, batched=True, remove_columns=['question', 'passage'])
model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', num_labels=1)

training_args = TrainingArguments(
    output_dir="/opt/ml/cross_encoder_results",
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=1000
)

metric = load_metric("pearsonr")

def compute_metrics(eval_pred):
    pearson = metric.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)
    pearson = pearson['pearsonr']
    
    return {'pearsonr': pearson}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()

eval_result = trainer.evaluate()
pearson_corr = eval_result["eval_pearsonr"]

print('pearson: ', pearson_corr)