from datasets import load_from_disk
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from torch.optim import AdamW
from tqdm import trange, tqdm
import torch.nn.functional as F
import pandas as pd
import torch
import random
import numpy as np
import json
import os
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModel, 
    get_linear_schedule_with_warmup, 
    is_torch_available, 
    TrainingArguments
)
from DenseModel import *
from rank_bm25 import BM25Okapi
import pickle

def set_seed(seed = 42):
    """ 학습시 seed 고정을 위한 함수

    Args:
        seed (int, optional): 입력 seed. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_data():
    """ 학습할 데이터를 불러오는 함수

    Returns:
        list, list: 정답이 포함된 유사도가 높은 Topk의 Passage list와 정답 Passage의 위치 list
    """
    data_path = "/opt/ml/input/data/topk_context.pkl"
    targets_path = "/opt/ml/input/data/targets.pkl"

    if os.path.isfile(data_path) and os.path.isfile(targets_path):
        with open(data_path, "rb") as file:
            topk_context = pickle.load(file)
        with open(targets_path, "rb") as file:
            targets = pickle.load(file)
    else:
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-finetuned-korquad")

        wiki_path = "/opt/ml/input/data/wikipedia_documents.json"
        file_path = "/opt/ml/input/data/bm25_sparse_embedding.bin"

        with open(file_path, "rb") as file:
            p_embedding = pickle.load(file)

        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        train_dataset = load_from_disk("/opt/ml/input/data/train_dataset")['train']
        contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        new_contexts = list()

        for context in tqdm(contexts):
            new_contexts.append(context.replace("\n", "").replace("\\n", ""))

        topk_context = list()
        targets = list()

        for data in tqdm(train_dataset):
            topk_doc = p_embedding.get_top_n(tokenizer.tokenize(data['question']), new_contexts, n=5)

            new_topk_doc = list()
            data_context = data['context'].replace("\n", "").replace("\\n", "")

            if data_context not in topk_doc:
                new_topk_doc.append(data_context)
                new_topk_doc.extend(topk_doc[:4])
            else:
                new_topk_doc.extend(topk_doc)

            count = 0

            while count != 4:
                sent = random.choice(new_contexts)

                if sent not in new_topk_doc and sent != data_context:
                    new_topk_doc.append(sent)
                    count += 1

            random.shuffle(new_topk_doc)

            topk_context.append(new_topk_doc)
            targets.append(new_topk_doc.index(data_context))

        with open(data_path, "wb") as file:
            pickle.dump(topk_context, file)
        with open(targets_path, "wb") as file:
            pickle.dump(targets, file)

    return topk_context, targets

def train():
    """ Dense Model 학습 함수
    """
    set_seed()

    topk_contexts, targets = load_data()

    args = TrainingArguments(
        output_dir = "DenseModel",
        per_device_train_batch_size = 4,
        learning_rate = 1e-5,
        adam_epsilon = 1e-8,
        gradient_accumulation_steps = 8,
        num_train_epochs = 10,
        weight_decay = 0.01,
        warmup_steps = 200
    )

    MODEL_NAME = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_from_disk("/opt/ml/input/data/train_dataset")

    train_dataset = dataset['train']

    tokenized_train_query_dataset = tokenizer(train_dataset['question'], stride=128, truncation=True, padding="max_length", return_tensors='pt').to('cuda')
    tmp_tokenized_train_context_dataset = list()
    tokenized_train_context_dataset = list()

    for context in tqdm(topk_contexts):
        tmp_tokenized_train_context_dataset.append(tokenizer(context, stride=128, truncation=True, padding="max_length", return_tensors='pt').to('cuda'))

    for i in tqdm(range(len(tmp_tokenized_train_context_dataset))):
        tokenized_train_context_dataset.append(
            torch.stack([tmp_tokenized_train_context_dataset[i]['input_ids'], 
                         tmp_tokenized_train_context_dataset[i]['token_type_ids'], 
                         tmp_tokenized_train_context_dataset[i]['attention_mask']]))

    tokenized_train_context_dataset = torch.stack(tokenized_train_context_dataset)

    train_datasets = TensorDataset(tokenized_train_query_dataset['input_ids'], tokenized_train_query_dataset['attention_mask'], tokenized_train_query_dataset['token_type_ids'],
                                   tokenized_train_context_dataset, torch.Tensor(targets).type(torch.LongTensor))

    Model = ColBERTModel.from_pretrained(MODEL_NAME)
    Model.to("cuda")

    # training

    # Dataloader
    train_sampler = RandomSampler(train_datasets)
    train_dataloader = DataLoader(train_datasets, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in Model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in Model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # train
    optimizer.zero_grad()
    Model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    batch_loss = 0

    for i in train_iterator:
        print(i)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    
        for step, batch in enumerate(epoch_iterator):
            Model.train()
    
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            q_inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]}

            p_inputs = {'input_ids': batch[3][:, 0, :, :].reshape(args.per_device_train_batch_size * 9, -1),
                        'attention_mask': batch[3][:, 1, :, :].reshape(args.per_device_train_batch_size * 9, -1),
                        'token_type_ids': batch[3][:, 2, :, :].reshape(args.per_device_train_batch_size * 9, -1)}
            
            q_outputs = Model.query(q_inputs['input_ids'], q_inputs['attention_mask'])
            p_outputs = Model.passage(p_inputs['input_ids'], p_inputs['attention_mask'])

            outputs = list()

            for b in range(args.per_device_train_batch_size):
                outputs.append(Model.get_score(q_outputs[b].reshape(1, 512, -1), p_outputs.reshape(args.per_device_train_batch_size, -1, 512, 128)[b]))
                
            outputs = torch.stack(outputs).squeeze()
            targets = batch[4]

            sim_scores = F.log_softmax(outputs, dim=1)

            if torch.cuda.is_available():
                targets = targets.to('cuda')

            loss = F.nll_loss(sim_scores, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            Model.zero_grad()
            batch_loss += loss.detach().cpu().numpy()

            torch.cuda.empty_cache()

    torch.save(Model.state_dict(), "/opt/ml/input/code/retrieval/Model.pt")

def validation():
    """ Dense Model 검증 함수
    """
    dataset = load_from_disk("/opt/ml/input/data/train_dataset")

    with open(os.path.join("/opt/ml/input/data/wikipedia_documents.json"), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    corpus = list(dict.fromkeys([v["text"] for v in wiki.values()])) 
    dev_dataset = dataset['validation']

    MODEL_NAME = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    Model = ColBERTModel.from_pretrained(MODEL_NAME)
    Model.load_state_dict(torch.load("/opt/ml/input/code/retrieval/Model.pt"))
    Model.to("cuda")

    with torch.no_grad():
        Model.eval()

        pickle_name = f"p_DenseEmbedding.bin"

        if os.path.isfile("/opt/ml/input/data/"+pickle_name):
            with open("/opt/ml/input/data/"+pickle_name, "rb") as file:
                corpus_embeddings = pickle.load(file)
        else:
            corpus_embeddings = list()
            
            for passage in tqdm(corpus):
                tokenized_passage = tokenizer(passage, stride=128, truncation=True, padding="max_length", return_tensors='pt').to('cuda')
                corpus_embedding = Model.passage(**tokenized_passage).detach().cpu().squeeze()
                corpus_embeddings.append(corpus_embedding)

            corpus_embeddings = torch.stack(corpus_embeddings)

            with open("/opt/ml/input/data/"+pickle_name, "wb") as file:
                pickle.dump(corpus_embeddings, file)

        # top ~ acc 구하기
        top_1 = 0
        top_3 = 0
        top_10 = 0
        top_25 = 0
        top_35 = 0
        top_100 = 0

        for sample_idx in tqdm(range(len(dev_dataset['question']))):
            tokenized_dev_query_dataset = tokenizer([dev_dataset[sample_idx]['question']], stride=128, truncation=True, padding="max_length", return_tensors='pt').to('cuda')
            query_embedding = Model.query(**tokenized_dev_query_dataset).detach().cpu()

            dot_prod_scores = Model.get_score(query_embedding, corpus_embeddings)
            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

            if sample_idx == rank[0]: 
                top_1 += 1
            if sample_idx in rank[0:3]: 
                top_3 += 1
            if sample_idx in rank[0:10]: 
                top_10 += 1
            if sample_idx in rank[0:25]: 
                top_25 += 1
            if sample_idx in rank[0:35]: 
                top_35 += 1
            if sample_idx in rank[0:100]: 
                top_100 += 1

        print('top-1 acc: ', top_1/240 * 100)
        print('top-3 acc: ', top_3/240 * 100)
        print('top-10 acc: ', top_10/240 * 100)
        print('top-25 acc: ', top_25/240 * 100)
        print('top-35 acc: ', top_35/240 * 100)
        print('top-100 acc: ', top_100/240 * 100)

if __name__ == "__main__":
    train()
    validation()