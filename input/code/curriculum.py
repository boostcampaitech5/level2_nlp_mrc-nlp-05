import torch
from datasets import load_from_disk
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm

dataset = load_from_disk('/opt/ml/input/data/train_dataset')
df = dataset['train']
n = len(dataset['train'])
loss_func = torch.nn.CrossEntropyLoss()
loss_sort_list = []

model = AutoModelForQuestionAnswering.from_pretrained('klue/roberta-large')
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')


for i in tqdm(range(n)):
    start_idx = df['answers'][i]['answer_start'][0]
    end_idx = start_idx + len(df['answers'][i]['text'][0])
    start_tok_idx = 0
    end_tok_idx = 0
    
    tok = tokenizer(
        df['question'][i], 
        df['context'][i],
        padding='max_length',
        return_tensors='pt',
        stride=128,
        max_length=512,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )
    
    with torch.no_grad():
        output = model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask'])

    if len(tok['offset_mapping']) >= 2:
        loss_lst = []
        
        for j in range(len(tok['offset_mapping'])):
            cnt = 0
            start_tok_idx = 0
            end_tok_idx = 0
            
            for idx, lst in enumerate(tok['offset_mapping'][j]):
                if torch.allclose(lst, torch.tensor([0,0])):
                    cnt += 1
                    
                if cnt == 2:
                    if lst[0] <= start_idx < lst[1]:
                        start_tok_idx = idx
                    
                    if lst[0] <= end_idx < lst[1]:
                        end_tok_idx = idx
                
                elif cnt == 3:
                    break

            start_loss = loss_func(output['start_logits'][j].unsqueeze(0), torch.tensor([start_tok_idx]))
            end_loss = loss_func(output['end_logits'][j].unsqueeze(0), torch.tensor([end_tok_idx]))
            loss_lst.append(start_loss + end_loss)
        
        loss_sort_list.append((sum(loss_lst) / len(loss_lst), int(i)))

    else:
        cnt = 0
        
        for idx, lst in enumerate(tok['offset_mapping']):
            if torch.allclose(lst, torch.tensor([0, 0])):
                cnt += 1
                    
            if cnt == 2:
                if lst[0] <= start_idx < lst[1]:
                    start_tok_idx = idx
                    
                if lst[0] <= end_idx < lst[1]:
                    end_tok_idx = idx
                
            elif cnt == 3:
                break
            
        start_loss = loss_func(output['start_logits'], torch.tensor([start_tok_idx]))
        end_loss = loss_func(output['end_logits'], torch.tensor([end_tok_idx]))
        loss_sort_list.append((start_loss + end_loss, int(i)))
        
loss_sort_list.sort()

output = pd.DataFrame(loss_sort_list, columns=['loss','index'])
output.to_csv('/opt/ml/input/code/loss_sort_data/LossData.csv', index=False)

print('complete!')