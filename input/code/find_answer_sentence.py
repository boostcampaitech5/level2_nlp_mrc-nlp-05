import pandas as pd

import transformers
import torch
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score

class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=2
        )
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.CrossEntropyLoss

    def forward(self, x):
        x = self.plm(**x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.squeeze())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.squeeze())
        self.log("val_loss", loss)
        preds = torch.argmax(logits, dim=1)
        
        self.log(
            "f1_score",
            f1_score(preds.cpu().numpy(), y.squeeze().cpu().numpy()),
        )
        
        self.log(
            "accuracy",
            accuracy_score(preds.cpu().numpy(), y.squeeze().cpu().numpy()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        
        self.log(
            "f1_score",
            f1_score(preds.cpu().numpy(), y.squeeze().cpu().numpy()),
        )
        
        self.log(
            "accuracy",
            accuracy_score(preds.cpu().numpy(), y.squeeze().cpu().numpy()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
 
 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return {
                "input_ids": self.inputs["input_ids"][idx],
                # "token_type_ids": self.inputs[idx]["token_type_ids"],
                "attention_mask": self.inputs["attention_mask"][idx],
            }
            
        else:
            return {
                "input_ids": self.inputs["input_ids"][idx],
                # "token_type_ids": self.inputs[idx]["token_type_ids"],
                "attention_mask": self.inputs["attention_mask"][idx],
            }, torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs['input_ids'])
       
    
class MyDataModule(pl.LightningDataModule):
    def __init__(self, data, batch, shuffle):
        super().__init__()
        self.data = Dataset(data)
        self.batch_size = batch
        self.shuffle = shuffle
        
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data, batch_size=self.batch_size, shuffle=self.shuffle,
        )    
    
    def __len__(self):
        return len(self.data)
    
class WikiInference():
    def __init__(self, model_path = "snunlp-KR-ELECTRA-discriminator-base.ckpt"):
        self.model = Model("snunlp/KR-ELECTRA-discriminator", 7e-6)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "snunlp/KR-ELECTRA-discriminator", max_length=200
        )
        self.trainer = pl.Trainer(
            accelerator="gpu", max_epochs=1, log_every_n_steps=1
        )
        
    def find_answer_sentence(self, df):
        new_df = pd.DataFrame(columns=['context', 'id', 'question', 'sentence_start', 'sentence_end', 'weights'])
        
        questions = []
        contexts = []
        len_list = []
        
        for _, row in df.iterrows():
            question = row['question']
            context = row['context'].split('.')
            for i in range(len(context)-1):
                context[i] = context[i] + '.'
            question = [question for _ in range(len(context))]
            questions += question
            contexts += context
            len_list.append(len(question))
           
        inputs = self.tokenizer(
            questions, contexts, add_special_tokens=True, padding="max_length", truncation=True, max_length=256, return_tensors="pt",
        )
            
        dataloader = MyDataModule(inputs, batch=64, shuffle=False)
        
        outputs = self.trainer.predict(model=self.model, datamodule=dataloader)
        if outputs[-1].dim() == 1:
            outputs[-1] = outputs[-1].unsqueeze(0)
            
        outputs = torch.cat(outputs) 
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        outputs = torch.softmax(outputs, dim=1).cpu().tolist()
             
        index = 0
        
        for i in range(len(len_list)):
            length = len_list[i]
            sentence_start, sentence_end = [], []
            weights = []
            total = 0
            
            for j in range(length):
                if preds[index+j] == 1:
                    sentence_start.append(total)
                    sentence_end.append(total+len(contexts[index+j]))
                    weights.append(outputs[index+j][1])
                total += len(contexts[index+j])
            index += length
            new_row = {'context': df.loc[i, 'context'], 'id': df.loc[i, 'id'], 'question' : df.loc[i, 'question'], 'sentence_start': sentence_start, 'sentence_end': sentence_end, 'weights': weights}
            new_df = new_df.append(new_row, ignore_index=True)
            
        return new_df
