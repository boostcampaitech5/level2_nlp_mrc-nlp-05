from transformers import AutoModel
import torch

class SequentialLSTM(torch.nn.Module):
    def __init__(self,hidden_size) :
        super().__init__()
        self.model=torch.nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             batch_first=True,
                             bidirectional=True)
  
    def forward(self,x):
        output, _= self.model(x)
        return output
    
class NewModel(torch.nn.Module) :
    def __init__(self,model_name,config) :
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.config = config
        self.additional_layer=torch.nn.Sequential(SequentialLSTM(hidden_size=config.hidden_size),
                                              torch.nn.ReLU(),
                                              torch.nn.Dropout(p=0.1),
                                              torch.nn.Linear(self.config.hidden_size*2,2))
  
    def forward(self,input_ids,attention_mask,token_type_ids=None) :
        output=self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).last_hidden_state
        real_output=self.additional_layer(output)
        output_dic = {'start_logits' : real_output[:,:,0], 'end_logits':real_output[:,:,1]}
        return output_dic