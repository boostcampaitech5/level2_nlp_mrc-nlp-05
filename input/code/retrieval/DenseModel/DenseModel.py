import torch
import torch.nn.functional as F
from transformers import BertPreTrainedModel, AutoModel, RobertaModel

class ColBERTModel(BertPreTrainedModel):
    def __init__(self, config):
        super(ColBERTModel, self).__init__(config)

        self.output_hidden_size = 128 # 768로 시도할 경우 에러 발생
        self.model = RobertaModel(config)
        self.projection = torch.nn.Linear(config.hidden_size, self.output_hidden_size, bias=False)

    def forward(self, p_inputs, q_inputs):
        Q = self.query(**q_inputs) # (batch_size, query_leng, hidden_size)
        P = self.passage(**p_inputs) # (batch_size, seq_leng, hidden_size) or (batch_size, seq, seq_leng, hidden_size)

        return self.get_score(Q, P)
    
    def query(self, input_ids, attention_mask, token_type_ids=None):
        Q = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        Q = self.projection(Q)

        return torch.nn.functional.normalize(Q, dim=2)
    
    def passage(self, input_ids, attention_mask, token_type_ids=None):
        P = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        P = self.projection(P)

        return torch.nn.functional.normalize(P, dim=2)
    
    def get_score(self, Q, P):
        batch_size = Q.shape[0]
        
        Q = Q.reshape(batch_size, 1, -1, self.output_hidden_size)
        P = P.transpose(1, 2)

        output = torch.matmul(Q, P) # (batch_size, batch_size, query_length, seq_length)
        output = torch.max(output, dim=3)[0] # (batch_size, batch_size, query_length)
        output = torch.sum(output, dim=2) # (batch_size, batch_size)

        return output
    
class BiEncoderModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BiEncoderModel, self).__init__(config)

        self.model = RobertaModel(config)

    def forward(self, p_inputs, q_inputs):
        Q = self.query(**q_inputs) # (batch_size, query_leng, hidden_size)
        P = self.passage(**p_inputs) # (batch_size, seq_leng, hidden_size) or (batch_size, seq, seq_leng, hidden_size)

        return self.get_score(Q, P)
    
    def query(self, input_ids, attention_mask, token_type_ids=None):
        Q = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output

        return torch.nn.functional.normalize(Q, dim=1)
    
    def passage(self, input_ids, attention_mask, token_type_ids=None):
        P = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output

        return torch.nn.functional.normalize(P, dim=1)
    
    def get_score(self, Q, P):
        output = torch.matmul(Q, P.T) # (batch_size, batch_size)

        return output