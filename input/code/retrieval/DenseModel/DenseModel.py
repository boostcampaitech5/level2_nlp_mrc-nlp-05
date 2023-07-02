import torch
import torch.nn.functional as F
from transformers import BertPreTrainedModel, AutoModel, RobertaModel

class ColBERTModel(BertPreTrainedModel):
    """ Colbert Model Class

    Args:
        BertPreTrainedModel (BertPreTrainedModel): Model Class를 상속받기 위한 인자
    """
    def __init__(self, config):
        """ Model 생성자

        Args:
            config (config): Model의 config
        """
        super(ColBERTModel, self).__init__(config)

        self.output_hidden_size = 128 # 768로 시도할 경우 에러 발생
        self.model = RobertaModel(config)
        self.projection = torch.nn.Linear(config.hidden_size, self.output_hidden_size, bias=False)

    def forward(self, p_inputs, q_inputs):
        """ Model의 forward 작업 함수

        Args:
            p_inputs (dict): Tokenized Passage 입력 값
            q_inputs (dict): Tokenized Query 입력 값

        Returns:
            torch.tensor: Query와 Passage간의 유사도
        """
        Q = self.query(**q_inputs) # (batch_size, query_leng, hidden_size)
        P = self.passage(**p_inputs) # (batch_size, seq_leng, hidden_size) or (batch_size, seq, seq_leng, hidden_size)

        return self.get_score(Q, P)
    
    def query(self, input_ids, attention_mask, token_type_ids=None):
        """ Query가 입력으로 들어왔을 때 동작하는 함수

        Args:
            input_ids (tensor): Tokenized Query의 input_ids
            attention_mask (tensor): Tokenized Query의 attention_mask
            token_type_ids (tensor, optional): Tokenized Query의 token_type_ids. Defaults to None.

        Returns:
            torch.tensor: Query의 last hidden state
        """
        Q = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        Q = self.projection(Q)

        return torch.nn.functional.normalize(Q, dim=2)
    
    def passage(self, input_ids, attention_mask, token_type_ids=None):
        """ Passage가 입력으로 들어왔을 때 동작하는 함수

        Args:
            input_ids (tensor): Tokenized Passage의 input_ids
            attention_mask (tensor): Tokenized Passage의 attention_mask
            token_type_ids (tensor, optional): Tokenized Passage의 token_type_ids. Defaults to None.

        Returns:
            torch.tensor: Tokenized Passage의 last hidden state
        """
        P = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        P = self.projection(P)

        return torch.nn.functional.normalize(P, dim=2)
    
    def get_score(self, Q, P):
        """ Query와 Passage의 last hidden state를 이용하여 유사도를 계산하는 함수

        Args:
            Q (tensor): Query의 last hidden state
            P (tensor): Tokenized Passage의 last hidden state

        Returns:
            torch.tensor: Query와 Passage간의 유사도
        """
        batch_size = Q.shape[0]
        
        Q = Q.reshape(batch_size, 1, -1, self.output_hidden_size)
        P = P.transpose(1, 2)

        output = torch.matmul(Q, P) # (batch_size, batch_size, query_length, seq_length)
        output = torch.max(output, dim=3)[0] # (batch_size, batch_size, query_length)
        output = torch.sum(output, dim=2) # (batch_size, batch_size)

        return output
    
class BiEncoderModel(BertPreTrainedModel):
    """ BiEncoder Model Class

    Args:
        BertPreTrainedModel (BertPreTrainedModel): Model Class를 상속받기 위한 인자
    """
    def __init__(self, config):
        """ Model 생성자

        Args:
            config (config): Model의 config
        """
        super(BiEncoderModel, self).__init__(config)

        self.model = RobertaModel(config)

    def forward(self, p_inputs, q_inputs):
        """ Model의 forward 작업 함수

        Args:
            p_inputs (dict): Tokenized Passage 입력 값
            q_inputs (dict): Tokenized Query 입력 값

        Returns:
            torch.tensor: Query와 Passage간의 유사도
        """
        Q = self.query(**q_inputs) # (batch_size, hidden_size)
        P = self.passage(**p_inputs) # (batch_size, hidden_size)

        return self.get_score(Q, P)
    
    def query(self, input_ids, attention_mask, token_type_ids=None):
        """ Query가 입력으로 들어왔을 때 동작하는 함수

        Args:
            input_ids (tensor): Tokenized Query의 input_ids
            attention_mask (tensor): Tokenized Query의 attention_mask
            token_type_ids (tensor, optional): Tokenized Query의 token_type_ids. Defaults to None.

        Returns:
            torch.tensor: Query의 last hidden state
        """
        Q = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output

        return torch.nn.functional.normalize(Q, dim=1)
    
    def passage(self, input_ids, attention_mask, token_type_ids=None):
        """ Passage가 입력으로 들어왔을 때 동작하는 함수

        Args:
            input_ids (tensor): Tokenized Passage의 input_ids
            attention_mask (tensor): Tokenized Passage의 attention_mask
            token_type_ids (tensor, optional): Tokenized Passage의 token_type_ids. Defaults to None.

        Returns:
            torch.tensor: Tokenized Passage의 last hidden state
        """
        P = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output

        return torch.nn.functional.normalize(P, dim=1)
    
    def get_score(self, Q, P):
        """ Query와 Passage의 last hidden state를 이용하여 유사도를 계산하는 함수

        Args:
            Q (tensor): Query의 last hidden state
            P (tensor): Tokenized Passage의 last hidden state

        Returns:
            torch.tensor: Query와 Passage간의 유사도
        """
        output = torch.matmul(Q, torch.transpose(P, 0, 1)) # (batch_size, batch_size)

        return output