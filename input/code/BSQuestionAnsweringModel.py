import torch
import torch.nn as nn
from transformers import (
    BertPreTrainedModel,
    AutoModelForQuestionAnswering,
)

class BSQuestionAnsweringModel(BertPreTrainedModel):
    def __init__(self, model_name, from_tf, config):
        super().__init__(config)
        
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, from_tf=from_tf, config=config,
        )
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_start=None,
        sentence_end=None,
        start_positions=None,
        end_positions=None,
        weights=None,
    ):
        
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )
        
        if weights == None:  
            weight = torch.ones_like(outputs['start_logits'])
            for i in range(outputs['start_logits'].size(0)):
                weight[i, sentence_start[i]:sentence_end[i] + 1] = 1.1
                
            outputs['start_logits'] = outputs['start_logits'] * weight
            outputs['end_logits'] = outputs['end_logits'] * weight
            
        else:
            weight = torch.ones_like(outputs['start_logits'])
            for i in range(outputs['start_logits'].size(0)):
                for sentence_s, sentence_e, w in zip(sentence_start[i], sentence_end[i], weights[i]):
                    weight[i, sentence_s:sentence_e+1] = 1 + 0.1 * w
            outputs['start_logits'] = outputs['start_logits'] * weight
            outputs['end_logits'] = outputs['end_logits'] * weight
                    
        if (start_positions is not None) and (end_positions is not None):
            loss_fn = nn.CrossEntropyLoss()
            start_loss = loss_fn(outputs['start_logits'], start_positions)
            end_loss = loss_fn(outputs['end_logits'], end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs['loss'] = total_loss
            
        return outputs