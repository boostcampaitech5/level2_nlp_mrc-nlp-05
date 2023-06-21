from datasets import load_from_disk
from transformers import AutoModelForQuestionAnswering
import OmegaConf
from custom import NewModelwithLinear, NewModelwithReverseLSTM
import torch

args = OmegaConf.load("/opt/ml/args.yaml")

model_args, data_args = args.model, args.data
checkpoint = torch.load('/opt/ml/models/train_dataset/pytorch_model.bin')
if model_args.Custom_model == False:
    model=NewModelwithReverseLSTM(model_name=model_args.model_name)
else :
    checkpoint = torch.load('/opt/ml/models/train_dataset/pytorch_model.bin')
    if model_args.Custom_model == "ReverseLSTM" :
        model=NewModelwithReverseLSTM(model_name=model_args.model_name)
        
    elif model_args.Custom_model == "Linear" :
        model=NewModelwithLinear(model_name=model_args.model_name)
    
    model.load_state_dict(checkpoint)


model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )