from convert_functions import mBERTbase_cv_func,mBERThidden_posemb_cut_cv_func
from models import mBERTbase
from models import mBERThidden4LossPosemb
import torch
from utils import Stack, Pad, Tuple

model_dict = {
    "mBERTbase":mBERTbase,
    "4losspos":mBERThidden4LossPosemb,
}
convert_func_dict = {
    "mBERTbase":mBERTbase_cv_func,
    "4losspos":mBERThidden_posemb_cut_cv_func,
}

def batchify_fn_decorator(tokenizer,trans_func,model_name):
    if model_name ==  "mBERTbase":
        return  lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),
            Stack(dtype='int64'),
            Stack(dtype='int64'),
            Stack(dtype='int64'),
            Pad(axis=0, pad_val=0, dtype='int32'),
            Pad(axis=0, pad_val=0, dtype='int32'),
            Stack(dtype='int64'),
            Stack(dtype='int64'),
        ): fn(list(map(trans_func, samples)))
    elif model_name == "4losspos":
        return lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),
            Stack(dtype='int64'),
            Stack(dtype='int64'),
            Stack(dtype='int64'),
            Pad(axis=0, pad_val=0, dtype='int32'),
            Pad(axis=0, pad_val=0, dtype='int32'),
            Stack(dtype='int64'),
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),
            Stack(dtype='int64'),
            Stack(dtype='int64'),
        ): fn(list(map(trans_func, samples)))
    return None
    

def get_optim(model,model_name,args):
    if model_name == "mBERTbase":
            return torch.optim.AdamW(
            [
                {'params':model.bert.parameters(),'lr':1e-5},
                {'params':model.event_type_embedding.parameters(),'lr':5e-3},
                {'params':model.argument_type_embedding.parameters(),'lr':5e-3},
                {'params':model.relative_pos_embedding.parameters(),'lr':5e-3},
            ],
            lr = 1e-3,
            weight_decay = args.weight_decay)
    elif model_name =="4losspos":
            return torch.optim.AdamW(
            [
                {'params':model.bert.parameters(),'lr':5e-6},
                {'params':model.event_type_embedding.parameters(),'lr':5e-3},
                {'params':model.argument_type_embedding.parameters(),'lr':5e-3},
                {'params':model.relative_pos_embedding.parameters(),'lr':5e-3},
            ],
            lr = 1e-3,
            weight_decay = args.weight_decay)
    return None