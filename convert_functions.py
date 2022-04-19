def clip(x1,x2):
    dis = abs(x1-x2)
    if dis >= 100:
        return 100
    return dis
def mBERThidden_posemb_cut_cv_func(example, tokenizer, vocab = None,  max_seq_len=512, 
                    event_vocab=None,argument_vocab = None,role_vocab = None):
        role = example["role"]
        event_type = example["event_type"]
        argument_type = example["argument-type"]
        bert_tokens = example["bert_tokens"]
        bert_tokens = bert_tokens[:max_seq_len-2]

        bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        token_type_ids = [0]*len(input_ids)
        pad_seq_len = len(input_ids)
        
        bert_trigger_index_start = example["bert_trigger_index_start"]
        bert_trigger_index_end  = example["bert_trigger_index_end"]
        bert_argument_index_start  = example["bert_argument_index_start"]
        bert_argument_index_end  = example["bert_argument_index_end"]

        dis = clip(example["subj_start"],example["obj_start"])
        
        trigger_mask = []
        argument_mask = []
        for i in range(1,len(input_ids)-1):
            if bert_argument_index_start <= i <= bert_argument_index_end:
                argument_mask.append(1)
            else:
                argument_mask.append(0)
            if bert_trigger_index_start <= i <= bert_trigger_index_end:
                trigger_mask.append(1)
            else:
                trigger_mask.append(0)
        trigger_mask.append(0)
        argument_mask.append(0)

        if "test" not in example["type"]: 
            example["trans_bert_tokens"] = example["trans_bert_tokens"][:510]
            try:
                trans_bert_tokens = ["[CLS]"] + example["trans_bert_tokens"] + ["[SEP]"]
            except:
                print(example["text"])
                print(example["trans_bert_tokens"])
                exit()
            trans_input_ids = tokenizer.convert_tokens_to_ids(trans_bert_tokens)
            trans_token_type_ids = [0]*len(trans_input_ids)
            trans_pad_seq_len = len(trans_input_ids)
        else:
            trans_input_ids = [0]
            trans_token_type_ids = [0]
            trans_pad_seq_len = [0]

        return input_ids,token_type_ids,pad_seq_len,argument_vocab[argument_type],event_vocab[event_type],trigger_mask,argument_mask,\
           dis,trans_input_ids,trans_token_type_ids,trans_pad_seq_len,role_vocab[role]

def mBERTbase_cv_func(example, tokenizer, vocab = None,  max_seq_len=512, 
                    event_vocab=None,argument_vocab = None,role_vocab = None):
        role = example["role"]
        event_type = example["event_type"]
        argument_vocab = example["argument-type"]

        tokenized_input = tokenizer(
            example["tokens"],
            return_length=True,
            is_split_into_words=True,
            max_length=max_seq_len,
            truncation = True)
        input_ids = tokenized_input['input_ids']
        token_type_ids = tokenized_input['token_type_ids']
        pad_seq_len = tokenized_input['length']

        dis = clip(example["subj_start"],example["obj_start"])
        
        bert_trigger_index_start = example["bert_trigger_index_start"]
        bert_trigger_index_end  = example["bert_trigger_index_end"]
        bert_argument_index_start  = example["bert_argument_index_start"]
        bert_argument_index_end  = example["bert_argument_index_end"]
        
        trigger_mask = [0]
        argument_mask = [0]
        for i in range(1,len(input_ids)-1):
            if bert_argument_index_start <= i <= bert_argument_index_end:
                argument_mask.append(1)
            else:
                argument_mask.append(0)
            if bert_trigger_index_start <= i <= bert_trigger_index_end:
                trigger_mask.append(1)
            else:
                trigger_mask.append(0)
        argument_mask.append(0)
        trigger_mask.append(0)

        return input_ids,token_type_ids,pad_seq_len,argument_vocab[argument_type],event_vocab[event_type],trigger_mask,argument_mask,dis,role_vocab[role]