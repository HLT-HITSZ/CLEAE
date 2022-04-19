import torch
import torch.nn as nn

class mBERTbase(nn.Module):
    def __init__(self, bert,role_vocab,argument_vocab,event_vocab,p = 0.5,device = "cpu",embedding_size = 128):
        super(mBERTbase, self).__init__()
        self.bert = bert
        self.device = device
        self.fc = nn.Linear(in_features = 2 * self.bert.config.hidden_size + 3*embedding_size, 
                                out_features = len(role_vocab))
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p = p)
        self.event_type_embedding = nn.Embedding(len(event_vocab),embedding_size)
        self.argument_type_embedding = nn.Embedding(len(argument_vocab),embedding_size)
        self.relative_pos_embedding = nn.Embedding(160,embedding_size)
    
    def forward(self,inputs,mode="train"):
        input_ids,token_type_ids,pad_seq_len,argument_type,event_type,trigger_mask,argument_mask,relative_pos,role = inputs
        
        sequence_out = self.bert(input_ids = input_ids, token_type_ids = token_type_ids)[-1][-1]
        sequence_out = self.dropout(sequence_out)

        triggers = torch.mul(sequence_out,trigger_mask.unsqueeze(-1)).float()
        triggers = torch.sum(triggers,dim = 1)
        trigger_lens = torch.sum(trigger_mask,dim = -1)
        triggers = triggers/trigger_lens.float().unsqueeze(-1)

        arguments = torch.mul(sequence_out,argument_mask.unsqueeze(-1)).float()
        arguments = torch.sum(arguments,dim = 1)
        argument_lens = torch.sum(argument_mask,dim = -1)
        arguments = arguments/argument_lens.float().unsqueeze(-1)

        event_embedding = self.event_type_embedding(event_type)
        argument_embedding = self.argument_type_embedding(argument_type)
        relative_pos_embedding = self.relative_pos_embedding(relative_pos)

        trigger_argument = torch.cat([triggers,arguments,event_embedding,argument_embedding,relative_pos_embedding],dim = -1)
        output = self.fc(trigger_argument)

        return self.loss(output,role),torch.argmax(output,dim = -1).cpu().numpy()
