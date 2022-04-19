import torch
import torch.nn as nn
import torch.nn.functional as F


class mBERThidden4LossPosemb(nn.Module):
    def __init__(self, bert,role_vocab,argument_vocab,event_vocab,p = 0.5,device = "cpu",embedding_size = 128,OT_threshold=0.3):
        super(mBERThidden4LossPosemb, self).__init__()
        self.bert = bert
        self.device = device
        self.OT_threshold = OT_threshold
        self.fc = nn.Linear(in_features = 2 * self.bert.config.hidden_size + 3*embedding_size, 
                                out_features = len(role_vocab))
        self.dropout = nn.Dropout(p = p)
        self.event_type_embedding = nn.Embedding(len(event_vocab),embedding_size)
        self.argument_type_embedding = nn.Embedding(len(argument_vocab),embedding_size)
        self.relative_pos_embedding = nn.Embedding(101,embedding_size)
        self.loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def distributed_sinkhorn(self,out_, eps = 5e-2,max_iter = 3):
            res = []
            for out in out_:
                Q = torch.exp(-out / eps).t() 
                B = Q.shape[1]
                K = Q.shape[0] 

                sum_Q = torch.sum(Q)
                Q /= sum_Q

                for it in range(max_iter):
                    sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                    Q /= sum_of_rows
                    Q /= K
                    Q /= torch.sum(Q, dim=0, keepdim=True)
                    Q /= B

                Q *= B 
                res.append(Q.t().unsqueeze(0))
            return torch.cat(res,dim = 0)

    def OT_attention(self,values, queries,values_len,queries_len,threshold=0.3):
        '''
        values B*L1*h
        queries B*L2*h
        values_len B
        queries_len B
        outputs B*L2*h
        '''
        def normal_sim(sim,values_len,queries_len):
            mask = torch.ones_like(sim)
            for bt in range(len(values_len)):
                mask[bt,values_len[bt]:,:] = 0
                mask[bt,:,queries_len[bt]:] = 0
            mask2 = torch.zeros_like(sim) + 0.001
            sim = sim * mask + mask2
            sim = sim/torch.sum(sim,dim=2).unsqueeze(-1)*mask
            return sim

        SIM = []
        for batch in range((values.size(0))):
            SIM.append(F.cosine_similarity(values[batch].unsqueeze(0),queries[batch].unsqueeze(1),dim=2).T.unsqueeze(0))
        SIM = torch.cat(SIM,dim=0)
        S = SIM.clone()
        S = S.detach()
        C = 1 - S
        P = self.distributed_sinkhorn(C)
        
        ones = torch.ones_like(P)
        zeros = torch.zeros_like(P)
        threshold_mask = torch.where(P >= threshold,ones,zeros)

        total_mask = threshold_mask
        total_mask = torch.where(total_mask >= 1,ones,total_mask)

        P = torch.where(total_mask>0,ones,zeros)
        P = P.detach()

        loss3 = torch.mean(-P.flatten()*torch.log(torch.sigmoid(SIM).flatten()))
        loss3 = loss3 + torch.mean(-(1-P.flatten())*torch.log(1-torch.sigmoid(SIM).flatten()))

        SIM = torch.where(SIM>0,SIM,zeros)
        SIM = normal_sim(SIM,values_len,queries_len)
        output = torch.matmul(SIM.transpose(1, 2),values) #B*L2*h
        return output,loss3

    def forward(self, inputs,mode="train"):
        input_ids,token_type_ids,pad_seq_len,argument_type,event_type,trigger_mask,argument_mask,\
            relative_pos,trans_input_ids,trans_token_type_ids,trans_pad_seq_len,role = inputs
        
        output = self.bert(input_ids = input_ids, token_type_ids = token_type_ids)
        origin_sequence_out = output[-1][-1]
        transformer_output = origin_sequence_out[:,1:,:]

        triggers = torch.mul(transformer_output,trigger_mask.unsqueeze(-1)).float()
        triggers = torch.sum(triggers,dim = 1)
        trigger_lens = torch.sum(trigger_mask,dim = -1)
        triggers = triggers/trigger_lens.float().unsqueeze(-1)

        arguments = torch.mul(transformer_output,argument_mask.unsqueeze(-1)).float()
        arguments = torch.sum(arguments,dim = 1)
        argument_lens = torch.sum(argument_mask,dim = -1)
        arguments = arguments/argument_lens.float().unsqueeze(-1)
        
        event_embedding = self.event_type_embedding(event_type)
        argument_embedding = self.argument_type_embedding(argument_type)
        relative_pos_embedding = self.relative_pos_embedding(relative_pos)
        
        if mode != "test":
            trans_output = self.bert(input_ids = trans_input_ids, token_type_ids = trans_token_type_ids)
            trans_sequence_out_ = trans_output[-1][-1]
            trans_transformer_output = trans_sequence_out_[:,1:,:]
            trans_sequence_out,loss3 = self.OT_attention(trans_transformer_output,transformer_output,trans_pad_seq_len,pad_seq_len)

            trans_triggers = torch.mul(trans_sequence_out,trigger_mask.unsqueeze(-1)).float()
            trans_triggers = torch.sum(trans_triggers,dim = 1)
            trans_triggers = trans_triggers/trigger_lens.float().unsqueeze(-1)

            trans_arguments = torch.mul(trans_sequence_out,argument_mask.unsqueeze(-1)).float()
            trans_arguments = torch.sum(trans_arguments,dim = 1)
            trans_arguments = trans_arguments/argument_lens.float().unsqueeze(-1)

            trans_trigger_argument = torch.cat([trans_triggers,trans_arguments,event_embedding,argument_embedding,relative_pos_embedding],dim = -1)
            trans_output = self.fc(trans_trigger_argument)
            loss2 = self.loss(trans_output,role)

        trigger_argument = torch.cat([triggers,arguments,event_embedding,argument_embedding,relative_pos_embedding],dim = -1)
        output = self.fc(trigger_argument)
        loss1 = self.loss(output,role)

        if mode == "train":
            return loss1 + loss2 + 0.5*loss3,None
        #if mode == "dev":
        #    return loss2,torch.argmax(trans_output,dim = -1).cpu().numpy()
        return loss1,torch.argmax(output,dim = -1).cpu().numpy()