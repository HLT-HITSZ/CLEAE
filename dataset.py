from torch.utils.data import Dataset
from utils import load_dict,read_from_pickle
import numpy as np

class CLEARLDataset(Dataset):
    def __init__(self, data_path,args,shuffle = False,max_seq_len=512,test = False):
        self.max_seq_len = max_seq_len
        self.args = args
        self.data = []
        if args.multi and not test:
            inputs = []
            for path in data_path:
                inputs = inputs + read_from_pickle(path)
        else:
            inputs = read_from_pickle(data_path)

        if test:
            self.data = self._filter_test_data(inputs)

        else:
            if self.args.hard_alignment:
                self.data = self.data + self._filter_origin_data(inputs)
            elif self.args.hidden_alignment:
                self.data = self.data + self._filter_trans_data(inputs)


        if shuffle:
            def _shuffle(data,idx):
                new_data = []
                for idx_ in idx:
                    new_data.append(data[idx_])
                return new_data
            idx = np.arange(len(self.data))
            np.random.shuffle(idx)
            idx = idx.tolist()
            self.data = _shuffle(self.data,idx)

    def _get_trigger_argument_helper(self,idxs):
        idxs = sorted(list(set(idxs)))
        res = [idxs[0]]
        for idx in idxs[1:]:
            while idx > 1 + res[-1]:
                res.append(res[-1]+1)
            res.append(idx)
        return res

    def _get_index_start_end_helper(self,bert_tokens,bert_alignment,argument_index,trigger_index):
        lens = self.max_seq_len-2
        bert_argument_index,bert_trigger_index = [],[]
        for idx in argument_index:
            for i,_ in bert_alignment[idx]:
                if i >= lens:
                    continue
                bert_argument_index.append(1 + i)
        if len(bert_argument_index) == 0:
            return None,None,None,None
        bert_argument_index_start,bert_argument_index_end = min(bert_argument_index),max(bert_argument_index)
        for idx in trigger_index:
            for i,_ in bert_alignment[(idx)]:
                if i >= lens:
                    continue
                bert_trigger_index.append(1 + i)
        if len(bert_trigger_index) == 0:
            return None,None,None,None
        bert_trigger_index_start,bert_trigger_index_end = min(bert_trigger_index),max(bert_trigger_index)
        return bert_trigger_index_start,bert_trigger_index_end,bert_argument_index_start,bert_argument_index_end

    def _filter_test_data(self,dataset):
        filtered_dataset = []
        for data in dataset:
            if "zh_trans_bert_alignment_info" in data and data["zh_trans_bert_alignment_info"] == None:
                continue
            new_data = {}
            new_data["type"] = "test"
            new_data["text"] = data["text"]
            try:
                new_data["trigger_index"] = self._get_trigger_argument_helper([data["subj_start"],data["subj_end"]])
                new_data["argument_index"] = self._get_trigger_argument_helper([data["obj_start"],data["obj_end"]])
            except:
                print(data["id"])
                exit()
            new_data["tokens"] = data["token"]
            new_data["role"] = data["relation"]
            new_data["event_type"] = data["subj_type"]
            new_data["entity-type"] = data["obj_type"]
            new_data["bert_alignment"] = data["zh_trans_bert_alignment_info"]
            new_data["bert_tokens"] = data["zh_trans_bert_tokens"]

            try:
                new_data["bert_trigger_index_start"],new_data["bert_trigger_index_end"],\
                        new_data["bert_argument_index_start"],new_data["bert_argument_index_end"] = self._get_index_start_end_helper(
                            new_data["bert_tokens"],new_data["bert_alignment"],new_data["argument_index"],new_data["trigger_index"]
                            )
            except:
                print(data["id"])
                exit()
            
            if new_data["bert_trigger_index_start"] == None:
                continue
            
            new_data["subj_start"],new_data["subj_end"] = data["subj_start"],data["subj_end"]
            new_data["obj_start"],new_data["obj_end"] = data["obj_start"],data["obj_end"]

            filtered_dataset.append(new_data)
        return filtered_dataset

    def _filter_trans_data(self,dataset):
        filtered_dataset = []
        for data in dataset:
            if data["bert_alignment"] == None:
                continue
            if data["zh_trans_bert_tokens"] == None:
                continue
            new_data = {}
            new_data["type"] = "trans"
            new_data["trans_tokens"] = data["zh_trans_token"]
            new_data["role"] = data["relation"]
            new_data["event_type"] = data["subj_type"]
            new_data["entity-type"] = data["obj_type"]
            new_data["stanford_pos"] = data["stanford_pos"]
            new_data["stanford_ner"] = data["stanford_ner"]
            new_data["bert_alignment"] = data["bert_alignment"]
            new_data["bert_tokens"] = data["bert_tokens"]
            try:
                new_data["trigger_index"] = self._get_trigger_argument_helper([data["subj_start"],data["subj_end"]])
                new_data["argument_index"] = self._get_trigger_argument_helper([data["obj_start"],data["obj_end"]])
            except:
                print(data["id"])
                exit()
            new_data["tokens"] = data["token"]
            new_data["trans_bert_tokens"] = data["zh_trans_bert_tokens"]

            new_data["bert_trigger_index_start"],new_data["bert_trigger_index_end"],\
                new_data["bert_argument_index_start"],new_data["bert_argument_index_end"] = self._get_index_start_end_helper(
                    new_data["bert_tokens"],new_data["bert_alignment"],new_data["argument_index"],new_data["trigger_index"]
                    )
            if new_data["bert_argument_index_start"] == None:
                continue
            new_data["text"] = data["text"]
            new_data["trans"] = data["zh_trans"]
            new_data['id'] = data["id"]
            new_data["subj_start"],new_data["subj_end"] = data["subj_start"],data["subj_end"]
            new_data["obj_start"],new_data["obj_end"] = data["obj_start"],data["obj_end"]
            filtered_dataset.append(new_data)
        return filtered_dataset

    def _filter_origin_data(self,dataset):
        filtered_dataset = []
        for data in dataset:
            if data["bert_tokens"] == None:
                continue
            new_data = {}
            new_data["text"] = data["zh_trans"]
            new_data["tokens"] = data["zh_trans_token"]
            new_data["type"] = "awesome"
            new_data["role"] = data["relation"]
            new_data["event_type"] = data["subj_type"]
            new_data["entity-type"] = data["obj_type"]
            new_data["bert_tokens"] = data["bert_tokens"]
            new_data["bert_alignment"] = data["bert_alignment"]
            new_data["subj_start"] = data["subj_start"]
            new_data["subj_end"] = data["subj_end"]
            new_data["obj_start"] = data["obj_start"]
            new_data["obj_end"] = data["obj_end"]
            new_data["trigger_index"] = self._get_trigger_argument_helper([data["subj_start"],data["subj_end"]])
            new_data["argument_index"] = self._get_trigger_argument_helper([data["obj_start"],data["obj_end"]])
            new_data["bert_trigger_index_start"],new_data["bert_trigger_index_end"],\
                new_data["bert_argument_index_start"],new_data["bert_argument_index_end"] = self._get_index_start_end_helper(
                    new_data["bert_tokens"],new_data["bert_alignment"],new_data["argument_index"],new_data["trigger_index"]
                    )
            if new_data["bert_trigger_index_start"] == None:
                continue
            filtered_dataset.append(new_data)
        return filtered_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]