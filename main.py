import argparse
import random
from functools import partial
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils import load_dict,ChunkEvaluator,to_device
from dataset import CLEARLDataset
from transformers import BertTokenizer, BertModel,BertConfig
from configuration import model_dict,convert_func_dict,batchify_fn_decorator,get_optim
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type = int, default = 300, help = "Number of epoches for fine-tuning.")
parser.add_argument("--train_data", type=str, default="./data/{}/train.pkl", help="train data")
parser.add_argument("--dev_data", type=str, default="./data/{}/dev.pkl", help="dev data")
parser.add_argument("--test_data", type=str, default="./data/{}_test/test.pkl", help="test data")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of tokens of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=100, help="validation step")
parser.add_argument("--skip_step", type=int, default=20, help="skip step")
parser.add_argument("--batch_size", type=int, default=8, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoints", type=str, default="./ckpt/{}", help="Directory to model checkpoint")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--cuda", default=0, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--event_vocab", type=str, default="./conf/event_vocab.txt")
parser.add_argument("--argument_vocab", type=str, default="./conf/argument_vocab.txt")
parser.add_argument("--role_vocab", type=str, default="./conf/role_vocab.txt")
parser.add_argument("--hidden_alignment", action='store_true', default=False)
parser.add_argument("--hard_alignment", action='store_true', default=False)
parser.add_argument("--model", type=str, default="mBERTbase")
parser.add_argument("--lang", type=str, default="en-zh")
parser.add_argument("--use_google", action='store_true', default=False)
parser.add_argument("--use_MS", action='store_true', default=False)
parser.add_argument("--multi", action='store_true', default=False)
parser.add_argument("--OT_threshold", type=float, default=0.3, help="Threshold for OT.")
args = parser.parse_args()

model_name = args.model
if not args.multi:
    args.train_data = args.train_data.format(args.lang)
    args.dev_data = args.dev_data.format(args.lang)
else:
    sources,target = args.lang.split("-")
    langs = ["{}-{}".format(sources[:2],target),"{}-{}".format(sources[2:],target)]
    args.train_data = [args.train_data.format(x) for x in langs]
    args.dev_data = [args.dev_data.format(x) for x in langs]

args.test_data = args.test_data.format(args.lang.split("-")[1])
args.checkpoints = args.checkpoints.format(args.lang)

if args.use_google:
    args.train_data = args.train_data.replace("train.pkl","google_train.pkl")
    args.dev_data = args.dev_data.replace("dev.pkl","google_dev.pkl")
elif args.use_MS:
    args.train_data = args.train_data.replace("train.pkl","MS_train.pkl")
    args.dev_data = args.dev_data.replace("dev.pkl","MS_dev.pkl")

bert_config = BertConfig.from_pretrained("./mBERT")
bert_config.output_hidden_states = True
pretrained_model = BertModel.from_pretrained('./mBERT',config = bert_config)
tokenizer = BertTokenizer.from_pretrained('./mBERT')

device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
vocab = tokenizer.get_vocab()
role_vocab = load_dict(args.role_vocab)
id2label = {val: key for key, val in role_vocab.items()}
event_vocab = load_dict(args.event_vocab)
argument_vocab = load_dict(args.argument_vocab)

trans_func = partial(
        convert_func_dict[model_name],
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        vocab = vocab,
        event_vocab = event_vocab,
        argument_vocab = argument_vocab,
        role_vocab = role_vocab,)

batchify_fn = batchify_fn_decorator(tokenizer,trans_func,model_name)
metric = ChunkEvaluator(label_list = list(role_vocab.values()), type = "micro",no_relation = role_vocab["no_relation"])

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate(model, metric, num_label, data_loader, device,get_results = False,mode="dev",output_file=None):
    if output_file != None:
        fout = open(output_file,"w")
    with torch.no_grad():
        results = []
        roles = []
        ids = []
        model.eval()
        metric.reset()
        losses = []
        for inputs in data_loader:
            inputs = to_device(inputs,device)
            loss, preds = model(inputs,mode=mode)
            role = inputs[-1]
            losses.append(loss.cpu().numpy())
            metric.update(preds,role.cpu().numpy())
            results.extend(preds.tolist())
            roles.extend(role.cpu().numpy().tolist())
        acc,f1,precision,recall,matrix = metric.compute()
        avg_loss = np.mean(losses)
        model.train()
    
    if output_file != None:
        for role,result in zip(roles,results):
            fout.write("{},{}\n".format(role,result))
        fout.close()
    if not get_results:
        return acc,f1,precision,recall,avg_loss
    return acc,f1,precision,recall,avg_loss,results,roles,matrix

def do_train():
    global config,pretrained_model,tokenizer,device
    global event_vocab,argument_vocab,vocab,role_vocab,id2label
    global trans_func,batchify_fn,metric
    local_rank = args.local_rank
    set_seed(args)
    model = model_dict[model_name](bert = pretrained_model,device = device,role_vocab = role_vocab,argument_vocab = argument_vocab,\
        event_vocab = event_vocab,OT_threshold=args.OT_threshold).to(device)
    
    print("============start train==========")

    train_ds = CLEARLDataset(args.train_data, shuffle = True,args = args)
    dev_ds = CLEARLDataset(args.dev_data,args = args)
    test_ds = CLEARLDataset(args.test_data,args = args,test = True)
    train_loader = DataLoader(
        dataset = train_ds, batch_size = args.batch_size,
        collate_fn = batchify_fn)
    dev_loader = DataLoader(
        dataset = dev_ds,
        batch_size = args.batch_size,
        collate_fn = batchify_fn)
    test_loader = DataLoader(
        dataset = test_ds,
        batch_size = args.batch_size,
        collate_fn = batchify_fn)

    num_training_steps = len(train_loader) * args.num_epoch
    optimizer = get_optim(model,model_name,args)
    
    step, best_f1 = 0,0.0
    model.train()
    patience = args.patience
    no_grow_setp = 0
    for epoch in range(args.num_epoch):
        flag = False
        since = time.time()
        for idx, inputs in enumerate(train_loader):
            inputs = to_device(inputs,device)
            loss, _ = model(inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item = loss.cpu().detach().numpy().item()
            if step > 0 and step % args.skip_step == 0 and local_rank == 0:
                print(f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}')
            if step > 0 and step % args.valid_step == 0 and local_rank == 0:
                acc,f1,precision,recall, avg_loss = evaluate(model, metric, len(role_vocab), dev_loader, device,mode="dev")
                print(f'dev step: {step} - loss: {avg_loss:.5f},acc:{acc:.5f}, precision: {precision:.5f}, recall: {recall:.5f}, ' \
                        f'f1: {f1:.5f} current best {best_f1:.5f}')
                if f1 > best_f1:
                    flag = True
                    best_f1 = f1
                    print(f'==============================================save best model ' \
                            f'best performerence {best_f1:5f}')
                    torch.save(model.state_dict(), '{}/best_model.bin'.format(args.checkpoints))
            step += 1

        time_elapsed = time.time() - since
        print('Epoch {} complete in {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60, time_elapsed % 60))
        if not flag:
            no_grow_setp += 1
        if no_grow_setp >= patience:
            print("early stop")
            break
    print("============start predict===============")
    state_dict = torch.load('{}/best_model.bin'.format(args.checkpoints), map_location='cuda:{}'.format(args.cuda))
    model.load_state_dict(state_dict)
    acc,f1,precision,recall, avg_loss = evaluate(model, metric, len(role_vocab), test_loader, device,mode="test")
    print(f'precision: {precision:.5f}, recall: {recall:.5f}, 'f'f1: {f1:.5f}')
    print("end")

def predict():
    print("============start predict===============")
    state_dict = torch.load('{}/best_model.bin'.format(args.checkpoints), map_location='cuda:{}'.format(args.cuda))
    global config,pretrained_model,tokenizer,device
    global event_vocab,argument_vocab,vocab,role_vocab,id2label
    global trans_func,batchify_fn,metric
    set_seed(args)
    model = model_dict[model_name](bert = pretrained_model,device = device,role_vocab = role_vocab,argument_vocab = argument_vocab,\
        event_vocab = event_vocab,OT_threshold=args.OT_threshold).to(device)
    model.load_state_dict(state_dict)
    test_ds = CLEARLDataset(args.test_data,args = args)
    test_loader = DataLoader(
        dataset = test_ds,
        batch_size = args.batch_size,
        collate_fn = batchify_fn)
    acc,f1,precision,recall, avg_loss = evaluate(model, metric, len(role_vocab), test_loader, device,mode="test",output_file = "predictions.txt")
    print(f'precision: {precision:.5f}, recall: {recall:.5f}, 'f'f1: {f1:.5f}')
    print("end")

if __name__ == '__main__':
    do_train()
    #predict()