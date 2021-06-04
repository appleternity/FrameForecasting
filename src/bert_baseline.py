import os
import numpy as np
import math
from bookcorpus_data import load_text_data, load_text_data_long
from coda_data import load_text_data as coda_load_text_data
from util import *
import scipy.sparse as sp
import math
import copy
import argparse

from transformers import BertModel, BertTokenizerFast, BertConfig
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

from torchnlp.samplers import BucketBatchSampler

class BertRegressor(nn.Module):
    def __init__(self, output_size=1221, model=None, config=None):
        super(BertRegressor, self).__init__()

        self.output_size = output_size

        if model is None:
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.config = BertConfig.from_pretrained("bert-base-uncased")
            self.linear = nn.Linear(self.config.hidden_size, self.output_size)
        else:
            self.model = model
            self.config = config
            self.linear = nn.Linear(self.config.hidden_size, self.output_size)

    def forward(self, x):
        output = self.model(x)[0][:, 0, :]
        output = self.linear(output)
        return output

class StoryDataset(data.Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

        if self.y is None:
            self.get_item = self.get_item_without_y
        else:
            self.get_item = self.get_item_with_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item_with_y(self, index):
        return self.x[index], self.y[index].toarray()

    def get_item_without_y(self, index):
        return self.x[index]

def tokenizer(x):
    return x.split(" ")

def train(arg):
    version             = arg.version
    device              = arg.device 
    block               = arg.block
    batch_size          = arg.batch_size
    eval_batch_size     = arg.eval_batch_size
    epoch_num           = arg.epoch_num
    learning_rate       = arg.learning_rate
    early_stop_epoch    = arg.early_stop_epoch
    valid_sample_num    = arg.valid_sample_num
    train_sample_num    = arg.train_sample_num
    max_len             = arg.max_len
    POSSIBLE_BATCH_SIZE = arg.possible_batch_size

    # build collate_fn function
    def my_collate_fn(data):
        # x, pad_id = 1
        # bert, pad_id = 0, cls = 101, seq = 102
        length = max(d[0].shape[0] for d in data)
        length = min(max_len, length)
        x = np.empty([len(data), length+2], dtype=np.int64)
        x.fill(0)
        x[:, 0] = 101
        x[:, -1] = 102
        for i, d in enumerate(data):
            l = min(d[0].shape[0], max_len)
            x[i, 1:l+1] = d[0][-l:]
    
        y = np.vstack([d[1] for d in data])
    
        # turn to torch tensor
        x = torch.LongTensor(x)
        y = torch.FloatTensor(y)
    
        return x, y

    # load data
    dl_model_dir = os.path.join(model_dir, arg.data_name, "bert", version)
    data_cached_path = os.path.join(dl_model_dir, "data.h5")
    os.makedirs(dl_model_dir, exist_ok=True)
    print(f"output model and all the info to '{dl_model_dir}'")

    # save config
    with open(os.path.join(dl_model_dir, "config.json"), 'w', encoding='utf-8') as outfile:
        json.dump({
            "block": block,
            "batch_size": batch_size,
            "epoch_num": epoch_num,
            "learning_rate": learning_rate,
            "early_stop_epoch": early_stop_epoch,
            "valid_sample_num": valid_sample_num,
            "train_sample_num": train_sample_num,
            "max_len": max_len,
        }, outfile, indent=4)

    if arg.data_name == "bookcorpus":
        if arg.history == 1: 
            _, x_train, y_train = load_text_data(block=block, phase="train", target_model=arg.model_type, verbose=True)
            _, x_valid, y_valid = load_text_data(block=block, phase="valid", target_model=arg.model_type, verbose=True)
            _, x_test, y_test = load_text_data(block=block, phase="test", target_model=arg.model_type, verbose=True)
        else:
            _, x_train, y_train = load_text_data_long(block=block, phase="train", target_model=arg.model_type, verbose=True, history=arg.history)
            _, x_valid, y_valid = load_text_data_long(block=block, phase="valid", target_model=arg.model_type, verbose=True, history=arg.history)
            _, x_test, y_test = load_text_data_long(block=block, phase="test", target_model=arg.model_type, verbose=True, history=arg.history)

    elif arg.data_name == "coda19":
        _, x_train, y_train = coda_load_text_data(block=block, phase="train", target_model=arg.model_type, verbose=True)
        _, x_valid, y_valid = coda_load_text_data(block=block, phase="valid", target_model=arg.model_type, verbose=True)
        _, x_test, y_test = coda_load_text_data(block=block, phase="test", target_model=arg.model_type, verbose=True)
    else:
        print(f"{arg.data_name} not supported yet!")
        quit()

    if arg.downsample != -1:
        random_index = np.random.RandomState(5516).permutation(x_train.shape[0])[:arg.downsample]
        x_train, y_train = x_train[random_index], y_train[random_index]

    random_index = np.random.permutation(x_valid.shape[0])[:valid_sample_num]
    x_valid, y_valid = x_valid[random_index], y_valid[random_index]
    random_index = np.random.permutation(x_test.shape[0])[:]
    x_test, y_test = x_test[random_index], y_test[random_index]

    print("Train", x_train.shape, y_train.shape)
    print("Test", x_test.shape, y_test.shape)
    print("Valid", x_valid.shape, y_valid.shape)

    x_valid, x_test = x_valid.tolist(), x_test.tolist()
    
    validation = data.DataLoader(
        StoryDataset(x_valid, y_valid), 
        batch_sampler=BucketBatchSampler(
            torch.utils.data.sampler.SequentialSampler(x_valid), 
            batch_size=batch_size, 
            drop_last=True, 
            sort_key=lambda i: x_valid[i].shape[0], 
            bucket_size_multiplier=100
        ),
        num_workers=3,
        collate_fn=my_collate_fn,
    )
    testing = data.DataLoader(
        StoryDataset(x_test, y_test), 
        batch_sampler=BucketBatchSampler(
            torch.utils.data.sampler.SequentialSampler(x_test), 
            batch_size=batch_size, 
            drop_last=True, 
            sort_key=lambda i: x_test[i].shape[0], 
            bucket_size_multiplier=100
        ),
        num_workers=3,
        collate_fn=my_collate_fn,
    )

    if arg.model_type == "bert":
        model = BertRegressor(output_size=y_train.shape[1])
    elif arg.model_type == "scibert":
        pretrained_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        pretrained_config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased")
        model = BertRegressor(
            output_size=y_train.shape[1], 
            model=pretrained_model, 
            config=pretrained_config
        )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = lambda y_pred, y_batch: 1-F.cosine_similarity(y_pred, y_batch).mean()
   
    best_epoch = 0
    best_cosine = 0.0
    stopper = EarlyStop(mode="max", history=early_stop_epoch)

    for epoch in range(1, epoch_num+1):
        # generate data
        if arg.downsample == -1 or arg.downsample > train_sample_num:
            random_index = np.random.permutation(x_train.shape[0])[:train_sample_num]
            x_train_epoch, y_train_epoch = x_train[random_index], y_train[random_index]
            x_train_epoch = x_train_epoch.tolist()
        else:
            x_train_epoch, y_train_epoch = x_train, y_train
            x_train_epoch = x_train_epoch.tolist()

        training = data.DataLoader(
            StoryDataset(x_train_epoch, y_train_epoch), 
            batch_sampler=BucketBatchSampler(
                torch.utils.data.sampler.SequentialSampler(x_train_epoch), 
                batch_size=batch_size if POSSIBLE_BATCH_SIZE == -1 else POSSIBLE_BATCH_SIZE, 
                drop_last=True, 
                sort_key=lambda i: x_train_epoch[i].shape[0], 
                bucket_size_multiplier=100
            ),
            num_workers=3,
            collate_fn=my_collate_fn,
        )
       
        # training
        model.train()
        total_loss = 0
        total_acc = 0
        total_count = len(training.dataset) // training.batch_sampler.batch_size
        error_case = 0
        if POSSIBLE_BATCH_SIZE != -1:
            accumulation_steps = batch_size//POSSIBLE_BATCH_SIZE
        for count, (x_batch, y_batch) in enumerate(training, 1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            try:
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch)
                loss.backward()
                total_loss += loss.item()

                if POSSIBLE_BATCH_SIZE == -1:
                    optimizer.step()
                    optimizer.zero_grad()
                elif count % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            except RuntimeError:
                #print(x_batch.shape, y_batch.shape)
                error_case += 1
                continue

            # compute cosine
            total_acc += F.cosine_similarity(y_pred, y_batch, dim=1).mean().item()
    
            print("\x1b[2K\rEpoch: {} / {} [{:.2f}%] Loss: {:.5f} Acc: {:.5f} Error: {}".format(
                epoch, epoch_num, 100.0*count/total_count, total_loss/count, total_acc/count, error_case), end="")

        print()
        if epoch % 1 == 0:
            cosine = evaluate(model, validation, device=device)

            if cosine > best_cosine:
                best_model = copy.deepcopy(model.state_dict())
                best_cosine = cosine
                best_epoch = epoch

            # check early stopping
            if stopper.check(cosine):
                print("Early Stopping at Epoch = ", epoch)
                break

    # finish training
    print("Loading model from epoch {}".format(best_epoch))
    torch.save(best_model, os.path.join(dl_model_dir, "best_model.pt"))
    model.load_state_dict(best_model)
    test_cosine = evaluate(model, testing, device)
    print("Testing Cosine = ", test_cosine)    
    with open(os.path.join(dl_model_dir, "result.json"), 'w', encoding='utf-8') as outfile:
        json.dump({
            "cosine": float(test_cosine),
            "best_cosine": float(best_cosine),
            "best_epoch": best_epoch,
            "max_len": max_len,
        }, outfile, indent=4)

def test():
    from tfidf_util import tfidf_metric_detail
    from util import h5_save

    # setup
    data_name="bookcorpus"
    model_name="bert"
    block=150
    batch_size=256
    max_len=500
    dl_model_dir=os.path.join(model_dir, "bookcorpus/longformer/bert_max500_150")

    # load data
    _, x_test, y_test = load_text_data(block=block, phase="test", target_model="bert", verbose=True)

    def my_collate_fn(data):
        # x, pad_id = 1
        # bert, pad_id = 0, cls = 101, seq = 102
        length = max(d[0].shape[0] for d in data)
        length = min(max_len, length)
        x = np.empty([len(data), length+2], dtype=np.int64)
        x.fill(0)
        x[:, 0] = 101
        x[:, -1] = 102
        for i, d in enumerate(data):
            l = min(d[0].shape[0], max_len)
            x[i, 1:l+1] = d[0][-l:]
    
        y = np.vstack([d[1] for d in data])
    
        # turn to torch tensor
        x = torch.LongTensor(x)
        y = torch.FloatTensor(y)
    
        return x, y

    training = data.DataLoader(
        StoryDataset(x_test, y_test), 
        num_workers=3,
        collate_fn=my_collate_fn,
        shuffle=False,
    )

    # load model
    model = BertRegressor(output_size=1221).to("cuda:1")
    weights = torch.load(os.path.join(dl_model_dir, "best_model.pt"), map_location="cuda:1")
    model.load_state_dict(weights)
    cosine, _ = evaluate(model, training, device="cuda:1", detail=True)
    print(cosine)

    h5_save(
        os.path.join(predict_dir, f"{data_name}_tfidf_{model_name}_{block}.h5"),
        name="cosine",
        data=cosine,
    )

def generate_human_eval():
    # setup
    data_name="bookcorpus"
    model_name="bert"
    block=20
    batch_size=256
    max_len=500
    dl_model_dir=os.path.join(model_dir, "bookcorpus/longformer/bert_max500_150")

    # load data
    with open(os.path.join(data_dir, "bookcorpus", f"human_evaluation_data_{block}_filled_1.json"), 'r', encoding='utf-8') as infile:
        my_data = json.load(infile)
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    x = [np.array(tokenizer.encode(d["x_text"], add_special_tokens=False)) for d in my_data]

    def my_collate_fn(data):
        # x, pad_id = 1
        # bert, pad_id = 0, cls = 101, seq = 102
        length = max(d.shape[0] for d in data)
        length = min(max_len, length)
        x = np.empty([len(data), length+2], dtype=np.int64)
        x.fill(0)
        x[:, 0] = 101
        x[:, -1] = 102
        for i, d in enumerate(data):
            l = min(d.shape[0], max_len)
            x[i, 1:l+1] = d[-l:]
    
    
        # turn to torch tensor
        x = torch.LongTensor(x)
        y = torch.zeros([1221, 1])

        return x, y

    training = data.DataLoader(
        StoryDataset(x), 
        num_workers=3,
        collate_fn=my_collate_fn,
        shuffle=False,
    )

    # load model
    model = BertRegressor(output_size=1221).to("cuda:1")
    weights = torch.load(os.path.join(dl_model_dir, "best_model.pt"), map_location="cuda:1")
    model.load_state_dict(weights)
    _, prediction = evaluate(model, training, device="cuda:1", detail=True)
    print(prediction.shape)

    for d, yy in zip(my_data, prediction):
        d["BERT"] = yy.tolist()

    with open(os.path.join(data_dir, "bookcorpus", f"human_evaluation_data_{block}_filled_2.json"), 'w', encoding='utf-8') as outfile:
        json.dump(my_data, outfile, indent=2)

def evaluate(model, eval_data, device, detail=False):
    model.eval()
    total_count = len(eval_data.dataset) // eval_data.batch_sampler.batch_size
    total_loss = 0
    total_acc = 0
    cosine_list = []
    prediction = []
    with torch.no_grad():
        for count, (x_batch, y_batch) in enumerate(eval_data, 1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            prediction.append(y_pred.cpu().numpy())

            # compute accuracy
            cosine = F.cosine_similarity(y_pred, y_batch, dim=1)
            cosine_list.append(cosine.cpu().numpy())
            total_acc += cosine.mean().item()

            print("\x1b[2K\rEval [{:.2f}%] Cosine: {:.5f}".format(
                100.0*count/total_count, total_acc/count), end="")
     
    all_cosine = np.hstack(cosine_list)
    cosine = all_cosine.mean()
    prediction = np.vstack(prediction)
    print("\nCosine: {:.5f}\n".format(cosine))
    if detail:
        return all_cosine, prediction
    else:
        return cosine


def check_length(block=20):
    _, x_train, y_train = coda_load_text_data(block=block, phase="train", verbose=True)
    #_, x_valid, y_valid = load_text_data(block=block, phase="valid", verbose=True)
    #_, x_test, y_test = load_text_data(block=block, phase="test", verbose=True)
    
    length = x_train.apply(lambda x: len(x))
    for p in [0.5, 0.75, 0.9, 0.95]:
        print("{:.2f}% of data is less than {} tokens".format(p*100, length.quantile(p)))

def check_length_batch():
    #for block in [5, 10, 20, 50, 100, 150, 200]:
    for block in [1, 3, 5]:
        print("block num = {}".format(block))
        check_length(block)

def parse_args():
    parser = argparse.ArgumentParser(description="Bert Baseline.")
    parser.add_argument("--device", help="Device used for training", type=str, default="cuda:0")
    parser.add_argument("--version", help="Naming for version control", type=str, default="bert")
    parser.add_argument("--epoch_num", help="Numbers of training epochs", type=int, default=200)
    parser.add_argument("--block", help="Size of the story block", type=int, default=20)
    parser.add_argument("--batch_size", help="Training batch size", type=int, default=16)
    parser.add_argument("--eval_batch_size", help="Evaluating batch size", type=int, default=16)

    parser.add_argument("--learning_rate", help="Learning rate used for the adam optimizer", type=float, default=1e-5)
    parser.add_argument("--early_stop_epoch", help="Number of epochs for early stopping if there is no improvement", type=int, default=5)
    parser.add_argument("--valid_sample_num", help="Number of instances used for validation", type=int, default=20000)
    parser.add_argument("--train_sample_num", help="Number of instances used for training in each epoch", type=int, default=100000)
    parser.add_argument("--max_len", help="The maximum length of tokens", type=int, default=500)
    parser.add_argument("--data_name", help="Corpus used for training and testing", type=str, default="bookcorpus")
    parser.add_argument("--model_type", help="Pretrained model [bert / scibert]", type=str, default="bert")

    parser.add_argument("--downsample",  help="Downsampling size", type=int, default=-1)
    parser.add_argument("--possible_batch_size", dest="possible_batch_size", type=int, default=-1)
    parser.add_argument("--history", help="Number of past story blocks used for input", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)
    #test()
    #check_length_batch()

if __name__ == "__main__":
    main()



