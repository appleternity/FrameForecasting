import torch
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import ujson as json
from collections import Counter
import pickle
import h5py

from event_data import extract_tuple 
from config import *
from event_model import Seq2Frame

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import scipy.sparse as sp
import copy
import argparse
from util import EarlyStop, h5_load

class EventFrameDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.int64)
        print("x.shape", self.x.shape)
        
        self.y = y.astype(np.float32)
        print("y.shape", self.y.shape)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (
            self.x[index], 
            self.y[index].toarray(), 
        )

def load_data(phrase="train", block=20, note="no_subj", data_name="bookcorpus", history=1):
    print("Data Loading: phrase = {}, block = {}".format(phrase, block))
    
    if history == 1:
        y = sp.load_npz(os.path.join(data_dir, data_name, "event-{}_block{}_{}_y.npz".format(note, block, phrase)))
        with open(os.path.join(data_dir, data_name, "event-{}_block{}_{}_x.json".format(note, block, phrase)), 'r', encoding='utf-8') as infile:
            x = json.load(infile)
    else:
        y = sp.load_npz(os.path.join(data_dir, data_name, "event-{}_block{}_{}_h{}_y.npz".format(note, block, phrase, history)))
        with open(os.path.join(data_dir, data_name, "event-{}_block{}_{}_h{}_x.json".format(note, block, phrase, history)), 'r', encoding='utf-8') as infile:
            x = json.load(infile)


    print("x_event length", len(x))
    print("y_frame shape", y.shape)

    return x, y

def load_y(phrase="train", block=20, note="no_subj", data_name="bookcorpus"):
    y = sp.load_npz(
        os.path.join(data_dir, data_name, "event-{}_block{}_{}_y.npz".format(note, block, phrase))
    )
    return y

def build_vocab_source(data, min_count=5):
    word_counts = Counter()

    for events in data:
        for event in events:
            word_counts.update(event)

    # build mapping
    mapping = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    for word, count in word_counts.items():
        if count > min_count:
            mapping[word] = len(mapping)

    return mapping

def vectorize_x(data, mapping, max_len=4):
    unk_index = mapping["<UNK>"]
    processed_data = []
    for events in data:
        sequence = [
            mapping.get(element, unk_index)
            for event in events
            for element in event
        ]
        processed_data.append(sequence)

    len_stat = np.array([len(d) for d in processed_data])
    print("mean", len_stat.mean(), end=" ")
    print("std", len_stat.std())
    print("max", len_stat.max(), end=" ")
    print("min", len_stat.min())
    print("50%", np.percentile(len_stat, 50), end="  ")
    print("80%", np.percentile(len_stat, 80), end="  ")
    print("90%", np.percentile(len_stat, 90), end="  ")
    print("95%", np.percentile(len_stat, 95))

    # turn 2 matrix
    matrix = np.ones([len(processed_data), max_len], dtype=np.int32) * mapping["<PAD>"]
    for i, d in enumerate(processed_data):
        length = min(len(d), max_len)
        matrix[i, max_len-length:] = d[:length]

    return matrix

def textify_x(data, mapping):
    reverse_mapping = {v:k for k, v in mapping.items()}

    event_list = []
    for sample in data:
        event = []
        for i, element in enumerate(sample):
            event.append(reverse_mapping[element])
        event_list.append(event)
    return event_list

def textify_y(data, subj_mapping, verb_mapping, obj_mapping, mod_mapping):
    reverse_mapping_list = [
        {v:k for k, v in subj_mapping.items()},
        {v:k for k, v in verb_mapping.items()},
        {v:k for k, v in obj_mapping.items()},
        {v:k for k, v in mod_mapping.items()},
    ]
    event_list = []
    for sample in data:
        event = []
        for i, element in enumerate(sample):
            event.append(reverse_mapping_list[i%4][element])

            if i % 4 == 3:
                event_list.append(event)
                event = []

    return event_list

def parse_arg():
    parser = argparse.ArgumentParser(description="Event-Representation Baseline Training Script.")
    parser.add_argument("--device", help="Device used for training and evaluation", type=str, default="cuda:0")

    parser.add_argument("--epoch_num", help="Number of training epochs", type=int, default=200)
    parser.add_argument("--early_stop_epoch", help="Number of epochs for early stop when there is no improvement", type=int, default=3)
    parser.add_argument("--learning_rate", help="Learning rate for optimizer", type=float, default=3e-5)
    parser.add_argument("--batch_size", help="Batch size of the training process", type=int, default=64)
    parser.add_argument("--evaluation_batch_size", help="Evaluation batch size", type=int, default=32)
    parser.add_argument("--possible_batch_size", help="Maximum possible match size on your device", type=int, default=-1)
    parser.add_argument("--save_model_freq", help="Number of epochs for saving the model periodically", type=int, default=5)

    parser.add_argument("--hidden_size", help="Hidden size of the LSTM", type=int, default=512)
    parser.add_argument("--layer_num", help="Number of layers of the LSTM", type=int, default=5)
    parser.add_argument("--dropout_rate", help="Dropout rate of the model", type=float, default=0.05)

    parser.add_argument("--block_size", help="Story block size", type=int, default=20)
    #parser.add_argument("--max_len", help="Maximum length of the input string (to remove the extreme cases)", type=int, default=150)
    parser.add_argument("--data_name", help="Corpus used for training", type=str, default="bookcorpus")
    parser.add_argument("--downsample", help="Downsample size", type=int, default=-1)
    parser.add_argument("--exp_name", help="Experiment name for version control", type=str, default="")
    parser.add_argument("--history", help="Number of story blocks used for input", type=int, default=1)

    return parser.parse_args()

def train():
    arg = parse_arg()

    # hyper-parameter setting
    hidden_size         = arg.hidden_size
    epoch_num           = arg.epoch_num
    learning_rate       = arg.learning_rate
    batch_size          = arg.batch_size
    layer_num           = arg.layer_num
    save_model_freq     = arg.save_model_freq
    dropout_rate        = arg.dropout_rate
    block_size          = arg.block_size
    max_len             = 0 # temp value, we have an auto heuristic rule to decide this later
    note                = "no_subj"
    device              = arg.device
    evaluation_batch_size   = arg.evaluation_batch_size
    early_stop_epoch        = arg.early_stop_epoch
    POSSIBLE_BATCH_SIZE     = arg.possible_batch_size

    force_reload = False
    if arg.exp_name != "":
        version = "e2f_all_{}_{}_{}".format(arg.exp_name, block_size, note)
    else:
        if arg.history == 1:
            version = "e2f_all_{}_{}".format(block_size, note)
        else:
            version = "e2f_all_{}_{}_h{}".format(block_size, note, arg.history)

    stopper = EarlyStop(mode="max", history=early_stop_epoch)

    # set up folder
    model_folder = os.path.join(model_dir, arg.data_name, "event_rep_baseline", version)
    os.makedirs(model_folder, exist_ok=True)
    data_path = os.path.join(model_folder, "data.h5")
    print(f"output all the data and information to:\n{model_folder}\n")

    with open(os.path.join(model_folder, "parameters.json"), 'w', encoding='utf-8') as outfile:
        json.dump(vars(arg), outfile, indent=2)

    if not os.path.isfile(data_path) or force_reload:
        x, y = load_data("train", block=block_size, note=note, data_name=arg.data_name, history=arg.history)
        x_valid, y_valid = load_data("valid", block=block_size, note=note, data_name=arg.data_name, history=arg.history)
        x_test, y_test = load_data("test", block=block_size, note=note, data_name=arg.data_name, history=arg.history)

        x_length = np.array([len(xx) for xx in x])
        print("Train Token Length X = ", x_length.mean(), x_length.std(), np.median(x_length))
        print("Max length = {}, Min length = {}".format(x_length.max(), x_length.min()))
        for p in [75, 80, 90, 95, 98, 99, 99.5, 99.9]:
            print("Percentile {}% = {}".format(p, np.percentile(x_length, p)))

        # auto max_len
        max_len = int(np.percentile(x_length, 95)) * 4

        # build vocab
        vocab = build_vocab_source(x)
        save_dictionary(model_folder, vocab)

        # vectorize
        x = vectorize_x(x, vocab, max_len=max_len)
        x_valid = vectorize_x(x_valid, vocab, max_len=max_len)
        x_test = vectorize_x(x_test, vocab, max_len=max_len)

        with h5py.File(data_path, 'w') as outfile:
            outfile.create_dataset("x", data=x)
            outfile.create_dataset("x_valid", data=x_valid)
            outfile.create_dataset("x_test", data=x_test)
    else:
        vocab = load_dictionary(model_folder)
        x, x_valid, x_test = h5_load(
            data_path, 
            ["x", "x_valid", "x_test"], 
        )
        y = load_y("train", block_size, note=note, data_name=arg.data_name)
        y_valid = load_y("valid", block_size, note=note, data_name=arg.data_name)
        y_test = load_y("test", block_size, note=note, data_name=arg.data_name)

    if arg.downsample != -1:
        random_index = np.random.RandomState(5516).permutation(x.shape[0])[:arg.downsample]
        x, y = x[random_index], y[random_index]

    print("Information") 
    print("Train", x.shape, y.shape)
    print("Test", x_test.shape, y_test.shape)
    print("Valid", x_valid.shape, y_valid.shape)

    if POSSIBLE_BATCH_SIZE == -1:
        training = data.DataLoader(EventFrameDataset(x, y), batch_size=batch_size, shuffle=True, num_workers=2)
        validation = data.DataLoader(EventFrameDataset(x_valid, y_valid), batch_size=evaluation_batch_size, shuffle=False, num_workers=2)
        testing = data.DataLoader(EventFrameDataset(x_test, y_test), batch_size=evaluation_batch_size, shuffle=False, num_workers=2)
    else:
        training = data.DataLoader(EventFrameDataset(x, y), batch_size=POSSIBLE_BATCH_SIZE, shuffle=True, num_workers=2)
        validation = data.DataLoader(EventFrameDataset(x_valid, y_valid), batch_size=evaluation_batch_size, shuffle=False, num_workers=2)
        testing = data.DataLoader(EventFrameDataset(x_test, y_test), batch_size=evaluation_batch_size, shuffle=False, num_workers=2)

    model = Seq2Frame(
        len(vocab), hidden_size, y.shape[1],
        layer_num=layer_num,
        dropout_rate=dropout_rate,
        padding_index=vocab["<PAD>"],
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_function = lambda y_pred, y_batch: 1-F.cosine_similarity(y_pred, y_batch).mean()
    
    val_loss, val_acc = evaluate(model, validation)
    history = {
        "training_loss": [],
        "training_acc": [],
        "validation_loss": [],
        "validation_acc": []
    }

    best_score = 0.0
    best_model = copy.deepcopy(model.state_dict())
    best_epoch = 0
    
    if POSSIBLE_BATCH_SIZE != -1:
        accumulation_steps = batch_size//POSSIBLE_BATCH_SIZE

    for epoch in range(1, epoch_num+1):
        # train
        model.train()
        total_loss = 0
        total_acc = 0
        total_count = len(training.dataset) // batch_size

        for count, (x, y) in enumerate(training, 1):

            # optimize length
            x_index = torch.sum(x, dim=0).bool()
            x = x[:, x_index]
            y = y.squeeze()

            x = x.to(device)
            y = y.to(device)

            p = model(x)
            loss = loss_function(p, y)
            loss.backward()
            total_loss += loss.item()

            if POSSIBLE_BATCH_SIZE == -1:
                optimizer.step()
                optimizer.zero_grad()
            elif count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # compute cosine
            total_acc += F.cosine_similarity(y, p, dim=1).mean().item()

            print("\x1b[2K\rEpoch: {} / {} [{:.2f}%] Loss: {:.5f} Cosine: {:.5f}".format(
                epoch, epoch_num, 100.0*count/total_count, total_loss/count, total_acc/count), end="")

        print()
        train_loss, train_acc = total_loss/count, total_acc/count

        # validation
        val_loss, val_acc = evaluate(model, validation)
        
        # save model
        if epoch % save_model_freq == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_folder, "model_epoch_{}.cp".format(epoch))
            )

        # check best model
        if val_acc > best_score:
            best_score = val_acc
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        # check early stopping
        if stopper.check(val_acc):
            print("Early Stopping at Epoch = ", epoch)
            break

        # output history
        history["training_loss"].append(float(train_loss))
        history["training_acc"].append(float(train_acc))
        history["validation_loss"].append(float(val_loss))
        history["validation_acc"].append(float(val_acc))
        output_history(
            os.path.join(model_folder, "history.json"),
            history,
        )

    # testing
    print("loading model from epoch {}".format(best_epoch))
    torch.save(best_model, os.path.join(model_folder, "best_model.pt"))
    model.load_state_dict(best_model)
    test_loss, test_acc = evaluate(model, testing)
    print("Testing Cosine = ", test_acc)
    with open(os.path.join(model_folder, "result.json"), 'w', encoding='utf-8') as outfile:
        json.dump({
            "cosine": float(test_acc),
            "best_score": float(best_score),
            "best_epoch": best_epoch,
            "max_len": max_len,
        }, outfile, indent=4)

def compute_acc(y, p, ignore_index):
    acc = (y==p).double()[y!=ignore_index]
    return torch.mean(acc)

def save_dictionary(folder_path, vocab):
    with open(os.path.join(folder_path, "dictionary.json"), 'w', encoding='utf-8') as outfile:
        json.dump({
            "vocab": vocab,
        }, outfile, indent=4)

def load_dictionary(folder_path):
    with open(os.path.join(folder_path, "dictionary.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    return data["vocab"]

def output_history(filepath, history):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(history, outfile, indent=2)

def evaluate(model, data, loss_function=F.mse_loss, recored_data=False):
    device = "cuda:0"
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = len(data.dataset) // data.batch_size
    total_count = max(total_count, 1)
    cosine_list = []

    if recored_data:
        all_result = {
            "x":[],
            "y":[],
            "p":[],
        }
    with torch.no_grad():
        for count, (x, y) in enumerate(data, 1):
            # optimize length
            x_index = torch.sum(x, dim=0).bool()
            x = x[:, x_index]
            y = y.squeeze(dim=1)

            x = x.to(device)
            y = y.to(device)
            p = model(x)
            
            loss = loss_function(p, y) 
            total_loss += loss.item()

            # compute accuracy
            cosine = F.cosine_similarity(y, p, dim=1)
            cosine_list.append(cosine.cpu().numpy())
            total_acc += cosine.mean().item()
           
            if recored_data:
                p1 += offset[0]
                p2 += offset[1]
                p3 += offset[2]
                p4 += offset[3]
                p = np.vstack([p1.cpu(), p2.cpu(), p3.cpu(), p4.cpu()]).transpose()
                y = np.vstack([y1.cpu(), y2.cpu(), y3.cpu(), y4.cpu()]).transpose()

                all_result["x"].append(x.cpu().numpy())
                all_result["y"].append(y)
                all_result["p"].append(p)

                # output text
                p_text = textify_y(p, subj_mapping, verb_mapping, obj_mapping, mod_mapping)
                y_text = textify_y(y, subj_mapping, verb_mapping, obj_mapping, mod_mapping)
                x_text = textify_x(x.cpu().numpy(), vocab) 
                all_result["x_text"].append(x_text)
                all_result["y_text"].append(y_text)
                all_result["p_text"].append(p_text)

            print("\x1b[2K\rEval [{:.2f}%] Loss: {:.10f} ACC: {:.10f}".format(
                100.0*count/total_count, total_loss/count, total_acc/count), end="")
    print()

    cosine = np.hstack(cosine_list)
    cosine = cosine.mean()

    if recored_data:
        all_result["x"] = np.vstack(all_result["x"])
        all_result["y"] = np.vstack(all_result["y"])
        all_result["p"] = np.vstack(all_result["p"])
        all_result["x_text"] = [dd for d in all_result["x_text"] for dd in d]
        all_result["y_text"] = [dd for d in all_result["y_text"] for dd in d]
        all_result["p_text"] = [dd for d in all_result["p_text"] for dd in d]

        return total_loss/count, cosine, all_result
    else:
        return total_loss/count, cosine

def main():
    train()

if __name__ == "__main__":
    main()
