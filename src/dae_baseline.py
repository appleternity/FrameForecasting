import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data

from config import *
from bookcorpus_data import load_tfidf, load_tfidf_long, tokenizer
from coda_data import load_tfidf as coda_load_tfidf
from tfidf_util import tfidf_metric, print_tfidf_metric
from autoencoder import AutoEncoder
import copy
from util import *
from datetime import datetime
import argparse

class TFIDF_Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index].toarray(), self.y[index].toarray()

def train(block=200, data_name="bookcorpus", downsample=-1, dropout_rate=0.2, history=None, device="cuda:0", params=None):
    # version 1 - tfidf as feature
    if data_name == "bookcorpus":
        if history is None:
            x_train, y_train = load_tfidf("train", block, verbose=True, redo=False)
            x_test, y_test = load_tfidf("test", block, verbose=True, redo=False)
            x_valid, y_valid = load_tfidf("valid", block, verbose=True, redo=False)
        else:
            x_train, y_train = load_tfidf_long("train", block, verbose=True, redo=False, history=history)
            x_test, y_test = load_tfidf_long("test", block, verbose=True, redo=False, history=history)
            x_valid, y_valid = load_tfidf_long("valid", block, verbose=True, redo=False, history=history)

    elif data_name == "coda19":
        x_train, y_train = coda_load_tfidf("train", block, verbose=True, redo=False)
        x_test, y_test = coda_load_tfidf("test", block, verbose=True, redo=False)
        x_valid, y_valid = coda_load_tfidf("valid", block, verbose=True, redo=False)
    else:
        print("Not supported yet")
        quit()

    if downsample != -1:
        random_index = np.random.RandomState(5516).permutation(x_train.shape[0])[:downsample]
        x_train, y_train = x_train[random_index], y_train[random_index]

    # parameter setting
    vocab_size      = x_train.shape[1]
    output_size     = y_train.shape[1]
    hidden_size     = 512 if params is None else params.hidden_size
    epoch_num       = 2000 if params is None else params.epoch_num
    batch_size      = 512 if params is None else params.batch_size
    layer_num       = 5 if params is None else params.layer_num
    learning_rate   = 1e-4 if params is None else params.learning_rate
    early_stop_epoch = 20 if params is None else params.early_stop
    device          = device

    if downsample == -1:
        note = f"cosine - auto2 - {dropout_rate}"
    else:
        note = f"cosine - auto2 - {dropout_rate} - downsample"

    # build dataset
    training_dataset = TFIDF_Dataset(x_train, y_train)
    training = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testing_dataset = TFIDF_Dataset(x_test, y_test)
    testing = data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    valid_dataset = TFIDF_Dataset(x_valid, y_valid)
    valid = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # build model
    model = AutoEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_rate=dropout_rate,
        device=device,
        layer_num=layer_num,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = lambda y_pred, y_batch: 1-F.cosine_similarity(y_pred, y_batch).mean()

    # first evaluation
    evaluate(model, valid, loss_function=loss_function)

    best_epoch = 0
    best_cosine = 0
    best_model = copy.deepcopy(model.state_dict())
    stopper = EarlyStop(mode="max", history=early_stop_epoch)

    # train model
    for epoch in range(1, epoch_num+1):
        # train
        model.train()
        total_loss = 0
        total_count = np.ceil(x_train.shape[0] / batch_size)
        total_cosine = 0

        for count, (x_batch, y_batch) in enumerate(training, 1):
            x_batch = x_batch.squeeze()
            y_batch = y_batch.squeeze()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            cosine = F.cosine_similarity(y_batch, y_pred)
            total_cosine += cosine.mean().item()

            print("\x1b[2K\rEpoch: {} / {} [{:.2f}%] Loss: {:.4f} Cosine: {:.4f}".format(
                epoch, epoch_num, 100.0*count/total_count, total_loss/count, total_cosine/count), end="")
        print()

        # valid
        if epoch % 1 == 0 or epoch==epoch_num:
            cosine, _ = evaluate(model, valid, loss_function=loss_function) 
            if cosine > best_cosine:
                best_model = copy.deepcopy(model.state_dict()) 
                best_epoch = epoch
                best_cosine = cosine
            
            # check early stopping
            if stopper.check(cosine):
                print("Early Stopping at Epoch = ", epoch)
                break

    # load best model & test & save
    print("loading model from epoch {}".format(best_epoch))
    torch.save(best_model, os.path.join(model_dir, data_name, "{}_autoencoder_{}.pt".format(note, block)))
    model.load_state_dict(best_model)
    cosine, y_pred = evaluate(model, testing, device=device, loss_function=loss_function)
    print("testing cosine:", cosine)

    # config filename
    if history is None:
        filename = os.path.join(result_dir, f"{data_name}_dl_baseline.json")
        prediction_filename = os.path.join(predict_dir, "bookcorpus", f"block{block}_autoencoder_{note.replace(' ', '')}.h5")
    else:
        filename = os.path.join(result_dir, f"history_exp_{data_name}_dl_baseline.json")
        prediction_filename = os.path.join(predict_dir, "bookcorpus", f"history_block{block}_autoencoder_{note.replace(' ', '')}.h5")

    print_tfidf_metric(
        {
            "cosine": float(cosine),
            "block": block,
            "model": "autoencoder",
            "note": "clean - autoencoder - tfidf - deep - {}".format(note)
        }, 
        filename=filename
    )
    
    save_prediction(prediction_filename, y_pred)

def evaluate(model, data, device="cuda:0", loss_function=F.mse_loss):
    model.eval()
    total_loss = 0
    total_cosine = 0
    total_count = np.ceil(len(data.dataset) / data.batch_size)
    cosine_list = []
    y_pred_list = []
    with torch.no_grad():
        for count, (x_batch, y_batch) in enumerate(data, 1):
            x_batch = x_batch.squeeze()
            y_batch = y_batch.squeeze()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            y_pred_list.append(y_pred.cpu().numpy())

            # loss
            loss = loss_function(y_pred, y_batch)
            total_loss += loss.item()

            # cosine
            cosine = F.cosine_similarity(y_batch, y_pred)
            cosine_list.append(cosine.cpu().numpy())
            total_cosine += cosine.mean().item()
            print("\x1b[2K\rEvaluation [{:.2f}%] Loss: {:.4f} Cosine: {:.4f}".format(
                100.0*count/total_count, total_loss/count, total_cosine/count), end="")
    print("\n")

    cosine = np.hstack(cosine_list)
    cosine = cosine.mean()
    y_pred = np.vstack(y_pred_list)
    return cosine, y_pred

#############################################
# Experiment script
def run_history_exp():
    dropout_rate = 0.3
    for block in [100, 200, 50, 20]:
        for history in [2, 5, 10]:
            start_time = datetime.now()
            print(f"\n\nStart History Exp with model=DAE, block={block}, history={history} removal at {start_time}.")

            train(
                block=block, 
                data_name="bookcorpus", 
                dropout_rate=dropout_rate,
                history=history,
            )
            
            end_time = datetime.now()
            print(f"Finish History Exp with model=DAE, block={block}, history={history} removal at {end_time}.")
            print(f"Took {(end_time-start_time).total_seconds()} Seconds!")

def run_dropout_exp():
    for block in [1, 3, 5]:
        for dropout_rate in [0.3, 0.4, 0.5]:
            train(block=block, data_name="coda19", dropout_rate=dropout_rate)

############################################
# arg parser and main
def parse_args():
    parser = argparse.ArgumentParser(description="DAE Baseline.")
    parser.add_argument("--device", help="device used for computing tfidf and training [cpu/cuda:0/cuda:1]", type=str, default="cpu")
    parser.add_argument("--block", help="story block size", type=int, default=20)
    parser.add_argument("--data", help="Corpus used for training and testing [bookcorpus/coda19]", type=str, default="bookcorpus")
    parser.add_argument("--downsample", help="Downsampling size", type=int, default=-1)
    parser.add_argument("--history", help="Number of story blocks used for input", type=int, default=None)

    parser.add_argument("--hidden_size", help="Hidden size of the DAE model", type=int, default=512)
    parser.add_argument("--layer_num", help="Number of layers of the DAE model", type=int, default=5)
    parser.add_argument("--dropout_rate", help="Dropout rate for the DAE model", type=float, default=0.2)
    
    parser.add_argument("--epoch_num", help="Number of training epoch", type=int, default=2000)
    parser.add_argument("--batch_size", help="Batch size for training", type=int, default=512)
    parser.add_argument("--learning_rate", help="Learning rate for training", type=float, default=1e-4)
    parser.add_argument("--early_stop", help="Number of epoch for early stop", type=float, default=20)

    return parser.parse_args()

def main():
    args = parse_args()

    train(
        block=args.block, 
        data_name=args.data, 
        downsample=args.downsample, 
        dropout_rate=args.dropout_rate, 
        history=args.history, 
        device=args.device,
        params=args
    )

if __name__ == "__main__":
    main()






