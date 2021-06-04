import numpy as np
import pandas as pd
from config import *
from bookcorpus_data import load_tfidf
from coda_data import load_tfidf as coda_load_tfidf
from tfidf_util import tfidf_metric, print_tfidf_metric
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import torch 
import os
import argparse

def my_cosine():
    dot = y_true * y_hat
    dot = np.sum(dot, axis=1)
    y_true_norm = np.linalg.norm(y_true, axis=1)
    y_hat_norm = np.linalg.norm(y_hat, axis=1)
    cosine = np.divide(dot, y_true_norm*y_hat_norm)
    return cosine

def tfidf_ir_pytorch(block, data_name="bookcorpus", downsample=-1, device="cpu"):
    # using frame-vector as a query
    if data_name == "bookcorpus":
        x_train, y_train = load_tfidf("train", block)
        x_test, y_test = load_tfidf("test", block)
    elif data_name == "coda19":
        x_train, y_train = coda_load_tfidf("train", block)
        x_test, y_test = coda_load_tfidf("test", block)
    else:
        print(f"{data_name} not supported yet!")
        quit()

    x_train, y_train = x_train.todense(), y_train.todense()
    x_test, y_test = x_test.todense(), y_test.todense()

    if downsample != -1:
        random_index = np.random.RandomState(5516).permutation(x_train.shape[0])[:downsample]
        x_train, y_train = x_train[random_index], y_train[random_index]
        print(f"downsampling x_train.shape = {x_train.shape}, y_train.shape = {y_train.shape}")

    partition_size = 100000
    answer_list = np.ones([x_test.shape[0]], dtype=np.int32) * (-1)
    best_distance_list = np.zeros([x_test.shape[0]], dtype=np.float64)
    total_count = x_test.shape[0]
    length = x_test.shape[0]
    partition_num = x_train.shape[0] // partition_size

    for partition_id, train_start_index in enumerate(range(0, x_train.shape[0], partition_size)):
        train = x_train[train_start_index:train_start_index+partition_size]
        train = torch.DoubleTensor(train).to(device)
        train_length = train.shape[0]
        #print(partition_id, "train.shape =", train.shape)

        # go through all the testing instances
        with torch.no_grad():
            for index in range(0, x_test.shape[0]):
                if index % 100 == 0:
                    print("\x1b[2K\rpartition {} / {} , predicting {} / {} [{:.3f}%]".format(
                            partition_id, partition_num, index, total_count, 100.0*index/total_count
                        ), end="")
                x_batch = x_test[index:index+1]
                x_batch = torch.DoubleTensor(x_batch).to(device)
                x_batch = x_batch.repeat([train_length, 1])
                distances = F.cosine_similarity(x_batch, train)
                best_distance = torch.max(distances).cpu().numpy()
                answer_index = torch.argmax(distances).cpu().numpy()
                if best_distance > best_distance_list[index]:
                    best_distance_list[index] = best_distance
                    answer_list[index] = answer_index

    print()    
    y_pred = y_train[answer_list] 
    print(y_pred.shape, y_test.shape)
    res = tfidf_metric(y_test, y_pred)
    print(res)

    print_tfidf_metric({
        "cosine": float(res),
        "block": block,
        "skip": 0,
        "note": "frame - ir - downsample" if downsample != -1 else "frame - ir"
    }, filename=os.path.join(result_dir, f"{data_name}_tfidf_ir_baseline.json"))

def parse_args():
    parser = argparse.ArgumentParser(description="IR Baseline.")
    parser.add_argument("--device", help="device used for computing tfidf [cpu/cuda:0/cuda:1]", type=str, default="cpu")
    parser.add_argument("--block", help="story block size", type=int, default=20)
    parser.add_argument("--data", help="Corpus used for training and testing [bookcorpus/coda19]", type=str, default="bookcorpus")
    parser.add_argument("--downsample", help="Downsampling size", type=int, default=-1)
    return parser.parse_args()

def main():
    args = parse_args()
    
    tfidf_ir_pytorch(
        block=args.block,
        data_name=args.data,
        downsample=args.downsample,
        device=args.device,
    )

if __name__ == "__main__":
    main()
