import numpy as np
import pandas as pd
from config import *
from bookcorpus_data import load_tfidf, tokenizer
from coda_data import load_tfidf as coda_load_tfidf
from tfidf_util import tfidf_metric, print_tfidf_metric, tfidf_metric_detail
from util import h5_save
import argparse

def tfidf_replay_baseline_skip_block(block, skip, data_name="bookcorpus", device="cpu"):
    print("block = {}, skip = {}".format(block, skip))
    if data_name == "bookcorpus":
        x_test, y_test = load_tfidf("test", block, skip, verbose=True, redo=False)
    elif data_name == "coda19":
        x_test, y_test = coda_load_tfidf("test", block, verbose=True)
    else:
        print("Not supported yet!")
        quit()

    x_test, y_test = x_test.todense(), y_test.todense()
    y_pred = x_test

    res = tfidf_metric(y_test, y_pred, device=device)
    print(res)
    print_tfidf_metric({
        "cosine": float(res),
        "block": block,
        "skip": skip,
        "note": data_name 
    }, filename=os.path.join(result_dir, f"{data_name}_tfidf_skip_block.json"))

def tfidf_prior_baseline(block, data_name="bookcorpus", downsample=-1, device="cpu"):
    if data_name == "bookcorpus":
        x_train, y_train = load_tfidf("train", block)
        x_test, y_test = load_tfidf("test", block)
    elif data_name == "coda19":
        x_train, y_train = coda_load_tfidf("train", block, verbose=True)
        x_test, y_test = coda_load_tfidf("test", block, verbose=True)
    else:
        print("Not supported yet!")
        quit()

    # downsample
    if downsample != -1:
        random_index = np.random.RandomState(5516).permutation(x_train.shape[0])[:downsample]
        x_train, y_train = x_train[random_index], y_train[random_index]

    x_train, y_train = x_train.todense(), y_train.todense()
    x_test, y_test = x_test.todense(), y_test.todense()

    prior = np.mean(y_train, axis=0)
    print(prior)
    prior = np.repeat(prior.reshape([1, -1]), y_test.shape[0], axis=0)
    print(prior.shape)
    res = tfidf_metric(y_test, prior)
    print(res)

    cosine_detail = tfidf_metric_detail(y_test, prior, device=device)
    print(cosine_detail)
    h5_save(
        os.path.join(predict_dir, f"{data_name}_tfidf_prior_{block}.h5"),
        name="cosine",
        data=cosine_detail,
    )

    print_tfidf_metric({
        "cosine": float(res),
        "block": block,
        "skip": 0,
        "note": "downsample",
    }, filename=os.path.join(result_dir, f"{data_name}_tfidf_prior_baseline.json"))

def parse_args():
    parser = argparse.ArgumentParser(description="Naive Baseline.")
    parser.add_argument("--device", help="device used for computing tfidf [cpu/cuda:0/cuda:1]", type=str, default="cpu")
    parser.add_argument("--block", help="story block size", type=int, default=20)
    parser.add_argument("--data", help="Corpus used for training and testing [bookcorpus/coda19]", type=str, default="bookcorpus")
    parser.add_argument("--model", help="Model type [replay/prior]", type=str, default="replay")
    parser.add_argument("--downsample", help="Downsampling size", type=int, default=-1)
    parser.add_argument("--skip", help="Skipping distance for replay baseline", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.model == "prior":
        tfidf_prior_baseline(
            block=args.block,
            data_name=args.data,
            downsample=args.downsample,
            device=args.device,
        )
    elif args.model == "replay":
        tfidf_replay_baseline_skip_block(
            block=args.block,
            data_name=args.data,
            skip=args.skip,
            device=args.device,
        )
    else:
        print(f"{args.model} is not supported. We only support 'prior' and 'replay' in this script.")

if __name__ == "__main__":
    main()
