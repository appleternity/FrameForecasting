import numpy as np
import pandas as pd
from config import *
from util import save_prediction
from tfidf_util import tfidf_metric, print_tfidf_metric
from bookcorpus_data import load_tfidf, load_tfidf_detail, load_tfidf_long
from coda_data import load_tfidf as coda_load_tfidf

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import joblib
from datetime import datetime
import argparse

RANDOM_SEED=1024

def tokenizer(x):
    return x.split(" ")

def metric(y_true, y_hat): 
    micro = precision_recall_fscore_support(y_true, y_hat, average="micro")[:-1]
    macro = precision_recall_fscore_support(y_true, y_hat, average="macro")[:-1]
    sample = precision_recall_fscore_support(y_true, y_hat, average="samples")[:-1]
    return [macro, micro, sample]

def print_metric(res, filename=None):
    table = pd.DataFrame(res, columns=["Precision", "Recall", "F1"], index=["Macro", "Micro", "Sample"])
    table = table.transpose()
    print(table)

    if filename is not None:
        table.to_csv(filename)

def tf_ml_baseline(block=200, model_name="RandomForest", data_name="bookcorpus", downsample=-1, history=None, n_jobs=10, device="cpu"):
    print("loading data")

    # tfidf as feature
    if data_name == "bookcorpus":
        if history is None:
            x_train, y_train = load_tfidf("train", block, verbose=True)
            x_test, y_test = load_tfidf("test", block, verbose=True)
        else:
            x_train, y_train = load_tfidf_long("train", block, verbose=True, history=history)
            x_test, y_test = load_tfidf_long("test", block, verbose=True, history=history)
    elif data_name == "coda19":
        x_train, y_train = coda_load_tfidf("train", block, verbose=True)
        x_test, y_test = coda_load_tfidf("test", block, verbose=True)
    else:
        print("Not supported yet!")
        quit()
   
    if downsample != -1:
        random_index = np.random.RandomState(5516).permutation(x_train.shape[0])[:88720]
        x_train, y_train = x_train[random_index], y_train[random_index]

    # do sampling if the training data is too big
    if x_train.shape[0] > 1000000:
        index_list = np.random.RandomState(seed=RANDOM_SEED).permutation(x_train.shape[0])[:1000000]
        index_list = np.sort(index_list)
        x_train, y_train = x_train[index_list], y_train[index_list]

    x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)                                                                                                             
    x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)

    x_train, y_train = x_train.todense(), y_train.todense()
    x_test, y_test = x_test.todense(), y_test.todense()

    print("train: x = {}, y = {}".format(str(x_train.shape), str(y_train.shape)))
    print("test: x = {}, y = {}".format(str(x_test.shape), str(y_test.shape)))
    print("building model using", model_name)

    # parameter setting
    rf_param = {
        "max_depth": 10, 
        "random_state": RANDOM_SEED, 
        "n_jobs": n_jobs, 
        "n_estimators": 30,
        "verbose": 10,
    }
    lgbm_param = {
        "max_depth": 3,
        "num_leaves": 5,
        "random_state": RANDOM_SEED,
        "n_estimators":100,
        "n_jobs": 1,
        "verbose": -1,
        "force_row_wise":True,
        "device":"gpu",
    }
    if model_name == "RandomForest":
        model = RandomForestRegressor(**rf_param)
    elif model_name == "LGBM":
        model = MultiOutputRegressor(LGBMRegressor(**lgbm_param), n_jobs=n_jobs)
    else:
        print("Please use the available model")

    print("training")
    model.fit(x_train, y_train)

    if history is None:
        model_output = os.path.join(model_dir, data_name, "block{}_{}.joblib".format(block, model_name))
        filename = os.path.join(result_dir, f"{data_name}_ml_baseline.json")
    else:
        model_output = os.path.join(model_dir, data_name, "history_block{}_{}.joblib".format(block, model_name))
        filename = os.path.join(result_dir, f"history_exp_{data_name}_ml_baseline.json")
    
    # save model
    joblib.dump(model, model_output)

    # make prediction
    print("prediting")
    print("block number = {}".format(block))
    y_pred = model.predict(x_test)
    res = tfidf_metric(y_test, y_pred, device=device)
    print("cosine", res)
    print_tfidf_metric(
        {
            "cosine": float(res),
            "block": block,
            "model": model_name,
            "note": "clean - tfidf - downsample" if downsample != -1 else "clean - tfidf",
            "history": history, 
        }, 
        filename=filename
    )
    
    # output y_pred
    if downsample == -1:
        if history:
            outpath = os.path.join(predict_dir, "bookcorpus", f"history_block{block}_{model_name}_h{history}.h5")
        else:
            outpath = os.path.join(predict_dir, "bookcorpus", f"block{block}_{model_name}.h5")
    else:
        outpath = os.path.join(predict_dir, "bookcorpus", f"downsample_block{block}_{model_name}.h5")
    save_prediction(outpath, y_pred)

def build_model(model_name, param):
    if model_name == "RandomForest":
        return RandomForestRegressor(**param)
    elif model_name == "LGBM":
        return MultiOutputRegressor(LGBMRegressor(**param))
    else:
        print(f"{model_name} not supported yet!")
        quit()

def tf_ml_baseline_tuning(block=200, model_name="RandomForest", data_name="bookcorpus", sampling=False):
    print("loading data")

    # tfidf as feature
    if data_name == "bookcorpus":
        x_train, y_train = load_tfidf("train", block, verbose=True)
        x_valid, y_valid = load_tfidf("valid", block, verbose=True)
        x_test, y_test = load_tfidf("test", block, verbose=True)
    elif data_name == "coda19":
        x_train, y_train = coda_load_tfidf("train", block, verbose=True)
        x_valid, y_valid = coda_load_tfidf("valid", block, verbose=True)
        x_test, y_test = coda_load_tfidf("test", block, verbose=True)
    else:
        print("Not supported yet!")
        quit()
   
    # do sampling if the training data is too big
    if x_train.shape[0] > 1000000:
        index_list = np.random.RandomState(seed=RANDOM_SEED).permutation(x_train.shape[0])[:1000000]
        index_list = np.sort(index_list)
        x_train, y_train = x_train[index_list], y_train[index_list]

    x_train, y_train = x_train.todense(), y_train.todense()
    x_valid, y_valid = x_valid.todense(), y_valid.todense()
    x_test, y_test = x_test.todense(), y_test.todense()

    print("train: x = {}, y = {}".format(str(x_train.shape), str(y_train.shape)))
    print("test: x = {}, y = {}".format(str(x_test.shape), str(y_test.shape)))

    print("building model using", model_name)

    # parameter setting
    rf_param_list = [
        {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "n_jobs": 4,
            "random_state": RANDOM_SEED,
            "verbose": 1
        }
        for max_depth in [10, 20, 30]
        for n_estimators in [100, 150, 200, 250]
    ]

    lgbm_param_list = [
        {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "random_state": RANDOM_SEED,
            "n_jobs": 10,
            "verbose": 0,
            "force_row_wise":True,
        }
        for max_depth in [10, 20, 30]
        for n_estimators in [100, 150, 200, 250]
    ]
   
    param_dict = {
        "RandomForest": rf_param_list,
        "LGBM": lgbm_param_list,
    }
    param_list = param_dict[model_name]

    print("finding the best parameters")
    best_param = None
    best_score = 0.0
    best_model = None
    for i, param in enumerate(param_list):
        print(f"Running parameter tuning {i} / {len(param_list)}")
        print(param)
        model = build_model(model_name, param)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        score = tfidf_metric(y_valid, y_pred)
        
        if score > best_score:
            best_param = param
            best_score = score
            best_model = model
    
    print()
    
    joblib.dump(
        best_model,
        os.path.join(model_dir, data_name, "block{}_{}.joblib".format(block, model_name))
    )
    #with open(os.path.join(model_dir, data_name, f"block{block}_{model_name}_param.json"), 'w', encoding='utf-8') as outfile:
    #    json.dump(best_param)

    print("prediting")
    print("block number = {}".format(block))
    y_pred = best_model.predict(x_test)    
    res = tfidf_metric(y_test, y_pred)
    print("cosine", res)
    print("best parameter", best_param)
    print_tfidf_metric(
        {
            "cosine": float(res),
            "block": block,
            "model": model_name,
            "note": "clean - tfidf - new",
            "best_parameter": best_param
        }, 
        filename=os.path.join(result_dir, f"{data_name}_ml_baseline.json")
    )

def parse_args():
    parser = argparse.ArgumentParser(description="ML Baseline (LGBM / RandomForest).")
    parser.add_argument("--device", help="device used for computing tfidf [cpu/cuda:0/cuda:1]", type=str, default="cpu")
    parser.add_argument("--block", help="story block size", type=int, default=20)
    parser.add_argument("--data", help="Corpus used for training and testing [bookcorpus/coda19]", type=str, default="bookcorpus")
    parser.add_argument("--downsample", help="Downsampling size", type=int, default=-1)
    parser.add_argument("--history", help="Number of story blocks used for input", type=int, default=None)
    parser.add_argument("--n_jobs", help="Processes used for computing", type=int, default=10)
    parser.add_argument("--model", help="ML model. (LGBM / RandomForest)", type=str, default="LGBM")
    return parser.parse_args()

def main():
    args = parse_args()
    
    tf_ml_baseline(
        block=args.block, 
        model_name=args.model, 
        data_name=args.data, 
        downsample=args.downsample, 
        history=args.history,
        n_jobs=args.n_jobs,
        device=args.device,
    )

if __name__ == "__main__":
    main()
