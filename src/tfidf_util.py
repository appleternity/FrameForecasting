import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kl_div
import os
import json
import torch.nn.functional as F
import torch

def tfidf_metric(y_true, y_hat, device="cuda:0", batch_size=2048):
    # torch implementation
    try:
        cosines = []
        for start in range(0, y_true.shape[0], batch_size):
            y1 = torch.DoubleTensor(y_true[start:start+batch_size, :]).to(device)
            y2 = torch.DoubleTensor(y_hat[start:start+batch_size, :]).to(device)
            cosines.append(
                F.cosine_similarity(y1, y2).cpu()
            )
        cosine = torch.cat(cosines, 0).mean().item()
        return cosine

    except RuntimeError:
        torch.cuda.empty_cache()
        
        # cpu version
        cosines = []
        for start in range(0, y_true.shape[0], batch_size):        
            y1 = torch.DoubleTensor(y_true[start:start+batch_size, :])
            y2 = torch.DoubleTensor(y_hat[start:start+batch_size, :])
            cosines.append(
                F.cosine_similarity(y1, y2).cpu()
            )
        cosine = torch.cat(cosines, 0).mean().item()
        return cosine

def tfidf_metric_detail(y_true, y_hat, device="cuda:0", batch_size=2048):
    cosines = []
    for start in range(0, y_true.shape[0], batch_size):
        y1 = torch.DoubleTensor(y_true[start:start+batch_size, :]).to(device)
        y2 = torch.DoubleTensor(y_hat[start:start+batch_size, :]).to(device)
        cosines.append(
            F.cosine_similarity(y1, y2).cpu()
        )
    cosine = torch.cat(cosines, 0).numpy()
    return cosine

def tfidf_kl_div(y_true, y_hat):
    value = kl_div(y_true, y_hat)
    print(value)

def print_tfidf_metric(result, filename):
    if os.path.isfile(filename):
        with open(filename, 'r', encoding='utf-8') as infile:
            results = json.load(infile)
    else:
        results = []

    results.append(result)

    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

def test():
    a = np.ones([5, 10]) * 0.3
    b = np.ones([5, 10]) * 0.5
    res = tfidf_metric(a, b)
    print(res)

    a = np.array([[1, 2, 3]])
    b = np.array([[5, 6, 7]])
    res = tfidf_metric(a, b)
    print(res, 0.9683296637314885)

    a = np.array([[0.1036361 , 0.14080719, 0.09930763]])
    b = np.array([[0.20482169, 0.56282634, 0.29362836]])
    res = tfidf_metric(a, b)
    print(res, 0.9665578213338701)

if __name__ == "__main__":
    test()
