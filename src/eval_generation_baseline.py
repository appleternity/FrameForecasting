import numpy as np
import pandas as pd
from config import *
from util import save_prediction
from tfidf_util import tfidf_metric, print_tfidf_metric
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import json
import numpy as np
import argparse

def tokenizer(x):
    return x.split(" ")

def compute_score(block=20):
    print(f"computing result for block size = {block}")

    # load result file
    with open(os.path.join(predict_dir, f"generation_result_block{block}_parsed.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "bookcorpus", "tfidf_model", f"block_{block}.joblib")
    model = joblib.load(tfidf_model_path)

    # turn to vector
    frame_list = []
    for d in data:
        story_frames = " ".join([
            frame["Frame"] for frame in d["frames"]
        ])
        frame_list.append(story_frames)

    predicted_vectors = model.fit_transform(frame_list)
    predicted_vectors = predicted_vectors.toarray()
    print("predicted_vectors.shape = ", predicted_vectors.shape)
    
    # get ground truth
    y = []
    for d in data:
        y.append(d["y"][0])
    y = np.array(y)
    print(f"y.shape = {y.shape}")

    cosine = tfidf_metric(y[:1000, :], predicted_vectors[:1000, :])
    print(f"cosine = {cosine}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Frame Parsing for GPT2 Stories.")
    parser.add_argument("--block", help="Size of the story block", type=int, default=20)
    return parser.parse_args()

def main():
    args = parse_args()
    compute_score(args.block)

if __name__ == "__main__":
    main()
