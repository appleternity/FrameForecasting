import json
import pickle
import os
import random
import numpy as np
import scipy.sparse as sp
from config import *
from nltk.corpus import framenet as fn
import h5py 
from collections import Counter, OrderedDict
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from nltk import word_tokenize
from word2number import w2n
import pandas as pd
import feather
from bookcorpus_data import get_global_frame_dictionary, get_size

def generate_split():
    train_ratio, valid_ratio, test_ratio = 0.7, 0.1, 0.2
    codacorpus = os.path.join(data_dir, "coda19", "CODA19_v1_20200504", "human_label")

    data_index = {}
    for setting in ["train", "dev", "test"]:
        if setting == "dev":
            data_setting = "valid"
        else:
            data_setting = setting

        data_index[data_setting] = [
            {
                "book": f.replace(".json", ".txt"),
                "start": -1,
                "end": -1
            }
            for f in os.listdir(os.path.join(codacorpus, setting))
            if "swp" not in f and os.path.isfile(os.path.join(data_dir, "coda19", "frame", f"{f.replace('.json', '.txt')}"))
        ]
    
    total_number = sum(len(val) for val in data_index.values())    
    print("total number", total_number)
    print("train:", len(data_index["train"]))
    print("valid", len(data_index["valid"]))
    print("test", len(data_index["test"]))

    # save data
    with open(os.path.join(data_dir, "coda19", "split.json"), 'w', encoding='utf-8') as outfile:
        json.dump(data_index, outfile, indent=2)

def tokenizer(x):
    return x.split(" ")

def build_tfidf_model(block=20):
    print("building tfidf model for block =", block)
    book_list = get_coda_split()["train"]
    frame_dictionary = get_global_frame_dictionary()
    model = TfidfVectorizer(
        vocabulary=frame_dictionary,
        norm=None,
        lowercase=None,
        tokenizer=tokenizer,
    )
    
    # build model
    content = []
    split_index = []
    for count, book_info in enumerate(book_list):
        print("\x1b[2K\rBuilding tf-idf model {:>5} / {:>5} [{:.2f}%]".format(
                count, len(book_list), 100.0*count/len(book_list)
            ), end="")
        with open(book_info["path"], 'r', encoding='utf-8') as infile:
            book = json.load(infile)
            if book_info["start"] != -1:
                book = book[book_info["start"]:book_info["end"]]
            for start in range(0, len(book), block):
                story_start, story_end = start, start+block
                
                if story_end > len(book):
                    continue

                story_frames = " ".join([
                    frame["Frame"]
                    for sent in book[story_start:story_end]
                        for frame in sent["frame"]
                ])
                content.append(story_frames)
    print()

    vectors = model.fit_transform(content)
    get_size(vectors)

    # save model
    joblib.dump(
        model, 
        os.path.join(data_dir, "coda19", "tfidf_model", "block_{}.joblib".format(block))
    )

def load_tfidf(phrase="train", block=20, skip=0, redo=False, verbose=True):
    print("Loading block = {}, phrase = {}, skip = {}".format(block, phrase, skip))
    x_path = os.path.join(data_dir, "coda19", "cache", "tfidf_vectors_block{}_skip{}_{}_x.npz".format(block, skip, phrase))
    y_path = os.path.join(data_dir, "coda19", "cache", "tfidf_vectors_block{}_skip{}_{}_y.npz".format(block, skip, phrase))

    if not redo and os.path.isfile(x_path) and os.path.isfile(y_path):
        x = sp.load_npz(x_path)
        y = sp.load_npz(y_path)
        print("x.shape =", x.shape, ", y.shape =", y.shape)
        return x, y

    book_list = get_coda_split()[phrase][:]
   
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "coda19", "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=20)

    # build model
    block_skip = skip + 1
    x = []
    y = []
    for count, book_info in enumerate(book_list):
        print("\x1b[2K\rComputing tf-idf {:>5} / {:>5} [{:.2f}%]".format(
                count, len(book_list), 100.0*count/len(book_list)
            ), end="")
        content = []
        with open(book_info["path"], 'r', encoding='utf-8') as infile:
            book = json.load(infile)
            if book_info["start"] != -1:
                book = book[book_info["start"]:book_info["end"]]
            for start in range(0, len(book), block):
                story_start, story_end = start, start+block
                
                if story_end > len(book):
                    continue

                story_frames = " ".join([
                    frame["Frame"]
                    for sent in book[story_start:story_end]
                        for frame in sent["frame"]
                ])
                content.append(story_frames)
        if content:
            vectors = model.transform(content)
            x.extend(vectors[:-block_skip])
            y.extend(vectors[block_skip:])
    print()

    x = sp.vstack(x)
    y = sp.vstack(y)
    get_size(x, "Block {}, x".format(block))
    get_size(y, "Block {}, y".format(block))
    sp.save_npz(x_path, x)
    sp.save_npz(y_path, y)

    return x, y

def load_tfidf_detail(phrase="train", block=20, skip=0, redo=False, verbose=True):
    print("Loading block = {}, phrase = {}, skip = {}".format(block, phrase, skip))
    x_path = os.path.join(data_dir, "coda19", "cache", "detail_tfidf_vectors_block{}_skip{}_{}_x.npz".format(block, skip, phrase))
    y_path = os.path.join(data_dir, "coda19", "cache", "detail_tfidf_vectors_block{}_skip{}_{}_y.npz".format(block, skip, phrase))
    info_path = os.path.join(data_dir, "coda19", "cache", "detail_tfidf_vectors_block{}_skip{}_{}_info.json".format(block, skip, phrase))

    if not redo and os.path.isfile(x_path) and os.path.isfile(y_path) and os.path.isfile(info_path):
        x = sp.load_npz(x_path)
        y = sp.load_npz(y_path)
        with open(info_path, 'r', encoding='utf-8') as infile:
            info = json.load(infile)
        print("x.shape =", x.shape, ", y.shape =", y.shape, ", info.len = ", len(info))
        return x, y, info

    book_list = get_coda_split()[phrase][:100]
   
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "coda19", "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=20)

    # build model
    block_skip = skip + 1
    x = []
    y = []
    x_text = []
    x_frame = []
    y_text = []
    y_frame = []
    b_list = []
    position = []
    for count, book_info in enumerate(book_list):
        print("\x1b[2K\rComputing tf-idf {:>5} / {:>5} [{:.2f}%]".format(
                count, len(book_list), 100.0*count/len(book_list)
            ), end="")
        content = []
        temp_book = []
        temp_position = []
        with open(book_info["path"], 'r', encoding='utf-8') as infile:
            book = json.load(infile)
            if book_info["start"] != -1:
                book = book[book_info["start"]:book_info["end"]]
            for start in range(0, len(book), block):
                story_start, story_end = start, start+block
                
                if story_end > len(book):
                    continue

                story_frames = " ".join([
                    frame["Frame"]
                    for sent in book[story_start:story_end]
                        for frame in sent["frame"]
                ])
                content.append(story_frames)
                
                # book info
                temp_book.append(book_info["path"])
                if book_info["start"] != -1:
                    temp_position.append([book_info["start"]+story_start, book_info["start"]+story_end])
                else:
                    temp_position.append([story_start, story_end])

        if content:
            vectors = model.transform(content)
            x.extend(vectors[:-block_skip])
            y.extend(vectors[block_skip:])
            b_list.extend(temp_book[:-block_skip])
            position.extend(temp_position[:-block_skip])

    print()

    x = sp.vstack(x)
    y = sp.vstack(y)
    get_size(x, "Block {}, x".format(block))
    get_size(y, "Block {}, y".format(block))
    sp.save_npz(x_path, x)
    sp.save_npz(y_path, y)

    with open(info_path, 'w', encoding='utf-8') as outfile:
        json.dump({"book_list":b_list, "position": position}, outfile, indent=2)

    return x, y, {"book_list":b_list, "position": position}

def load_text_data(phrase="train", block=20, skip=0, redo=False, verbose=True, target_model="bert"):
    from transformers import LongformerTokenizerFast, BertTokenizerFast, AutoTokenizer

    #print("Loading block = {}, phrase = {}, skip = {}".format(block, phrase, skip))
    x_path = os.path.join(data_dir, "coda19", "cache", "{}_text_tfidf_vectors_block{}_skip{}_{}_x.parquet".format(target_model, block, skip, phrase))
    y_path = os.path.join(data_dir, "coda19", "cache", "{}_text_tfidf_vectors_block{}_skip{}_{}_y.npz".format(target_model, block, skip, phrase))

    if not redo and os.path.isfile(x_path) and os.path.isfile(y_path):
        #x = feather.read_dataframe(x_path)["text"]
        data = pd.read_parquet(x_path)
        x = data["id"]
        text = data["text"]
        y = sp.load_npz(y_path)
        print("x.shape =", x.shape, ", y.shape =", y.shape)
        return text, x, y

    book_list = get_coda_split()[phrase][:]
   
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "coda19", "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=20)

    # build model
    if target_model == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif target_model == "longerformer":
        tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    elif target_model == "scibert":
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    else:
        print(f"{target_model} not supported yet")
        quit()

    block_skip = skip + 1
    x_text = []
    x_id = []
    y = []
    for count, book_info in enumerate(book_list):
        print("\x1b[2K\rComputing tf-idf {:>5} / {:>5} [{:.2f}%]".format(
                count, len(book_list), 100.0*count/len(book_list)
            ), end="")
        content = []
        text = []
        ids = []
        with open(book_info["path"], 'r', encoding='utf-8') as infile:
            book = json.load(infile)
            if book_info["start"] != -1:
                book = book[book_info["start"]:book_info["end"]]
            for start in range(0, len(book), block):
                story_start, story_end = start, start+block
                
                if story_end > len(book):
                    continue

                # frame
                story_frames = " ".join([
                    frame["Frame"]
                    for sent in book[story_start:story_end]
                        for frame in sent["frame"]
                ])
                content.append(story_frames)

                # text
                t = "\n".join([sent["text"] for sent in book[story_start:story_end]])
                text.append(t)
                ids.append(np.array(tokenizer.encode(t, add_special_tokens=False)))

        if content:
            assert len(content) == len(text) == len(ids)
            vectors = model.transform(content)
            x_text.extend(text[:-block_skip])
            x_id.extend(ids[:-block_skip])
            y.extend(vectors[block_skip:])
    print()

    #x = sp.vstack(x)
    #get_size(x, "Block {}, x".format(block))
    #sp.save_npz(x_path, x)
    x = pd.DataFrame([x_text, x_id], index=["text", "id"])
    x = x.transpose()
    #print(x)
    print("Block {}, x size = {:.2f} MB".format(block, x.memory_usage(index=False, deep=True)[0]/1024/1024))
    x.to_parquet(x_path)

    y = sp.vstack(y)
    get_size(y, "Block {}, y".format(block))
    sp.save_npz(y_path, y)

    return text, x, y

def compute_avg_length():
    book_list = get_coda_split()["train"][:]

    running_mean = 0
    running_count = 0
    for count, book_info in enumerate(book_list):
        content = []
        with open(book_info["path"], 'r', encoding='utf-8') as infile:
            book = json.load(infile)
            if book_info["start"] != -1:
                book = book[book_info["start"]:book_info["end"]]
            
            for i, sent in enumerate(book):
                tokens = word_tokenize(sent["text"])
                running_mean = (running_count)/(running_count+1)*running_mean + 1.0/(running_count+1)*len(tokens)
                running_count += 1

                if i % 100 == 0:
                    print("\x1b[2K\rComputing Sentence Length {:>5} / {:>5} [{:.2f}%] Average Length = {:.5f}".format(
                            count, len(book_list), 100.0*count/len(book_list), running_mean
                        ), end="")

    with open(os.path.join(output_dir, "mean_length.txt"), 'w', encoding='utf-8') as outfile:
        outfile.write("Mean Length = {}".format(running_mean))

###########################################################
# scripts
def batch_build_tfidf_model():
    for block in [1, 3, 5]:
        build_tfidf_model(block=block)

def batch_build_tfidf():
    for block in [1, 3, 5]:
        load_tfidf(phrase="train", block=block, redo=True)
        load_tfidf(phrase="valid", block=block, redo=True)
        load_tfidf(phrase="test", block=block, redo=True)

def batch_tfidf():
    for block in [5, 10, 20, 50, 100, 150, 200]:
        build_tfidf_model(block=block)
        load_tfidf(phrase="train", block=block, redo=True)
        load_tfidf(phrase="valid", block=block, redo=True)
        load_tfidf(phrase="test", block=block, redo=True)

def test():
    old_x, old_y = old_load_tfidf(phrase="train", block=200)
    x, y = load_tfidf(phrase="train", block=200)
    return x, y, old_x, old_y

def statistic_length():
    length = []
    book_dictionary = get_coda_split()
    for book_list in book_dictionary.values():
        for book_info in book_list:
            with open(book_info["path"], 'r', encoding='utf-8') as infile:
                book = json.load(infile)
                length.append(len(book))

    length = np.array(length)
    print(f"Mean Length = {length.mean()}")  
    for p in np.arange(0, 101, 10):
        print(f"{p}% of data has {np.percentile(length, p)} sentences")

def main():
    #statistic_length()
    #generate_split()
    batch_build_tfidf_model()
    batch_build_tfidf()
    #remove_header()
    #batch_tfidf()
    #compute_avg_length()
    #quit()

    #for block in [50, 100, 150, 200]:
    for block in [1, 3, 5]:
        load_text_data(phrase="train", block=block, redo=True, target_model="scibert")
        load_text_data(phrase="valid", block=block, redo=True, target_model="scibert")
        load_text_data(phrase="test", block=block, redo=True, target_model="scibert")
        
        load_text_data(phrase="train", block=block, redo=True, target_model="bert")
        load_text_data(phrase="valid", block=block, redo=True, target_model="bert")
        load_text_data(phrase="test", block=block, redo=True, target_model="bert")

if __name__ == "__main__":
    main()

