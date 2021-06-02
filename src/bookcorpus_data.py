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

# NOTE:
# scikit-learn's implementation of TFIDF adds one to the IDF (make sure this is modified)
os.environ["TOKENIZERS_PARALLELISM"]="false"

def get_global_frame_dictionary():
    frame_dict = {
        f["name"] : i
        for i, f in enumerate(fn.frames())
    }
    return frame_dict

def get_size(matrix, note=None):
    size = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
    size = size / 1024 / 1024
    if note is None:
        print("size = {:.2f} MB".format(size))
    else:
        print("{} size = {:.2f} MB".format(note, size))

def remove_header():
    bookcorpus_dir = os.path.join(data_dir, "bookcorpus", "segment")
    book_list = os.listdir(bookcorpus_dir)
    total_number = len(book_list)

    # check all books
    counter = Counter()
    last_counter = Counter()
    book_info = []
    for count, book_name in enumerate(book_list):
        print("\x1b[2K\rRemoving Header Loading Data {:>5} / {:>5} [{:.2f}%]".format(
                count, len(book_list), 100.0*count/len(book_list)
            ), end="")

        if not os.path.isfile(os.path.join(data_dir, "bookcorpus", "frame", book_name)):
            continue

        with open(os.path.join(bookcorpus_dir, book_name), 'r', encoding='utf-8') as infile:
            lines = infile.read().split("\n")

            # find chapter one
            chapter_one = -1
            for i, line in enumerate(lines):
                if line[:7].lower() == "chapter":
                    tokens = word_tokenize(line)
                    if len(tokens) < 2:
                        continue

                    # ignore table of content
                    if sum([1 for token in tokens if token.lower()=="chapter"]) > 1:
                        continue
                
                    # find chapter one
                    try:
                        chapter_num = w2n.word_to_num(tokens[1])
                    except ValueError:
                        chapter_num = None

                    if chapter_num == 1:
                        counter.update([line])
                        chapter_one = i
                        break

            if chapter_one == -1:
                continue
            if chapter_one > 300:
                continue

            # find last chapter
            chapter_info = []
            for i, line in enumerate(lines):
                if line[:7].lower() == "chapter":
                    tokens = word_tokenize(line)
                    if len(tokens) < 2:
                        continue

                    # ignore table of content
                    if sum([1 for token in tokens if token.lower()=="chapter"]) > 1:
                        continue

                    try:
                        chapter_num = w2n.word_to_num(tokens[1])
                    except ValueError:
                        chapter_num = None

                    if chapter_num is not None:
                        chapter_info.append([chapter_num, i, line])
           
            if chapter_info[-1][0] > 1:
                start = chapter_one
                end = chapter_info[-1][1]
                last_counter.update([chapter_info[-1][2]])
                book_info.append({
                    "book": book_name,
                    "start": start,
                    "end": end,
                })

    print()
    sorted_result = sorted(counter.items(), key=lambda x:x[1], reverse=True)
    counter = OrderedDict()
    for k, v in sorted_result:
        counter[k] = v
   
    sorted_result = sorted(last_counter.items(), key=lambda x:x[1], reverse=True)
    last_counter = OrderedDict()
    for k, v in sorted_result:
        last_counter[k] = v

    print("book count = ", sum(counter.values()))
    print("book last_count = ", sum(last_counter.values()))

    with open("chapter_mapping.json", 'w', encoding='utf-8') as outfile:
        json.dump(counter, outfile, indent=2)
    
    with open("last_chapter_mapping.json", 'w', encoding='utf-8') as outfile:
        json.dump(last_counter, outfile, indent=2)

    # generate split for the clean book info
    train_ratio, valid_ratio, test_ratio = 0.7, 0.1, 0.2
    total_number = len(book_info)
    index_list = np.random.permutation(total_number)
    
    # index split
    test_start, test_end = 0, int(total_number * test_ratio)
    valid_start, valid_end = test_end, test_end + int(total_number * valid_ratio)
    train_start, train_end = valid_end, total_number

    # split
    test_index_list = index_list[test_start:test_end]
    valid_index_list = index_list[valid_start:valid_end]
    train_index_list = index_list[train_start:train_end]

    train_index_list = sorted(train_index_list.tolist())
    valid_index_list = sorted(valid_index_list.tolist())
    test_index_list = sorted(test_index_list.tolist())

    # get filenames
    train = [book_info[i] for i in train_index_list]
    valid = [book_info[i] for i in valid_index_list]
    test = [book_info[i] for i in test_index_list]

    print("total number", total_number)
    print("train:", len(train))
    print("valid", len(valid))
    print("test", len(test))

    # save data
    with open(os.path.join(data_dir, "bookcorpus", "clean_split.json"), 'w', encoding='utf-8') as outfile:
        json.dump({
            "train":train,
            "valid":valid,
            "test":test,
        }, outfile, indent=2)

def tokenizer(x):
    return x.split(" ")

def build_tfidf_model(block=20):
    print("building tfidf model for block =", block)
    book_list = get_book_split()["train"]
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
        os.path.join(data_dir, "bookcorpus", "tfidf_model", "block_{}.joblib".format(block))
    )

def load_tfidf(phase="train", block=20, skip=0, redo=False, verbose=True):
    print("Loading block = {}, phase = {}, skip = {}".format(block, phase, skip))
    x_path = os.path.join(data_dir, "bookcorpus", "cache", "tfidf_vectors_block{}_skip{}_{}_x.npz".format(block, skip, phase))
    y_path = os.path.join(data_dir, "bookcorpus", "cache", "tfidf_vectors_block{}_skip{}_{}_y.npz".format(block, skip, phase))

    if not redo and os.path.isfile(x_path) and os.path.isfile(y_path):
        x = sp.load_npz(x_path)
        y = sp.load_npz(y_path)
        print("x.shape =", x.shape, ", y.shape =", y.shape)
        return x, y

    book_list = get_book_split()[phase][:]
   
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "bookcorpus", "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=block)

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

def load_tfidf_long(phase="train", block=20, history=1, redo=False, verbose=True):
    print("Loading block = {}, phase = {}, history = {}".format(block, phase, history))
    x_path = os.path.join(data_dir, "bookcorpus", "cache", "tfidf_vectors_block{}_history{}_{}_x.npz".format(block, history, phase))
    y_path = os.path.join(data_dir, "bookcorpus", "cache", "tfidf_vectors_block{}_history{}_{}_y.npz".format(block, history, phase))

    if not redo and os.path.isfile(x_path) and os.path.isfile(y_path):
        x = sp.load_npz(x_path)
        y = sp.load_npz(y_path)
        print("x.shape =", x.shape, ", y.shape =", y.shape)
        return x, y

    book_list = get_book_split()[phase][:]
   
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "bookcorpus", "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=block)

    # build model
    x = []
    y = []
    for count, book_info in enumerate(book_list):
        print("\x1b[2K\rComputing tf-idf {:>5} / {:>5} [{:.2f}%]".format(
                count, len(book_list), 100.0*count/len(book_list)
            ), end="")
        content = []
        with open(book_info["path"], 'r', encoding='utf-8') as infile:
            book = json.load(infile)
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
            
            # generate x
            x_temp = []
            for i in range(0, history):
                x_temp.append(vectors[i:-history+i])
            x_temp = sp.hstack(x_temp, format="csr")
            x.extend(x_temp)
            
            # generate y
            y.extend(vectors[history:])
    print()

    x = sp.vstack(x)
    y = sp.vstack(y)
    get_size(x, "Block {}, x".format(block))
    get_size(y, "Block {}, y".format(block))
    sp.save_npz(x_path, x)
    sp.save_npz(y_path, y)

    return x, y

def load_tfidf_detail(phase="train", block=20, skip=0, redo=False, verbose=True):
    print("Loading block = {}, phase = {}, skip = {}".format(block, phase, skip))
    x_path = os.path.join(data_dir, "bookcorpus", "cache", "detail_tfidf_vectors_block{}_skip{}_{}_x.npz".format(block, skip, phase))
    y_path = os.path.join(data_dir, "bookcorpus", "cache", "detail_tfidf_vectors_block{}_skip{}_{}_y.npz".format(block, skip, phase))
    info_path = os.path.join(data_dir, "bookcorpus", "cache", "detail_tfidf_vectors_block{}_skip{}_{}_info.json".format(block, skip, phase))

    if not redo and os.path.isfile(x_path) and os.path.isfile(y_path) and os.path.isfile(info_path):
        x = sp.load_npz(x_path)
        y = sp.load_npz(y_path)
        with open(info_path, 'r', encoding='utf-8') as infile:
            info = json.load(infile)
        print("x.shape =", x.shape, ", y.shape =", y.shape, ", info.len = ", len(info))
        return x, y, info

    book_list = get_book_split()[phase][:100]
   
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "bookcorpus", "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=block)

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
                temp_position.append([book_info["start"]+story_start, book_info["start"]+story_end])

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

def load_text_data(phase="train", block=20, skip=0, redo=False, verbose=True, target_model="bert"):
    from transformers import LongformerTokenizerFast, BertTokenizerFast

    #print("Loading block = {}, phase = {}, skip = {}".format(block, phase, skip))
    x_path = os.path.join(data_dir, "bookcorpus", "cache", "{}_text_tfidf_vectors_block{}_skip{}_{}_x.parquet".format(target_model, block, skip, phase))
    y_path = os.path.join(data_dir, "bookcorpus", "cache", "{}_text_tfidf_vectors_block{}_skip{}_{}_y.npz".format(target_model, block, skip, phase))

    if not redo and os.path.isfile(x_path) and os.path.isfile(y_path):
        #x = feather.read_dataframe(x_path)["text"]
        data = pd.read_parquet(x_path)
        x = data["id"]
        text = data["text"]
        y = sp.load_npz(y_path)
        print("x.shape =", x.shape, ", y.shape =", y.shape)
        return text, x, y

    book_list = get_book_split()[phase][:]
   
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "bookcorpus", "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=block)

    # build model
    #tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    if target_model == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif target_model == "longerformer":
        tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    else:
        print(f"{target_model} not supported yet")

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
    
    return text, x["id"], y

def load_text_data_long(phase="train", block=20, history=1, redo=False, verbose=True, target_model="bert"):
    from transformers import LongformerTokenizerFast, BertTokenizerFast

    #print("Loading block = {}, phase = {}, skip = {}".format(block, phase, skip))
    x_path = os.path.join(data_dir, "bookcorpus", "cache", "{}_text_tfidf_vectors_block{}_history{}_{}_x.parquet".format(target_model, block, history, phase))
    y_path = os.path.join(data_dir, "bookcorpus", "cache", "{}_text_tfidf_vectors_block{}_history{}_{}_y.npz".format(target_model, block, history, phase))

    if not redo and os.path.isfile(x_path) and os.path.isfile(y_path):
        #x = feather.read_dataframe(x_path)["text"]
        data = pd.read_parquet(x_path)
        x = data["id"]
        text = data["text"]
        y = sp.load_npz(y_path)
        print("x.shape =", x.shape, ", y.shape =", y.shape)
        return text, x, y

    book_list = get_book_split()[phase][:]
   
    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, "bookcorpus", "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=block)

    # build model
    #tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    if target_model == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif target_model == "longerformer":
        tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    else:
        print(f"{target_model} not supported yet")

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
            #print(f"vectors.shape = {vectors.shape}")

            # generate x
            x_text_temp = []
            x_id_temp = []
            for i in range(0, history):
                x_text_temp.append(text[i:-history+i])
                x_id_temp.append(ids[i:-history+i])
            
            for temp in zip(*x_text_temp):
                x_text.append("\n".join(temp))
            #print(len(x_text))

            for temp in zip(*x_id_temp):
                x_id.append(np.hstack(temp))
            #print(len(x_id))
            #print(x_id)

            # generate y
            y.extend(vectors[history:])
            #break
    print()

    #x = sp.vstack(x)
    #get_size(x, "Block {}, x".format(block))
    #sp.save_npz(x_path, x)
    x = pd.DataFrame([x_text, x_id], index=["text", "id"])
    x = x.transpose()
    print("x.shape = ", x.shape)
    print("Block {}, x size = {:.2f} MB".format(block, x.memory_usage(index=False, deep=True)[0]/1024/1024))
    x.to_parquet(x_path)

    y = sp.vstack(y)
    get_size(y, "Block {}, y".format(block))
    sp.save_npz(y_path, y)

    return text, x["id"], y

def compute_avg_length():
    book_list = get_book_split()["train"][:]

    running_mean = 0
    running_count = 0
    for count, book_info in enumerate(book_list):
        content = []
        with open(book_info["path"], 'r', encoding='utf-8') as infile:
            book = json.load(infile)
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
    for block in [300, 500, 1000]:
        build_tfidf_model(block=block)

def batch_build_tfidf():
    for block in [300, 500, 1000]:
        load_tfidf(phase="train", block=block, redo=True)
        load_tfidf(phase="valid", block=block, redo=True)
        load_tfidf(phase="test", block=block, redo=True)

def batch_tfidf():
    for block in [5, 10, 20, 50, 100, 150, 200]:
        build_tfidf_model(block=block)
        load_tfidf(phase="train", block=block, redo=True)
        load_tfidf(phase="valid", block=block, redo=True)
        load_tfidf(phase="test", block=block, redo=True)

def test():
    old_x, old_y = old_load_tfidf(phase="train", block=200)
    x, y = load_tfidf(phase="train", block=200)
    return x, y, old_x, old_y

def build_tfidf_long():
    for block in [20, 50, 100, 200, 150, 10]:
    #for block in [150, 10]:
        for phase in ["train", "valid", "test"]:
            for history in [2, 5]:
                #load_tfidf_long(phase=phase, block=block, history=history)
                load_text_data_long(phase=phase, block=block, history=history, redo=True)

def main():
    build_tfidf_long()   
    #load_tfidf_long(phase="train", block=150, history=2)
    #load_text_data_long(phase="train", block=10, history=2)
    quit()

    #generate_split()
    batch_build_tfidf_model()
    batch_build_tfidf()
    #remove_header()
    #batch_tfidf()
    #compute_avg_length()
    #quit()
    for block in [300, 500, 1000]:
        load_text_data(phase="train", block=block, redo=True)
        load_text_data(phase="valid", block=block, redo=True)
        load_text_data(phase="test", block=block, redo=True)

if __name__ == "__main__":
    main()

