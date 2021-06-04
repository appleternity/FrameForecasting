import ujson as json
import re
from config import *
import os
from pprint import pprint
from collections import namedtuple, Counter
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import numpy as np
import h5py
import argparse

class Stemmer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def stem(self, word, tag=None):

        if tag in ['v', 'n', 'a', 'r']:
            return self.wnl.lemmatize(word, tag)
        temp = self.wnl.lemmatize(word, 'v') # check VERB
        if temp != word:
            return temp
        temp = self.wnl.lemmatize(word, 'n') # check NOUN
        if temp != word:
            return temp
        temp = self.wnl.lemmatize(word, 'a') # check ADJ
        if temp != word:
            return temp
        temp = self.wnl.lemmatize(word, 'r') # check ADV
        if temp != word:
            return temp

        return word

    def stemPOS(self, word, pos):
        if 'VERB' in pos:
            temp = self.wnl.lemmatize(word, 'v')
            return temp
        elif 'NOUN' in pos:
            temp = self.wnl.lemmatize(word, 'n')
            return temp
        elif 'ADJ' in pos:
            temp = self.wnl.lemmatize(word, 'a')
            return temp
        elif 'ADV' in pos:
            temp = self.wnl.lemmatize(word, 'r')
            return temp
        else:
            return word

my_stemmer = Stemmer()
Dep = namedtuple("Dep", ["rel", "gov_index", "gov", "dep_index", "dep"])
FeatureTuple = namedtuple("Tuple", ["s", "v", "o", "m"])

pattern = re.compile(r"\((?P<mode>.+?), (?P<p1>\d+?) - (?P<p2>.+?), (?P<p3>\d+?) - (?P<p4>.+?)\) ")
def extract_dep_list(dep_str):
    res = pattern.findall(dep_str+" ")
    res = [Dep(*dep) for dep in res]
    return res

pos_pattern = re.compile(r"\((?P<word>.+?) - (?P<pos>.+?)\)")
def extract_pos_list(pos_str):
    res = pos_pattern.findall(pos_str)
    return res

def extract(dep_list, pos_list, note="no_subj", verbose=False):
    if verbose:
        print()
        print(dep_list)
        print(pos_list)

    dep_info = {}
    for dep in dep_list:
        gov_index = int(dep.gov_index)
        dep_index = int(dep.dep_index)
        
        if gov_index not in dep_info:
            dep_info[gov_index] = []
        if dep_index not in dep_info:
            dep_info[dep_index] = []

        dep_info[gov_index].append(dep)
        dep_info[dep_index].append(dep)

    # find verb
    verb_list = [i for i, (token, pos) in enumerate(pos_list, 1) if pos == "VERB"]
    if verbose:
        print(verb_list)  

    # find dep
    subj_set = {"csubj", "csubj:pass", "nsubj", "nsubj:pass"}
    obj_set = {"obj", "iobj"}
    modifier_dict = {
        "advmod":0,
        "advcl":1,
        "obl":2,
        "xcomp":3,
        "ccomp":4,
        "compound":5,
        "acl":6,
        "parataxis":7,
    }
    results = []
    for verb_index in verb_list:
        related_list = dep_info[verb_index]
        tuple_list = []
        verb = pos_list[verb_index-1] 
        stemmed_verb = my_stemmer.stemPOS(verb[0], verb[1])

        # find obj
        obj_list = []
        for dep in related_list:
            if dep.rel in obj_set:
                target = pos_list[int(dep.dep_index)-1]
                obj = my_stemmer.stemPOS(target[0], target[1])
                obj_list.append(obj)

        if not obj_list:
            obj_list.append("<EMPTY_OBJ>")

        # find modifier
        modifier_list = [(modifier_dict[dep.rel], dep) for dep in related_list if dep.rel in modifier_dict]
        modifier_list = sorted(modifier_list, key=lambda x: x[0])
        if modifier_list:
            dep = modifier_list[0][1]
            if dep.dep != verb[0]:
                target = pos_list[int(dep.dep_index)-1]
                mod = my_stemmer.stemPOS(target[0], target[1])
                modifier = mod
            else:
                target = pos_list[int(dep.gov_index)-1]
                mod = my_stemmer.stemPOS(target[0], target[1])
                modifier = mod
        else:
            modifier = "<EMPTY_MODIFIER>"

        # need at least subj 
        for dep in related_list:
            if dep.rel in subj_set:
                for obj in obj_list:
                    target = pos_list[int(dep.dep_index)-1]
                    subj = my_stemmer.stemPOS(target[0], target[1])
                    tuple_list.append(
                        FeatureTuple(subj.lower(), stemmed_verb.lower(), obj.lower(), modifier.lower())
                    )
        
        # handle cases of no subj
        if note == "subj":
            if not tuple_list:
                for obj in obj_list:
                    tuple_list.append(
                        FeatureTuple("<EMPTY_SUBJ>".lower(), stemmed_verb.lower(), obj.lower(), modifier.lower())
                    )

        results.extend(tuple_list)

    return results

def extract_tuple(filename, data_name="bookcorpus", note="no_subj", verbose=False):
    event_file_path = os.path.join(data_dir, data_name, f"event_{note}", filename)

    # if exist, load it from file
    if os.path.isfile(event_file_path):
        with open(event_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
            return data

    # if not exist, extract!
    with open(os.path.join(data_dir, data_name, "dependency", filename), 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    res = []
    for sent in data[:]:
        if sent["dep_list"] is None:
            res.append([])
        elif type(sent["dep_list"]) == str:
            dep_list = extract_dep_list(sent["dep_list"])
            pos_list = extract_pos_list(sent["pos_list"])
            tuple_list = extract(dep_list, pos_list)
            res.append(tuple_list)
        else:
            temp = []
            try:
                for dep_list, pos_list in zip(sent["dep_list"], sent["pos_list"]):
                    dep_list = extract_dep_list(dep_list)
                    pos_list = extract_pos_list(pos_list)
                    tuple_list = extract(dep_list, pos_list)
                    temp.extend(tuple_list)
            except TypeError as e:
                print(e)
                pprint(sent)
                quit()
            res.append(temp)

    with open(event_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(res, outfile, indent=2)

    return res

def tokenizer(x):
    return x.split(" ")

def process_event_tfidf_data(phase="train", block=20, history=1, note="no_subj", data_name="bookcorpus"):
    from scipy import sparse as sp
    import joblib
    
    os.makedirs(os.path.join(data_dir, data_name, f"event_{note}"), exist_ok=True)

    # load tfidf model
    tfidf_model_path = os.path.join(data_dir, data_name, "tfidf_model", "block_{}.joblib".format(block))
    if os.path.isfile(tfidf_model_path):
        model = joblib.load(tfidf_model_path)
    else:
        build_tfidf_model(block=20)
    
    with open(os.path.join(data_dir, data_name, "clean_split.json"), 'r', encoding='utf-8') as infile:
        book_list = json.load(infile)[phase][:10]

    final_data = []
    x = []
    y = []
    for c, book_info in enumerate(book_list, 1):
        print("\x1b[2K\rComputing tf-idf {:>5} / {:>5} [{:.2f}%]".format(c, len(book_list), 100.0*c/len(book_list)), end="")
        frame_path = os.path.join(data_dir, data_name, "frame", book_info["book"])
        if not os.path.isfile(frame_path):
            continue

        event_path = os.path.join(data_dir, data_name, "dependency", book_info["book"])
        if not os.path.isfile(event_path):
            continue

        # load event data
        event_data = extract_tuple(book_info["book"], data_name=data_name, note=note)

        with open(frame_path, 'r', encoding='utf-8') as infile:
            book = json.load(infile)
            if book_info["start"] != -1:
                book = book[book_info["start"]:book_info["end"]]
                event_data = event_data[book_info["start"]:book_info["end"]]
            if len(book) != len(event_data):
                print(book_info["book"], c)
            assert len(book) == len(event_data)

            content = []
            event_list = []
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

                # event data
                events = [
                    event
                    for sent in event_data[story_start:story_end]
                    for event in sent
                ]
                event_list.append(events)

            if content:
                vectors = model.transform(content)
                assert vectors.shape[0] == len(event_list)

                # generate x
                x_temp = []
                for i in range(0, history):
                    x_temp.append(event_list[i:-history+i])
                
                for temp in zip(*x_temp):
                    x.append([tt for t in temp for tt in t])

                # generate y
                y.extend(vectors[history:])
    print()
    x = [[list(e) for e in event] for event in x]
    y = sp.vstack(y)
    print(len(x))
    print(y.shape)
    avg_event_num = np.array([len(events) for events in x])
    avg_event_num = float(avg_event_num.mean())
    print("avg_event_num", avg_event_num)

    if history == 1:
        sp.save_npz(os.path.join(data_dir, data_name, "event-{}_block{}_{}_y.npz".format(note, block, phase)), y)
        with open(os.path.join(data_dir, data_name, "event-{}_block{}_{}_x.json".format(note, block, phase)), 'w', encoding='utf-8') as outfile:
            json.dump(x, outfile)
    else:
        sp.save_npz(os.path.join(data_dir, data_name, "event-{}_block{}_{}_h{}_y.npz".format(note, block, phase, history)), y)
        with open(os.path.join(data_dir, data_name, "event-{}_block{}_{}_h{}_x.json".format(note, block, phase, history)), 'w', encoding='utf-8') as outfile:
            json.dump(x, outfile)
    print()

def parse_args():
    parser = argparse.ArgumentParser(description="Data Pre-processing for Event-Representation Baseline.")
    parser.add_argument("--block", help="story block size", type=int, default=20)
    parser.add_argument("--data", help="Corpus used for training and testing [bookcorpus/coda19]", type=str, default="bookcorpus")
    parser.add_argument("--history", help="Number of story blocks used for input", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    for phase in ["train", "valid", "test"]:
        process_event_tfidf_data(
            phase=phase, 
            block=args.block, 
            history=args.history, 
            note="no_subj", 
            data_name=args.data,
        )

if __name__ == "__main__":
    main()



