from __future__ import print_function
from config import *
from datetime import datetime
import time
from unidecode import unidecode
from pprint import pprint
import json
import codecs
import re
import os
import sys

open_sesame_dir = os.path.join(root_dir, "third_party", "open_sesame")
os.chdir(open_sesame_dir)
sys.path.append(open_sesame_dir)
sys.path.append(os.path.join(open_sesame_dir, "sesame"))

from sesame.predict_targetid_wrapper import model as targetid_model
from sesame.predict_targetid_wrapper import load_instances as load_targetid_instances
from sesame.predict_targetid_wrapper import predict as predict_targetid
from sesame.predict_targetid_wrapper import print_as_conll

from sesame.predict_frameid_wrapper import model as frameid_model
from sesame.predict_frameid_wrapper import load_instances as load_frameid_instances
from sesame.predict_frameid_wrapper import predict as predict_frameid
from sesame.predict_frameid_wrapper import parse_conll

################################################
# This is the semantic frame parsing function
def parse_frame(text):
    # parse targetid
    text = unidecode(text)
    instances = load_targetid_instances(text)
    num_sentence = len(instances)
    predictions = predict_targetid(instances)
    conll = print_as_conll(instances, predictions)

    if conll.strip() == "":
        return [], "", 1

    # parse frameid
    instances = load_frameid_instances(conll.strip()+"\n")
    predictions = predict_frameid(instances)
    frames = parse_conll(instances, predictions)

    return frames, conll, num_sentence

###############################################
# batch
def load_fiction_list():
    with codecs.open(os.path.join(data_dir, "bookcorpus", "clean_split.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        data = [
            book_info
            for phase, book_info_list in data.items()
            for book_info in book_info_list
        ]
        return data

def batch_parsing():
    fiction_list = load_fiction_list()
    frame_dir = os.path.join(data_dir, "bookcorpus", "frame")
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)

    for i, info in enumerate(fiction_list, 1):
        print("\x1b[2K\r{} / {} [{:.2f}%]".format(i, len(fiction_list), 100.0*i/len(fiction_list)), end="")

        filename = os.path.join(data_dir, "bookcorpus", "segment", info["book"])
        if not os.path.isfile(filename): continue
        results = []
        with codecs.open(filename, 'r', encoding='utf-8') as infile:
            for line in infile:
                frames, _, _ = parse_frame(line)
                results.append({
                    "text": line,
                    "frame": frames,
                })

        with codecs.open(os.path.join(data_dir, "bookcorpus", "frame", info["book"]), 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=2)

def main():
    batch_parsing()

if __name__ == "__main__":
    main()
