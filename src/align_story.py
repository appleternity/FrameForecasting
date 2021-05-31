from config import *
import os
import json
from blingfire import text_to_sentences
import traceback
from collections import Counter

def process_to_sentence(lines):
    temp_sentences = []
    sentences = []
    for line in lines:
        if not line.strip():
            if temp_sentences:
                sents = text_to_sentences(
                    " ".join(temp_sentences).strip().replace("\n", " ")
                ).split("\n")
                sentences.extend(sents)
                temp_sentences = []
        else:
            temp_sentences.append(line.strip())

    return sentences

def get_story_raw_mapping():
    mapping = []
    dir_path = "/dgxhome/czh5679/workspace/StoryNet/data/home/henry0122/BERT-pytorch-master/data/bookcorpus/corpus"
    filelist = os.listdir(dir_path)
    for i, filename in enumerate(filelist[:]):
        print("\x1b[2K\r{}/{} [{:.2f}%]".format(i, len(filelist), 100.0*i/len(filelist)), end="")
        try:
            with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as infile:
                data = infile.read().split("\n")[:1000]
                sentences = process_to_sentence(data)
                file_index = filename.split("_")[0]
                #mapping["|||||".join(sent for sent in sentences[:50])] = file_index
                mapping.append(["|||||".join(sent for sent in sentences[:50]), file_index])
        except Exception:
            traceback.print_exc()
    print()
    print("story raw mapping = ", len(mapping))
    return mapping

def get_story_processed_mapping():
    mapping = []
    filename_template = "/dgxhome/czh5679/workspace/StoryNet/data/books/book_{}.json"
    for i in range(0, 11000):
    #for i in range(0, 11):
        filename = os.path.join(filename_template.format(i))
        if not os.path.isfile(filename):
            continue
        print("\x1b[2K\r{}/{} [{:.2f}%] mapping length = {}".format(i, 11000, 100.0*i/11000, len(mapping)), end="")
        with open(filename, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
            key = "|||||".join(d["text"] for d in data[:50])
            mapping.append([i, key])
    print()
    print("story processed mapping = ", len(mapping))
    return mapping

def get_meta_data():
    mapping = {}
    with open(os.path.join(data_dir, "url_list.jsonl"), 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            key = data["page"].split("/")[-1]
            mapping[key] = data
    print("meta mapping = ", len(mapping))
    return mapping

def align():
    with open("alignment_mapping.json", 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        meta_mapping = data["meta_mapping"]
        processed_story_mapping = data["processed_story_mapping"]
        raw_story_mapping = data["raw_story_mapping"]

    # check repeated
    counter = Counter()
    for val in processed_story_mapping.values():
        counter.update([val.split("|||||")[0]])
    for key, freq in counter.items():
        if freq > 1:
            print(key)

    # check first sentence
    print("processed_story_mapping", len(processed_story_mapping), len(set(processed_story_mapping.values())))
    processed_story_mapping = {
        #key : val.split("|||||")[0]
        key : val
        for key, val in processed_story_mapping.items()        
    }
    print("processed_story_mapping", len(processed_story_mapping), len(set(processed_story_mapping.values())))
    
    print("raw_story_mapping", len(raw_story_mapping))
    raw_story_mapping = {
        #key.split("|||||")[0] : val
        key : val
        for key, val in raw_story_mapping.items()        
    }
    print("raw_story_mapping", len(raw_story_mapping))


    # for those who has unique 1st sentence
    pass


def main():
    meta_mapping = get_meta_data()
    processed_story_mapping = get_story_processed_mapping()
    raw_story_mapping = get_story_raw_mapping()

    with open("alignment_mapping.json", 'w', encoding='utf-8') as outfile:
        json.dump({
            "meta_mapping":meta_mapping,
            "processed_story_mapping":processed_story_mapping,
            "raw_story_mapping": raw_story_mapping,
        }, outfile, indent=2)

    #for i, (key, val) in enumerate(raw_story_mapping.items()):
    #    print(key, val)
    #    print()
    #    if i == 11:
    #        break

    #print("===================================================") 
    #for i, (key, val) in enumerate(processed_story_mapping.items()):
    #    print(key, val)
    #    print()


if __name__ == "__main__":
    main()
    #align()
