from config import *
from blingfire import text_to_sentences
import os
import json
from pprint import pprint
import re
from unidecode import unidecode

def load_fiction_list():
    with open(os.path.join(data_dir, "bookcorpus", "clean_split.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        data = [
            book_info
            for phase, book_info_list in data.items()
            for book_info in book_info_list
        ]
    return data 

pattern = re.compile(r"\n\n+")
def process_text_to_sentence(text):
    blocks = pattern.split(text)
    results = []
    for block in blocks:
        sents = text_to_sentences(block.strip("\n").replace("\n", " ")).split("\n")
        sents = [sent for sent in sents if sent != ""]
        results.extend(sents)
    return results

def sent_segmentation():
    fiction_list = load_fiction_list()
    fiction_list = fiction_list[:]

    os.makedirs(os.path.join(data_dir, "bookcorpus", "segment"), exist_ok=True)
    for i, info in enumerate(fiction_list):
        print("\x1b[2K\r{} / {} [{:.2f}%]".format(i, len(fiction_list), 100.0*i/len(fiction_list)), end="")
        
        with open(info["path"], 'r', encoding='utf-8-sig') as infile:
            text = infile.read().strip()
            text = unidecode(text)

        sentences = process_text_to_sentence(text)

        filename = info["path"].split("/")[-1]
        with open(os.path.join(data_dir, "bookcorpus", "segment", filename), 'w', encoding='utf-8') as outfile:
            out_text = "\n".join(sentences)
            out_text = out_text.replace(". . .", "...")
            outfile.write(out_text)

def main():
    sent_segmentation()

if __name__ == "__main__":
    main()
