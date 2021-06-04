from __future__ import print_function
from frame_parsing import *
import json
from nltk import sent_tokenize
import argparse
import os
from config import *
import codecs

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Frame Parsing for GPT2 Stories.")
    parser.add_argument("--block", help="Size of the story block", type=int, default=20)
    return parser.parse_args()

def load_stories(block=20):
    with codecs.open(os.path.join(predict_dir, "generation_result_block{}.json".format(block)), 'r', encoding='utf-8') as infile:
        results = json.load(infile)
    return results

def parse_gpt_story(block=20):
    data = load_stories(block=block)
    
    for count, story in enumerate(data, 1):
        print("\x1b[2K\r{} / {} [{:.2f}%]".format(count, len(data), 100.0*count/len(data)), end="")
        lines = sent_tokenize(story["res"])
        results = []
        for line in lines:
            frames, _, _ = parse_frame(line)
            results.extend(frames)
        story["frames"] = results

    with codecs.open(os.path.join(predict_dir, "generation_result_block{}_parsed.json".format(block)), 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2)

def main():
    args = parse_args()
    parse_gpt_story(block=args.block)

if __name__ == "__main__":
    main()
