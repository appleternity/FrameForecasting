import json
import os

from config import *
from collections import Counter
from word2number import w2n
from nltk import word_tokenize

def update_boundary():
    with open(os.path.join(data_dir, "bookcorpus", "clean_split.json"), 'r', encoding='utf-8') as infile:
        book_info_dict = json.load(infile)

    # check all books
    bookcorpus_dir = os.path.join(data_dir, "bookcorpus", "segment")
    counter = Counter()
    last_counter = Counter()
    book_info = []
    for phase, book_info_list in book_info_dict.items():
        for count, book_info in enumerate(book_info_list):
            book_name = book_info["book"]
            print("\x1b[2K\rRemoving Header in {}, {:>5} / {:>5} [{:.2f}%]".format(
                    phase, count, len(book_info_list), 100.0*count/len(book_info_list)
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

                # cannot find chapter 1
                if chapter_one == -1:
                    continue
                # remove the wired cases where chapter 1 come after more than 300 lines
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
                    book_info["start"] = start
                    book_info["end"] = end

    # save data
    with open(os.path.join(data_dir, "bookcorpus", "clean_split_updated.json"), 'w', encoding='utf-8') as outfile:
        json.dump(book_info_dict, outfile, indent=2)

def main():
    update_boundary()

if __name__ == "__main__":
    main()
