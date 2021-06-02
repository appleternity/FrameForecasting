import stanza
import os
from config import *
import argparse
import ujson as json
import traceback
from datetime import datetime
import numpy as np

class Parser:
    def __init__(self):
        self.nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

    def get_dep_list(self, words):
        #for word in words:
        #    print(word)
        #    print(word["deprel"]) 

        dep_list = " | ".join([
            "({}, {} - {}, {} - {})".format(
                word.deprel, 
                word.head,
                words[word.head-1].text if word.head > 0 else "ROOT",
                word.id,
                word.text,
            )
            for word in words
        ])
        return dep_list

    def parse(self, sent):
        doc = self.nlp(sent)

        if len(doc.sentences) == 1:
            sent = doc.sentences[0]
            dep_list = self.get_dep_list(sent.words)
        elif len(doc.sentences) == 0:
            print("GG si mi da")
            dep_list = None
        else:
            dep_list = [
                self.get_dep_list(sent.words) for sent in doc.sentences
            ]

        return dep_list

class AllParser:
    def __init__(self):
        self.nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

    def get_dep_list(self, words):
        #for word in words:
        #    print(word)
        #    print(word["deprel"])

        dep_list = " | ".join([
            "({}, {} - {}, {} - {})".format(
                word.deprel,
                word.head,
                words[word.head-1].text if word.head > 0 else "ROOT",
                word.id,
                word.text,
            )
            for word in words
        ])
        return dep_list

    def get_pos(self, words):
        pos_list = " | ".join([
            "({} - {}),".format(
                word.text, word.pos
            )
            for word in words
        ])
        return pos_list

    def parse(self, sent):
        doc = self.nlp(sent)

        if len(doc.sentences) == 1:
            sent = doc.sentences[0]
            dep_list = self.get_dep_list(sent.words)
            pos_list = self.get_pos(sent.words)
        elif len(doc.sentences) == 0:
            print("GG si mi da")
            dep_list = None
            pos_list = None
        else:
            dep_list = [
                self.get_dep_list(sent.words) for sent in doc.sentences
            ]
            pos_list = [
                self.get_pos(sent.words) for sent in doc.sentences
            ]

        return dep_list, pos_list

class PosParser:
    def __init__(self):
        self.nlp = stanza.Pipeline("en", processors="tokenize,pos")

    def get_pos(self, words):
        pos_list = " | ".join([
            "({} - {}),".format(
                word.text, word.pos
            )
            for word in words
        ])
        return pos_list

    def parse(self, sent):
        try:
            doc = self.nlp(sent)
        except Exception as e:
            print()
            print(e)
            return None

        doc.sentences[0].words
        if len(doc.sentences) == 1:
            pos_list = self.get_pos(doc.sentences[0].words)
        elif len(doc.sentences) == 0:
            print("GG si mi da")
            pos_list = None
        else:
            pos_list = [
                self.get_pos(sent.words) for sent in doc.sentences        
            ]
        return pos_list

def log_error(e, sent, book_id):
    error_message = traceback.format_exc()
    with open(os.path.join(history_dir, "new_book_error_parsing.log"), 'a', encoding='utf-8') as outfile:
        outfile.write("\n=======================================\n")
        outfile.write(str(datetime.now()) + "\n")
        outfile.write(error_message + "\n")
        outfile.write("Book ID: " + str(book_id) + "\n")
        outfile.write("DATA: " + sent)

def load_clean_fiction_list():
    os.makedirs(os.path.join(data_dir, "bookcorpus", "dependency"), exist_ok=True)
    with open(os.path.join(data_dir, "bookcorpus", "clean_split.json"), 'r', encoding='utf-8') as infile:
        book_split = json.load(infile)
        fiction_list = []
        for key, book_info_list in book_split.items():
            for book_info in book_info_list:
                fiction_list.append({
                    "book_id": book_info["book"].split("__")[0],
                    "path": os.path.join(data_dir, "bookcorpus", "segment", book_info["book"]),
                    "output_path": os.path.join(data_dir, "bookcorpus", "dependency", book_info["book"])
                })
    print("total fiction num = ", len(fiction_list))
    return fiction_list

def book_parse_arg():
    parser = argparse.ArgumentParser(description="Stanford Parser.")
    parser.add_argument("--start_index", dest="start_index", help="Integer", type=int, default=0)
    parser.add_argument("--end_index", dest="end_index", help="Integer", type=int, default=100000)
    parser.add_argument("--gpu", dest="gpu", help="gpu id", type=int, default=0)
    return parser.parse_args()

def bookcorpus_parse_all():
    # get variable
    args = book_parse_arg()
   
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

    # data & partition
    fiction_list = load_clean_fiction_list()
    fiction_list = fiction_list[args.start_index:args.end_index]

    parser = AllParser()

    error_count = 0
    time_info = []
    for c, fiction_info in enumerate(fiction_list, 1):
        # skip un-existed books
        if not os.path.isfile(fiction_info["path"]):
            continue

        # skip the ones that have been parsed
        if os.path.isfile(fiction_info["output_path"]):
            continue
        
        # parsing
        start_time = datetime.now()
        if time_info:
            avg_time = np.array(time_info).mean()
        else:
            avg_time = 0
        with open(fiction_info["path"], 'r', encoding='utf-8') as infile:
            data = infile.read().split("\n")

            data_length = len(data)
            parsed_data = []
            for dc, sent in enumerate(data, 1):
                if dc % 10 == 0:
                    print("\x1b[2K\r {:>5}-{:>5}/{:>5} [{:.2f}%] -> [{:.2f}%] ({}) [{}] Avg:{:.3f} sec".format(
                            args.start_index + c,
                            c, 
                            len(fiction_list), 
                            100.0*c/len(fiction_list), 
                            100.0*dc/data_length, 
                            error_count,
                            datetime.now().strftime("%d-%H:%M:%S"),
                            avg_time), 
                        end="")
                try:
                    dep, pos = parser.parse(sent)
                except Exception as e:
                    error_count += 1
                    log_error(e, sent["text"], index)
                    dep, pos = None, None

                parsed_data.append({
                    "dep_list": dep,
                    "pos_list": pos,
                    "text": sent,
                })

        with open(fiction_info["output_path"], 'w', encoding='utf-8') as outfile:
            json.dump(parsed_data, outfile, indent=2)
       
        end_time = datetime.now()
        time_info.append((end_time-start_time).total_seconds())

    print()

def test_parser():
    parser = Parser()
    test_cases = [
        "Barack Obama was born in Hawaii.",
        "Three Plays Published by Mike Suttons at Smashwords Copyright 2011 Mike Sutton ISBN 978-1-4659-8486-9.",
        "Character 1 standing downstage-center, performing the audience through a monologue.",
    ]
    
    for count, test_case in enumerate(test_cases):
        res = parser.parse(test_case)
        print("\n==========================================")
        print(count)
        print(test_case)
        print(res)

def test_pos_parser():
    parser = PosParser()
    test_cases = [
        "Barack Obama was born in Hawaii.",
        "Three Plays Published by Mike Suttons at Smashwords Copyright 2011 Mike Sutton ISBN 978-1-4659-8486-9.",
        "Character 1 standing downstage-center, performing the audience through a monologue.",
    ]
    
    for count, test_case in enumerate(test_cases):
        res = parser.parse(test_case)
        print("\n==========================================")
        print(count)
        print(test_case)
        print(res)


def test():
    #stanza.download("en")
    nlp = stanza.Pipeline("en", processors="tokenize,pos")
    test_cases = [
        "Eugene (sits down on the other couch and sighs) Thanks, Ill catch him then.",
        "Barack Obama was born in Hawaii.",
        "Three Plays Published by Mike Suttons at Smashwords Copyright 2011 Mike Sutton ISBN 978-1-4659-8486-9.",
        "Character 1 standing downstage-center, performing the audience through a monologue.",
    ]
    test_case = "\n\n".join(test_cases)
    doc = nlp(test_case)
    print(doc)
    return doc

if __name__ == "__main__":
    bookcorpus_parse_all()

