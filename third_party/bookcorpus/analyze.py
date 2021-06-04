import json
from pprint import pprint
from collections import Counter
import os
from bs4 import BeautifulSoup
from langdetect import detect

def load_books_path_mapping():
    path = "/home/czh5679/workspace/bookcorpus/bookcorpus/corpus"
    filenames = os.listdir(path)
    mapping = {}
    for filename in filenames:
        _id = filename.split("__")[0]

        if os.path.getsize(os.path.join(path, filename)) < 10000:
            #print(filename)
            continue

        if _id in mapping:
            print("Duplicated id")
            quit()

        mapping[_id] = os.path.join(path, filename)

    return mapping

def check_html(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        text = infile.read()
    return bool(BeautifulSoup(text, "html.parser").find())

def check_html_text(text):
    return bool(BeautifulSoup(text, "html.parser").find())

def safe_list_get(array, index, placeholder=None):
    if index < len(array):
        return array[index]
    else:
        return placeholder

def check_non_fiction():
    all_counter = Counter()
    fiction_count = 0
    non_fiction_count = 0
    mapping = load_books_path_mapping()
    fiction_list = []
    non_fiction_list = []

    with open("url_list.jsonl", 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        for i, line in enumerate(lines, 1):
            if i % 100 == 0:
                print("\x1b[2K\r{} / {} [{:.2f}%]".format(i, len(lines), 100.0*i/len(lines)), end="")
            data = json.loads(line)

            # check file exist
            _id = data["page"].split("/")[-1]
            if _id not in mapping:
                continue
            
            # load content
            with open(mapping[_id], 'r', encoding='utf-8') as infile:
                text = infile.read()

            # check not html
            if check_html_text(text):
                continue

            # check not epub
            if data["txt"] == "":
                continue

            # check language
            lang = detect(text)
            if lang != "en":
                continue

            # process genres
            genres = [g.strip()[10:].split(" » ") for g in data["genres"]]

            # check not anthology
            if any(safe_list_get(g, 1)=="Anthologies" for g in genres):
                continue

            # check not graphic
            if any(safe_list_get(g, 1)=="Graphic novels & comics" for g in genres):
                continue

            # count fiction/non-fiction
            if all(g[0]=="Fiction" for g in genres):
                fiction_count += 1
                fiction_list.append([_id, mapping[_id], data["page"], genres])
            else:
                non_fiction_count += 1
                non_fiction_list.append([_id, mapping[_id], data["page"], genres])

            for genre in genres:
                all_counter.update(genre)
    print()

    print("Fiction Count = ", fiction_count)
    print("Non-Fiction Count = ", non_fiction_count)

    with open("generes_count.json", 'w', encoding='utf-8') as outfile:
        json.dump(all_counter, outfile, indent=2)

    with open("fiction_list.json", 'w', encoding='utf-8') as outfile:
        json.dump({"Fiction":fiction_list, "Non-Fiction":non_fiction_count}, outfile, indent=2)

def main():
    check_non_fiction()

if __name__ == "__main__":
    main()
