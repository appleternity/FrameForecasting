import os
import platform
import json
import sys

if sys.version_info[0] >= 3:
    import pathlib
    root_dir = pathlib.Path(__file__).parent.parent.absolute()
else:
    root_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = "/"+os.path.join(*root_dir.split("/")[:-1])

data_dir = os.path.join(root_dir, "data")
model_dir = os.path.join(root_dir, "model")
output_dir = os.path.join(root_dir, "output")
result_dir = os.path.join(root_dir, "result")
history_dir = os.path.join(root_dir, "history")
predict_dir = os.path.join(root_dir, "predict")

# build folders
if sys.version_info[0] >= 3:
    for folder in [data_dir, model_dir, output_dir, result_dir, history_dir, predict_dir]:
        os.makedirs(folder, exist_ok=True)
    
    for data_name in ["bookcorpus", "coda19"]:
        os.makedirs(os.path.join(data_dir, data_name, "cache"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, data_name), exist_ok=True)
        os.makedirs(os.path.join(predict_dir, data_name), exist_ok=True)

START = 0
END = 1
UNK = 2

# load split
def get_book_split():
    with open(os.path.join(data_dir, "bookcorpus", "clean_split.json"), 'r', encoding='utf-8') as infile:
        book_split = json.load(infile)
        all_split = {}
        for key, book_info_list in book_split.items():
            info_list = []
            for book_info in book_info_list:
                book_info["path"] = os.path.join(data_dir, "bookcorpus", "frame", book_info["book"])
                if os.path.isfile(book_info["path"]):
                    info_list.append(book_info)
            all_split[key] = info_list[:100]
    return all_split

def get_coda_split():
    with open(os.path.join(data_dir, "coda19", "split.json"), 'r', encoding='utf-8') as infile:
        book_split = json.load(infile)
        for key, book_info_list in book_split.items():
            for book_info in book_info_list:
                book_info["path"] = os.path.join(data_dir, "coda19", "frame", book_info['book'])
    return book_split

