import json
import os
from config import *
import wget
import time

SLEEP_TIME = 0.3

def load_book_info():
    with open(os.path.join(data_dir, "bookcorpus", "clean_split.json"), 'r', encoding='utf-8') as infile:
        split_info = json.load(infile)

    book_info_list = [
        info
        for phase, info_list in split_info.items()
        for info in info_list
    ]
    return book_info_list

def download_files():
    book_info_list = load_book_info()
    print(f"There are a total of {len(book_info_list)} books.")
   
    os.makedirs(os.path.join(data_dir, "bookcorpus", "raw"), exist_ok=True)
    
    error_count = 0
    for i, book_info in enumerate(book_info_list, 1):
        print(f"\x1b[2K\rDownloading books {i} / {len(book_info_list)} [{100.0*i/len(book_info_list):.2f}%], Error count: {error_count}", end="")
        filepath = os.path.join(data_dir, "bookcorpus", "raw", book_info["book"])
        
        # check if file exists
        if os.path.isfile(filepath):
            continue
        try:
            wget.download(
                url=book_info["txt_link"],
                out=filepath
            )
        except Exception as e:
            error_count += 1

        if os.path.isfile(filepath):
            # check content
            with open(filepath, 'r', encoding='utf-8') as infile:
                content = infile.read()

            if "html" in content:
                os.remove(filepath)
                error_count += 1

        time.sleep(SLEEP_TIME) 

def main():
    download_files()

if __name__ == "__main__":
    main()
