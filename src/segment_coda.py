import os
import json

def build_list():
    data_dir = "/dgxhome/czh5679/workspace/StoryNet/data/coda19/CODA19_v1_20200504/human_label"
    output_dir = "/dgxhome/czh5679/workspace/StoryNet/data/coda19/segment"
    info = []
    for data_split in ["train", "dev", "test"]:
        filenames = [f for f in os.listdir(os.path.join(data_dir, data_split)) if ".swp" not in f]
        for filename in filenames[:]:
            filepath = os.path.join(os.path.join(data_dir, data_split, filename))
            if os.path.isdir(filepath):
                continue
            
            with open(filepath, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                info.append({
                    "metadata": data["metadata"],
                    "paper_id": data["paper_id"],
                    "split": data_split,
                })

    with open("/dgxhome/czh5679/workspace/StoryNet/data/coda_list.json", 'w', encoding='utf-8') as outfile:
        json.dump(info, outfile, indent=2)

def extract_data():
    data_dir = "/dgxhome/czh5679/workspace/StoryNet/data/coda19/CODA19_v1_20200504/human_label"
    output_dir = "/dgxhome/czh5679/workspace/StoryNet/data/coda19/segment"
    #for data_split in ["train", "dev", "test"]:
    for data_split in ["test"]:
        filenames = os.listdir(os.path.join(data_dir, data_split))
        for filename in filenames[:]:
            filepath = os.path.join(os.path.join(data_dir, data_split, filename))
            if os.path.isdir(filepath):
                continue
            # load data
            with open(filepath, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                sents = "\n".join([
                    " ".join([
                        segment["segment_text"]
                        for segment in sentence
                        if segment["crowd_label"] != "other"
                    ])

                    for paragraph in data["abstract"]    
                    for sentence in paragraph["sentences"]
                ])

            with open(os.path.join(output_dir, filename.replace(".json", ".txt")), 'w', encoding='utf-8') as outfile:
                outfile.write(sents)

def main():
    #extract_data()
    build_list()

if __name__ == "__main__":
    main()
