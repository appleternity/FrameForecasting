# FrameForecasting
This is the repo for paper Semantic Frame Forecasting (NAACL 2021).

## Dataset
Two of the datasets are used in our paper, Bookcorpus and CODA-19.
We will describe how to obtain the dataset and preprocess the code here.

### Bookcorpus
We provide the book list and their meta information in `data/bookcorpus/clean_split.json`.
Each of the field means:

- book: File name of the book.
- page_link: Link to the smashwords page.
- txt_link: Link to the `.txt` file on the smashwords page.
- genre: Genre of the book.
- title: Book name.
- author: Author name.
- start: The first line of the meaningful content (matched by regular expression).
- end: The last line of the meaningful content (matched by regular expression).

#### Getting book
Run the following script. This script will create a folder `data/bookcorpus/raw` and store all the books there.
```console
python get_books.py
```
#### Processing book
Segment the book using the following script. This script will create a folder `data/bookcorpus/segment` and store all the segemented books there.
```console
python segment_bookcorpus.py
```

#### Parsing Semantic Frame
Parse semantic frames using open-sesame (Plesase follow the open-sesame's instruction to set up all the model and data first). I put the python2 version of open-sesame in the third-party folder and implement a wrapper of both targetid and frameid modules. Notice that the following script needs **python2**. This script will create a folder `data/bookcorpus/frame` and store all the frame information there.
```console
python2 frame_parsing.py
```

#### Parsing Dependency
If you would like to run our implementation of the event-representation. You will need to parse the dependency using stanford's stanza parser.
You can parse it using the following script. This script will create a folder `data/bookcorpus/dependency` and store all the information there.
```console
python dependency_parsing.py [--gpu 0] [--start_index 0] [--end_index 100000]
```


### CODA-19
Under construction!


## Experiment
We implement several baselines.



## Using Frame-Forecasting in other places.
We have a wrapper that can take a piece of text and predict its following frame representation.


