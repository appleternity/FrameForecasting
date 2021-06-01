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

### Naive Baselines: Replay and Prior
```console
usage: naive_baseline.py [-h] [--device DEVICE] [--block BLOCK] [--data DATA]
                         [--model MODEL] [--downsample DOWNSAMPLE]
                         [--skip SKIP]

Naive Baseline.

optional arguments:
  -h, --help               show this help message and exit
  --device DEVICE          device used for computing tfidf [cpu/cuda:0/cuda:1]
  --block BLOCK            story block size
  --data DATA              Corpus used for training and testing [bookcorpus/coda19]
  --model MODEL            Model type [replay/prior]
  --downsample DOWNSAMPLE  Downsampling size
  --skip SKIP              Skipping distance for replay baseline
```

To run the replay baseline, use the following command. You can adjust the value of `skip` to run the replay baseline with a longer distance.
```console
python naive_baseline.py --block 20 --data bookcorpus --model replay --device cuda:0 [--skip 0]
```

To run the prior baseline, use the following command.
```console
python naive_baseline.py --block 20 --data bookcorpus --model prior --device cuda:0 [--downsample 88720]
```

### IR Baseline
```console
usage: ir_baseline.py [-h] [--device DEVICE] [--block BLOCK] [--data DATA]
                      [--downsample DOWNSAMPLE]

IR Baseline.

optional arguments:
  -h, --help                show this help message and exit
  --device DEVICE           device used for computing tfidf [cpu/cuda:0/cuda:1]
  --block BLOCK             story block size
  --data DATA               Corpus used for training and testing [bookcorpus/coda19]
  --downsample DOWNSAMPLE   Downsampling size
```

To run the ir baseline, use the following command. Since the data is huge, it will require gpu to accerlate. Please specify the gpu resource using the device flag. 
```console
python ir_baseline.py --block 20 --data bookcorpus --device cuda:0 [--downsample 88720]
```

### ML Baseline: LGBM & RandomForest
```console
usage: ml_baseline.py [-h] [--device DEVICE] [--block BLOCK] [--data DATA]
                      [--downsample DOWNSAMPLE] [--history HISTORY]
                      [--n_jobs N_JOBS] [--model MODEL]

ML Baseline (LGBM / RandomForest).

optional arguments:
  -h, --help                show this help message and exit
  --device DEVICE           device used for computing tfidf [cpu/cuda:0/cuda:1]
  --block BLOCK             story block size
  --data DATA               Corpus used for training and testing [bookcorpus/coda19]
  --downsample DOWNSAMPLE   Downsampling size
  --history HISTORY         Number of story blocks used for input
  --n_jobs N_JOBS           Processes used for computing
  --model MODEL             ML model. (LGBM / RandomForest)
```

To run the ml baseline, use the following command. The default history flag is `None` which will only use **one** previous story block as the input feature. You can specify it to a number `n` to use n story blocks. The hyper-parameters are hard-coded in the script but feel free to change if needed.
```console
python ml_baseline.py --block 20 --data bookcorpus --device cuda:0 --model LGBM [--history 2] [--n_jobs 10] [--downsample 88720]
```

We also have an ablation study script that is based on `ml_baseline.py`. You can use the following command to run the ablation study.
The `removed_dim` is the dimension that you want to remove for the ablation study.
```console
python ablation_exp.py --block 20 --data bookcorpus --device cuda:0 --model LGBM --removed_dim 10 [--n_jobs 10]
```
To run the same experiment using the same setting in our paper, use the following command.
```console
python ablation_exp.py
```

where each argument mean:
```console
usage: ablation_exp.py [-h] [--device DEVICE] [--block BLOCK] [--data DATA]
                       [--n_jobs N_JOBS] [--model MODEL]
                       [--removed_dim REMOVED_DIM]

Ablation Study for ML Baseline.

optional arguments:
  -h, --help                  show this help message and exit
  --device DEVICE             device used for computing tfidf [cpu/cuda:0/cuda:1]
  --block BLOCK               story block size
  --data DATA                 Corpus used for training and testing [bookcorpus/coda19]
  --n_jobs N_JOBS             Processes used for computing
  --model MODEL               ML model. [LGBM / RandomForest]
  --removed_dim REMOVED_DIM   Dimention you want to remove.
```

### DAE Baseline
```console
usage: dae_baseline.py [-h] [--device DEVICE] [--block BLOCK] [--data DATA]
                       [--downsample DOWNSAMPLE] [--history HISTORY]
                       [--hidden_size HIDDEN_SIZE] [--layer_num LAYER_NUM]
                       [--dropout_rate DROPOUT_RATE] [--epoch_num EPOCH_NUM]
                       [--batch_size BATCH_SIZE]
                       [--learning_rate LEARNING_RATE]
                       [--early_stop EARLY_STOP]

DAE Baseline.

optional arguments:
  -h, --help                      show this help message and exit
  --device DEVICE                 device used for computing tfidf and training [cpu/cuda:0/cuda:1]
  --block BLOCK                   story block size
  --data DATA                     Corpus used for training and testing [bookcorpus/coda19]
  --downsample DOWNSAMPLE         Downsampling size
  --history HISTORY               Number of story blocks used for input
  --hidden_size HIDDEN_SIZE       Hidden size of the DAE model
  --layer_num LAYER_NUM           Number of layers of the DAE model
  --dropout_rate DROPOUT_RATE     Dropout rate for the DAE model
  --epoch_num EPOCH_NUM           Number of training epoch
  --batch_size BATCH_SIZE         Batch size for training
  --learning_rate LEARNING_RATE   Learning rate for training
  --early_stop EARLY_STOP         Number of epoch for early stop
```

Use the following command to run the model. Default hyper-parameters are all set up in the argparser but feel free to change it if needed.
```console
python dae_baseline.py --block 20 --device cuda:0 --data bookcorpus --downsample 1000 --epoch_num 10
```


### BERT Baseline



### GPT-2 Baseline



## Using Frame-Forecasting in other places.
We have a wrapper that can take a piece of text and predict its following frame representation.


