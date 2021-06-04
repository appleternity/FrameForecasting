# -*- coding: utf-8 -*-
import ujson as json
import os
import sys
import time
from optparse import OptionParser

from dynet import *
from evaluation import *
from raw_data import make_data_instance
from semafor_evaluation import convert_conll_to_frame_elements
import dynet_config
from datetime import datetime
import numpy as np
import traceback
dynet_config.set_gpu("GPU:0")

############################################
# functions
def identify_frames(builders, tokens, postags, lexunit, targetpositions, goldframe=None):
    renew_cg()
    trainmode = (goldframe is not None)

    sentlen = len(tokens) - 1
    emb_x = [v_x[tok] for tok in tokens]
    pos_x = [p_x[pos] for pos in postags]

    emb2_xi = []
    for i in xrange(sentlen + 1):
        if tokens[i] in pretrained_embeddings_map:
            # If update set to False, prevents pretrained embeddings from being updated.
            emb_without_backprop = lookup(e_x, tokens[i], update=True)
            features_at_i = concatenate([emb_x[i], pos_x[i], emb_without_backprop])
        else:
            features_at_i = concatenate([emb_x[i], pos_x[i], u_x])
        emb2_xi.append(w_e * features_at_i + b_e)

    emb2_x = [rectify(emb2_xi[i]) for i in xrange(sentlen+1)]

    # initializing the two LSTMs
    if USE_DROPOUT and trainmode:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)
    f_init, b_init = [i.initial_state() for i in builders]

    fw_x = f_init.transduce(emb2_x)
    bw_x = b_init.transduce(reversed(emb2_x))

    # only using the first target position - summing them hurts :(
    targetembs = [concatenate([fw_x[targetidx], bw_x[sentlen - targetidx - 1]]) for targetidx in targetpositions]
    targinit = tlstm.initial_state()
    target_vec = targinit.transduce(targetembs)[-1]

    valid_frames = list(lufrmmap[lexunit.id])
    chosenframe = valid_frames[0]
    logloss = None
    if len(valid_frames) > 1:
        if USE_HIER and lexunit.id in relatedlus:
            lu_vec = esum([lu_x[luid] for luid in relatedlus[lexunit.id]])
        else:
            lu_vec = lu_x[lexunit.id]
        fbemb_i = concatenate([target_vec, lu_vec, lp_x[lexunit.posid]])
        # TODO(swabha): Add more Baidu-style features here.
        f_i = w_f * rectify(w_z * fbemb_i + b_z) + b_f
        if trainmode and USE_DROPOUT:
            f_i = dropout(f_i, DROPOUT_RATE)
        
        f_i = to_device(f_i, "CPU")
        logloss = log_softmax(f_i, valid_frames)

        if not trainmode:
            chosenframe = np.argmax(logloss.npvalue())

    if trainmode:
        chosenframe = goldframe.id

    losses = []
    if logloss is not None:
        losses.append(pick(logloss, chosenframe))

    prediction = {tidx: (lexunit, Frame(chosenframe)) for tidx in targetpositions}

    objective = -esum(losses) if losses else None
    return objective, prediction

def print_as_conll(goldexamples, pred_targmaps, output_file):
    with codecs.open(output_file, "w", "utf-8") as f:
        output_text = ""
        for g,p in zip(goldexamples, pred_targmaps):
            result = g.get_predicted_frame_conll(p) + "\n"
            output_text += result
        f.write(output_text)
        f.close()

def find_multitokentargets(examples, split):
    multitoktargs = tottargs = 0.0
    for tr in examples:
        tottargs += 1
        if len(tr.targetframedict) > 1:
            multitoktargs += 1
            tfs = set(tr.targetframedict.values())
            if len(tfs) > 1:
                raise Exception("different frames for neighboring targets!", tr.targetframedict)
    sys.stderr.write("multi-token targets in %s: %.3f%% [%d / %d]\n"
                     %(split, multitoktargs*100/tottargs, multitoktargs, tottargs))

def print_data_status(fsp_dict, vocab_str):
    sys.stderr.write("# {} = {}\n\tUnseen in dev/test = {}\n\tUnlearnt in dev/test = {}\n".format(
        vocab_str, fsp_dict.size(), fsp_dict.num_unks()[0], fsp_dict.num_unks()[1]))

# argument parsing
optpr = OptionParser()
optpr.add_option("--mode", dest="mode", type="choice", choices=["train", "test", "refresh", "predict"], default="train")
optpr.add_option("-n", "--model_name", help="Name of model directory to save model to.")
optpr.add_option("--hier", action="store_true", default=False)
optpr.add_option("--exemplar", action="store_true", default=False)
optpr.add_option("--config", type="str", metavar="FILE")
optpr.add_option("--dynet-devices", default="GPU:0")
optpr.add_option("--start_index", type=int, default=0)
optpr.add_option("--end_index", type=int, default=100)
(options, args) = optpr.parse_args()

# configuration
model_dir = "logs/{}/".format(options.model_name)
model_file_name = "{}best-frameid-{}-model".format(model_dir, VERSION)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if options.exemplar:
    train_conll = TRAIN_EXEMPLAR
else:
    train_conll = TRAIN_FTE

USE_DROPOUT = False
USE_WV = True
USE_HIER = options.hier

sys.stderr.write("_____________________\n")
sys.stderr.write("COMMAND: {}\n".format(" ".join(sys.argv)))
if options.mode in ["train", "refresh"]:
    sys.stderr.write("VALIDATED MODEL SAVED TO:\t{}\n".format(model_file_name))
else:
    sys.stderr.write("MODEL FOR TEST / PREDICTION:\t{}\n".format(model_file_name))
sys.stderr.write("PARSING MODE:\t{}\n".format(options.mode))
sys.stderr.write("_____________________\n\n")

trainexamples, m, t = read_conll(train_conll)
find_multitokentargets(trainexamples, "train")

post_train_lock_dicts()
lufrmmap, relatedlus = read_related_lus()
if USE_WV:
    pretrained_embeddings_map = get_wvec_map()
    PRETRAINED_DIM = len(pretrained_embeddings_map.values()[0])

lock_dicts()
UNKTOKEN = VOCDICT.getid(UNK)

# Default configurations.
configuration = {'train': train_conll,
                 'use_exemplar': options.exemplar,
                 'use_hierarchy': USE_HIER,
                 'unk_prob': 0.1,
                 'dropout_rate': 0.01,
                 'token_dim': 100,
                 'pos_dim': 100,
                 'lu_dim': 100,
                 'lu_pos_dim': 100,
                 'lstm_input_dim': 100,
                 'lstm_dim': 100,
                 'lstm_depth': 2,
                 'hidden_dim': 100,
                 'use_dropout': USE_DROPOUT,
                 'pretrained_embedding_dim': PRETRAINED_DIM,
                 'num_epochs': 100 if not options.exemplar else 25,
                 'patience': 25,
                 'eval_after_every_epochs': 100,
                 'dev_eval_epoch_frequency': 5}
configuration_file = os.path.join(model_dir, 'configuration.json')

json_file = open(configuration_file, "r")
configuration = json.load(json_file)

UNK_PROB = configuration['unk_prob']
DROPOUT_RATE = configuration['dropout_rate']

TOKDIM = configuration['token_dim']
POSDIM = configuration['pos_dim']
LUDIM = configuration['lu_dim']
LPDIM = configuration['lu_pos_dim']
INPDIM = TOKDIM + POSDIM

LSTMINPDIM = configuration['lstm_input_dim']
LSTMDIM = configuration['lstm_dim']
LSTMDEPTH = configuration['lstm_depth']
HIDDENDIM = configuration['hidden_dim']

NUM_EPOCHS = configuration['num_epochs']
PATIENCE = configuration['patience']
EVAL_EVERY_EPOCH = configuration['eval_after_every_epochs']
DEV_EVAL_EPOCH = configuration['dev_eval_epoch_frequency'] * EVAL_EVERY_EPOCH

sys.stderr.write("\nPARSER SETTINGS (see {})\n_____________________\n".format(configuration_file))
for key in sorted(configuration):
    sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

sys.stderr.write("\n")

print_data_status(VOCDICT, "Tokens")
print_data_status(POSDICT, "POS tags")
print_data_status(LUDICT, "LUs")
print_data_status(LUPOSDICT, "LU POS tags")
print_data_status(FRAMEDICT, "Frames")
sys.stderr.write("\n_____________________\n\n")

model = Model()
trainer = SimpleSGDTrainer(model)
# trainer = AdamTrainer(model, 0.0001, 0.01, 0.9999, 1e-8)

v_x = model.add_lookup_parameters((VOCDICT.size(), TOKDIM))
p_x = model.add_lookup_parameters((POSDICT.size(), POSDIM))
lu_x = model.add_lookup_parameters((LUDICT.size(), LUDIM))
lp_x = model.add_lookup_parameters((LUPOSDICT.size(), LPDIM))
if USE_WV:
    e_x = model.add_lookup_parameters((VOCDICT.size(), PRETRAINED_DIM))
    for wordid in pretrained_embeddings_map:
        e_x.init_row(wordid, pretrained_embeddings_map[wordid])

    # Embedding for unknown pretrained embedding.
    u_x = model.add_lookup_parameters((1, PRETRAINED_DIM), init='glorot')

    w_e = model.add_parameters((LSTMINPDIM, PRETRAINED_DIM+INPDIM))
    b_e = model.add_parameters((LSTMINPDIM, 1))

w_i = model.add_parameters((LSTMINPDIM, INPDIM))
b_i = model.add_parameters((LSTMINPDIM, 1))

builders = [
    LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
    LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
]

tlstm = LSTMBuilder(LSTMDEPTH, 2*LSTMDIM, LSTMDIM, model)

w_z = model.add_parameters((HIDDENDIM, LSTMDIM + LUDIM + LPDIM))
b_z = model.add_parameters((HIDDENDIM, 1))
w_f = model.add_parameters((FRAMEDICT.size(), HIDDENDIM))
b_f = model.add_parameters((FRAMEDICT.size(), 1))


###############################################################3
# Predict
sys.stderr.write("Loading model from {} ...\n".format(model_file_name))
model.populate(model_file_name)

# load data
def load_instances(filepath):
    print("Loading Instances from {}".format(filepath))
    instances, _, _ = read_conll(filepath)
    return instances

data_dir = "/dgxhome/czh5679/workspace/StoryNet/data"
def load_fiction_list():
    with codecs.open(os.path.join(data_dir, "fiction_list.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)["Fiction"]

    # change path
    fiction_list = [
        {
            "book_id": d[0],
            "path": os.path.join(data_dir, "bookcorpus", "target_id", d[1].split("/")[-1]),
            "output_file": os.path.join(data_dir, "bookcorpus", "frame", d[1].split("/")[-1]),
            "segment_file": os.path.join(data_dir, "bookcorpus", "segment", d[1].split("/")[-1]),
        }
        for d in data
    ]
    return fiction_list

def load_coda_list():
    with codecs.open(os.path.join(data_dir, "coda_list.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # change path
    coda_list = [
        {
            "paper_id": d["paper_id"],
            "path": os.path.join(data_dir, "coda19", "target_id", "{}.txt".format(d["paper_id"])),
            "output_file": os.path.join(data_dir, "coda19", "frame_2", "{}.txt".format(d["paper_id"])),
            "segment_file": os.path.join(data_dir, "coda19", "segment", "{}.txt".format(d["paper_id"]))
        }        
        for d in data        
    ]
    return coda_list

# predict
def predict(instances):
    predictions = []
    length = len(instances)
    for i, instance in enumerate(instances, 1):
        if i % 5 == 0:
            sys.stderr.write("\x1b[2K\r{} / {} [{:.3f}%]".format(i, length, i*100.0/length))
        _, prediction = identify_frames(builders, instance.tokens, instance.postags, instance.lu, instance.targetframedict.keys())
        predictions.append(prediction)
    return predictions

number_index = 6
lu_index = -3
frame_index = -2
def parse_conll(goldexamples, pred_targmaps, output_file, segment_file):
    res = {}
    for g,p in zip(goldexamples, pred_targmaps):
        sent = g.get_predicted_frame_conll(p)
        words = sent.strip().split("\n")
        try:
            index = int(words[0].split("\t")[number_index])
        except IndexError as e:
            traceback.print_exc()
            print(sent)
            print(words)
            quit()
        for word in words:
            info = word.split("\t")
            if info[frame_index] != "_":
                if index not in res:
                    res[index] = []
                res[index].append({"LU":info[lu_index], "Frame":info[frame_index]})

    with codecs.open(segment_file, 'r', encoding='utf-8') as infile:
        lines = infile.read().strip().split("\n")

    data = []
    for i, line in enumerate(lines):
        info = {
            "text": line,
            "frame": res.get(i, []),
        }
        data.append(info)

    with codecs.open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2)

#data_list = load_fiction_list()
data_list = load_coda_list()
data_list = data_list[options.start_index:options.end_index]
previous_time =  datetime.now()
time_info = []
for fiction_count, fiction_info in enumerate(data_list, 1):
    # skip wrong files
    if not os.path.isfile(fiction_info["path"]):
        continue
    
    # skip finished files
    if os.path.isfile(fiction_info["output_file"]):
        continue

    print "\nFiction Count: {} - {} / {} [{:.2f}%] {}".format(
            fiction_count+options.start_index, 
            fiction_count, 
            len(data_list), 
            100.0*fiction_count/len(data_list),
            str(datetime.now())
        )
    start_time = datetime.now()
    if time_info:
        avg_time = np.array(time_info).mean()
        print "time used = {:.4f} sec, average time = {:.4f} sec".format(time_info[-1], avg_time)

    try:
        instances = load_instances(fiction_info["path"])
        predictions = predict(instances)
        sys.stderr.write("\nPrinting output in CoNLL format to {}\n".format(fiction_info["output_file"]))
        #print_as_conll(instances, predictions, fiction_info["output_file"])
        parse_conll(instances, predictions, fiction_info["output_file"], fiction_info["segment_file"])

    except Exception as e:
        with codecs.open("frame_id.error.log", 'a', encoding='utf-8') as outfile:
            outfile.write("\n\n{} - {} - {}\n".format(str(datetime.now()), str(fiction_info), str(e)))

    end_time = datetime.now()
    time_info.append((end_time-start_time).total_seconds())


