# -*- coding: utf-8 -*-
import ujson as json
import os
import sys
import time
from optparse import OptionParser

from dynet import *
#from evaluation import *
from dataio_frame import *
from raw_data import make_data_instance
from semafor_evaluation import convert_conll_to_frame_elements
import dynet_config
from datetime import datetime
import numpy as np
import traceback
from pprint import pprint
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

def get_model_config():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = "/"+os.path.join(*root_folder.split("/")[:-3])
    
    with open(os.path.join(root_folder, "open-sesame.config"), "r") as infile:
        config = json.load(infile)

    model_dir = os.path.join(root_folder, "third_party", "open_sesame", config["frameid"]["model_dir"])
    model_file_name = os.path.join(model_dir, config["frameid"]["model_name"])
    return model_dir, model_file_name

# configuration
model_dir, model_file_name = get_model_config()
#model_dir = "logs/{}/".format("model/fn1.7-pretrained-frameid")
#model_file_name = "{}best-frameid-{}-model".format(model_dir, "1.7")

train_conll = TRAIN_FTE
USE_DROPOUT = False
USE_WV = True
USE_HIER = False

trainexamples, m, t = read_conll(train_conll)
find_multitokentargets(trainexamples, "train")

post_train_lock_dicts()
lufrmmap, relatedlus = read_related_lus()
if USE_WV:
    pretrained_embeddings_map = get_wvec_map()
    PRETRAINED_DIM = len(pretrained_embeddings_map.values()[0])

lock_dicts()
UNKTOKEN = VOCDICT_FRAMEID.getid(UNK)

# load configurations.
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

print_data_status(VOCDICT_FRAMEID, "Tokens")
print_data_status(POSDICT_FRAMEID, "POS tags")
print_data_status(LUDICT_FRAMEID, "LUs")
print_data_status(LUPOSDICT_FRAMEID, "LU POS tags")
print_data_status(FRAMEDICT_FRAMEID, "Frames")
sys.stderr.write("\n_____________________\n\n")

model = Model()
trainer = SimpleSGDTrainer(model)
# trainer = AdamTrainer(model, 0.0001, 0.01, 0.9999, 1e-8)

v_x = model.add_lookup_parameters((VOCDICT_FRAMEID.size(), TOKDIM))
p_x = model.add_lookup_parameters((POSDICT_FRAMEID.size(), POSDIM))
lu_x = model.add_lookup_parameters((LUDICT_FRAMEID.size(), LUDIM))
lp_x = model.add_lookup_parameters((LUPOSDICT_FRAMEID.size(), LPDIM))
if USE_WV:
    e_x = model.add_lookup_parameters((VOCDICT_FRAMEID.size(), PRETRAINED_DIM))
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
w_f = model.add_parameters((FRAMEDICT_FRAMEID.size(), HIDDENDIM))
b_f = model.add_parameters((FRAMEDICT_FRAMEID.size(), 1))

###############################################################3
# Predict
sys.stderr.write("Loading model from {} ...\n".format(model_file_name))
model.populate(model_file_name)

# load data
def load_instances(text):
    instances, _, _ = read_conll(text, stringio=True)
    return instances

# predict
def predict(instances):
    predictions = []
    length = len(instances)
    for i, instance in enumerate(instances, 1):
        #if i % 5 == 0:
        #    sys.stderr.write("\x1b[2K\r{} / {} [{:.3f}%]".format(i, length, i*100.0/length))
        _, prediction = identify_frames(builders, instance.tokens, instance.postags, instance.lu, instance.targetframedict.keys())
        predictions.append(prediction)
    return predictions

number_index = 6
lu_index = -3
frame_index = -2
def parse_conll(goldexamples, pred_targmaps):
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

    frames = [
        frame
        for key, val in res.items()
        for frame in val
    ]
    return frames

text = """
1	note	_	Note	_	NN	2	_	_	_	_	_	note.v	_	O
2	that	_	that	_	IN	2	_	_	_	_	_	_	_	O
3	model	_	model	_	NN	2	_	_	_	_	_	_	_	O
4	UNK	_	UNK	_	NN	2	_	_	_	_	_	_	_	O
5	is	_	be	_	VBZ	2	_	_	_	_	_	_	_	O
6	still	_	still	_	RB	2	_	_	_	_	_	_	_	O
7	an	_	an	_	DT	2	_	_	_	_	_	_	_	O
8	open	_	open	_	JJ	2	_	_	_	_	_	_	_	O
9	research	_	research	_	NN	2	_	_	_	_	_	_	_	O
10	question	_	question	_	NN	2	_	_	_	_	_	_	_	O
11	.	_	.	_	.	2	_	_	_	_	_	_	_	O

1	note	_	Note	_	NN	2	_	_	_	_	_	_	_	O
2	that	_	that	_	IN	2	_	_	_	_	_	that.adv	_	O
3	model	_	model	_	NN	2	_	_	_	_	_	_	_	O
4	UNK	_	UNK	_	NN	2	_	_	_	_	_	_	_	O
5	is	_	be	_	VBZ	2	_	_	_	_	_	_	_	O
6	still	_	still	_	RB	2	_	_	_	_	_	_	_	O
7	an	_	an	_	DT	2	_	_	_	_	_	_	_	O
8	open	_	open	_	JJ	2	_	_	_	_	_	_	_	O
9	research	_	research	_	NN	2	_	_	_	_	_	_	_	O
10	question	_	question	_	NN	2	_	_	_	_	_	_	_	O
11	.	_	.	_	.	2	_	_	_	_	_	_	_	O

1	note	_	Note	_	NN	2	_	_	_	_	_	_	_	O
2	that	_	that	_	IN	2	_	_	_	_	_	_	_	O
3	model	_	model	_	NN	2	_	_	_	_	_	_	_	O
4	UNK	_	UNK	_	NN	2	_	_	_	_	_	_	_	O
5	is	_	be	_	VBZ	2	_	_	_	_	_	_	_	O
6	still	_	still	_	RB	2	_	_	_	_	_	_	_	O
7	an	_	an	_	DT	2	_	_	_	_	_	_	_	O
8	open	_	open	_	JJ	2	_	_	_	_	_	_	_	O
9	research	_	research	_	NN	2	_	_	_	_	_	research.n	_	O
10	question	_	question	_	NN	2	_	_	_	_	_	_	_	O
11	.	_	.	_	.	2	_	_	_	_	_	_	_	O

1	each	_	Each	_	DT	3	_	_	_	_	_	_	_	O
2	introduced	_	introduce	_	VBD	3	_	_	_	_	_	_	_	O
3	method	_	method	_	NN	3	_	_	_	_	_	_	_	O
4	has	_	have	_	VBZ	3	_	_	_	_	_	_	_	O
5	certain	_	certain	_	JJ	3	_	_	_	_	_	_	_	O
6	pros	_	pro	_	NNS	3	_	_	_	_	_	pro.n	_	O
7	&	_	&	_	CC	3	_	_	_	_	_	_	_	O
8	cons	_	UNK	_	NNS	3	_	_	_	_	_	_	_	O
9	.	_	.	_	.	3	_	_	_	_	_	_	_	O
"""

instances = load_instances(text.strip()+"\n")
predictions = predict(instances)
res = parse_conll(instances, predictions)
pprint(res)

