# -*- coding: utf-8 -*-
import json
import os
import sys
import time
from optparse import OptionParser

from dynet import *
#from evaluation import *
from dataio import *
from raw_data import make_data_instance
from semafor_evaluation import convert_conll_to_frame_elements
import dynet_config
from datetime import datetime
import numpy as np
import traceback
from pprint import pprint
import nltk
#dynet_config.set_gpu("GPU:0")

##########################################
# functions
def combine_examples(corpus_ex):
    """
    Target ID needs to be trained for all targets in the sentence jointly, as opposed to
    frame and arg ID. Returns all target annotations for a given sentence.
    """
    combined_ex = [corpus_ex[0]]
    for ex in corpus_ex[1:]:
        if ex.sent_num == combined_ex[-1].sent_num:
            current_sent = combined_ex.pop()
            target_frame_dict = current_sent.targetframedict.copy()
            target_frame_dict.update(ex.targetframedict)
            current_sent.targetframedict = target_frame_dict
            combined_ex.append(current_sent)
            continue
        combined_ex.append(ex)
    sys.stderr.write("Combined {} instances in data into {} instances.\n".format(
        len(corpus_ex), len(combined_ex)))
    return combined_ex

def get_fn_pos_by_rules(pos, token):
    """
    Rules for mapping NLTK part of speech tags into FrameNet tags, based on co-occurrence
    statistics, since there is not a one-to-one mapping.
    """
    if pos[0] == "v" or pos in ["rp", "ex", "md"]:  # Verbs
        rule_pos = "v"
    elif pos[0] == "n" or pos in ["$", ":", "sym", "uh", "wp"]:  # Nouns
        rule_pos = "n"
    elif pos[0] == "j" or pos in ["ls", "pdt", "rbr", "rbs", "prp"]:  # Adjectives
        rule_pos = "a"
    elif pos == "cc":  # Conjunctions
        rule_pos = "c"
    elif pos in ["to", "in"]:  # Prepositions
        rule_pos = "prep"
    elif pos in ["dt", "wdt"]:  # Determinors
        rule_pos = "art"
    elif pos in ["rb", "wrb"]:  # Adverbs
        rule_pos = "adv"
    elif pos == "cd":  # Cardinal Numbers
        rule_pos = "num"
    else:
        sys.stderr.write("WARNING: Rule not defined for part-of-speech {} word {} - treating as noun.".format(pos, token))
        return "n"
    return rule_pos


def check_if_potential_target(lemma):
    """
    Simple check to see if this is a potential position to even consider, based on
    the LU index provided under FrameNet. Note that since we use NLTK lemmas,
    this might be lossy.
    """
    nltk_lem_str = LEMDICT.getstr(lemma)
    return nltk_lem_str in target_lu_map or nltk_lem_str.lower() in target_lu_map


def create_lexical_unit(lemma_id, pos_id, token_id):
    """
    Given a lemma ID and a POS ID (both lemma and POS derived from NLTK),
    create a LexicalUnit object.
    If lemma is unknown, then check if token is in the LU vocabulary, and
    use it if present (Hack).
    """
    nltk_lem_str = LEMDICT.getstr(lemma_id)
    if nltk_lem_str not in target_lu_map and nltk_lem_str.lower() in target_lu_map:
        nltk_lem_str = nltk_lem_str.lower()

    # Lemma is not in FrameNet, but it could be a lemmatization error.
    if nltk_lem_str == UNK:
        if VOCDICT.getstr(token_id) in target_lu_map:
            nltk_lem_str = VOCDICT.getstr(token_id)
        elif VOCDICT.getstr(token_id).lower() in target_lu_map:
            nltk_lem_str = VOCDICT.getstr(token_id).lower()
    assert nltk_lem_str in target_lu_map
    assert LUDICT.getid(nltk_lem_str) != LUDICT.getid(UNK)

    nltk_pos_str = POSDICT.getstr(pos_id)
    rule_pos_str = get_fn_pos_by_rules(nltk_pos_str.lower(), nltk_lem_str)
    rule_lupos = nltk_lem_str + "." + rule_pos_str

    # Lemma is not seen with this pos tag.
    if rule_lupos not in lu_names:
        # Hack: replace with anything the lemma is seen with.
        rule_pos_str = list(target_lu_map[nltk_lem_str])[0].split(".")[-1]
    return LexicalUnit(LUDICT.getid(nltk_lem_str), LUPOSDICT.getid(rule_pos_str))

def identify_targets(builders, tokens, postags, lemmas, gold_targets=None):
    """
    Target identification model, using bidirectional LSTMs, with a
    multilinear perceptron layer on top for classification.
    """
    renew_cg()
    train_mode = (gold_targets is not None)

    sentlen = len(tokens)
    emb_x = [v_x[tok] for tok in tokens]
    pos_x = [p_x[pos] for pos in postags]
    lem_x = [l_x[lem] for lem in lemmas]

    emb2_xi = []
    for i in xrange(sentlen):
        if tokens[i] in pretrained_map:
            # Prevent the pretrained embeddings from being updated.
            emb_without_backprop = lookup(e_x, tokens[i], update=False)
            features_at_i = concatenate([emb_x[i], pos_x[i], lem_x[i], emb_without_backprop])
        else:
            features_at_i = concatenate([emb_x[i], pos_x[i], lem_x[i], u_x])
        emb2_xi.append(w_e * features_at_i + b_e)

    emb2_x = [rectify(emb2_xi[i]) for i in xrange(sentlen)]

    # Initializing the two LSTMs.
    if USE_DROPOUT and train_mode:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)
    f_init, b_init = [i.initial_state() for i in builders]

    fw_x = f_init.transduce(emb2_x)
    bw_x = b_init.transduce(reversed(emb2_x))

    losses = []
    predicted_targets = {}
    for i in xrange(sentlen):
        if not check_if_potential_target(lemmas[i]):
            continue
        h_i = concatenate([fw_x[i], bw_x[sentlen - i - 1]])
        score_i = w_f * rectify(w_z * h_i + b_z) + b_f
        if train_mode and USE_DROPOUT:
            score_i = dropout(score_i, DROPOUT_RATE)
        
        score_i = to_device(score_i, "CPU")
        logloss = log_softmax(score_i, [0, 1])
        if not train_mode:
            is_target = np.argmax(logloss.npvalue())
        else:
            is_target = int(i in gold_targets)

        if int(np.argmax(logloss.npvalue())) != 0:
            predicted_targets[i] = (create_lexical_unit(lemmas[i], postags[i], tokens[i]), None)

        losses.append(pick(logloss, is_target))

    objective = -esum(losses) if losses else None
    return objective, predicted_targets


def print_as_conll(gold_examples, predicted_target_dict):
    """
    Creates a CoNLL object with predicted target and lexical unit.
    Spits out one CoNLL for each LU.
    """
    output_text = ""
    for gold, pred in zip(gold_examples, predicted_target_dict):
        for target in sorted(pred):
            result = gold.get_predicted_target_conll(target, pred[target][0]) + "\n"
            output_text += result
    return output_text

def print_data_status(fsp_dict, vocab_str):
    sys.stderr.write("# {} = {}\n\tUnseen in dev/test = {}\n\tUnlearnt in dev/test = {}\n".format(
        vocab_str, fsp_dict.size(), fsp_dict.num_unks()[0], fsp_dict.num_unks()[1]))

    with codecs.open("{}.json".format(vocab_str), 'w', encoding="utf-8") as outfile:
        json.dump(fsp_dict._strtoint, outfile, indent=2)

def get_model_config():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = "/"+os.path.join(*root_folder.split("/")[:-3])
    
    with open(os.path.join(root_folder, "open-sesame.config"), "r") as infile:
        config = json.load(infile)

    model_dir = os.path.join(root_folder, "third_party", "open_sesame", config["targetid"]["model_dir"])
    model_file_name = os.path.join(model_dir, config["targetid"]["model_name"])
    return model_dir, model_file_name

##########################################
# main
model_dir, model_file_name = get_model_config()
#model_dir = "logs/{}/".format("model/fn1.7-pretrained-targetid")
#model_file_name = "{}best-targetid-{}-model".format(model_dir, "1.7")

train_conll = TRAIN_FTE
USE_DROPOUT = False

##
train_examples, _, _ = read_conll(train_conll)
combined_train = combine_examples(train_examples)

# Need to read all LUs before locking the dictionaries.
target_lu_map, lu_names = create_target_lu_map()
post_train_lock_dicts()

# Read pretrained word embeddings.
pretrained_map = get_wvec_map()
PRETRAINED_DIM = len(pretrained_map.values()[0])

lock_dicts()
UNKTOKEN = VOCDICT.getid(UNK)

# load configuration file
configuration_file = os.path.join(model_dir, "configuration.json")
json_file = open(configuration_file, "r")
configuration = json.load(json_file)

UNK_PROB = configuration["unk_prob"]
DROPOUT_RATE = configuration["dropout_rate"]

TOK_DIM = configuration["token_dim"]
POS_DIM = configuration["pos_dim"]
LEMMA_DIM = configuration["lemma_dim"]
INPUT_DIM = TOK_DIM + POS_DIM + LEMMA_DIM

LSTM_INP_DIM = configuration["lstm_input_dim"]
LSTM_DIM = configuration["lstm_dim"]
LSTM_DEPTH = configuration["lstm_depth"]
HIDDEN_DIM = configuration["hidden_dim"]

NUM_EPOCHS = configuration["num_epochs"]
PATIENCE = configuration["patience"]
EVAL_EVERY_EPOCH = configuration["eval_after_every_epochs"]
DEV_EVAL_EPOCH = configuration["dev_eval_epoch_frequency"] * EVAL_EVERY_EPOCH

sys.stderr.write("\nPARSER SETTINGS (see {})\n_____________________\n".format(configuration_file))
for key in sorted(configuration):
    sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

sys.stderr.write("\n")

print_data_status(VOCDICT, "Tokens")
print_data_status(POSDICT, "POS_tags")
print_data_status(LEMDICT, "Lemmas")
sys.stderr.write("\n_____________________\n\n")

###### build model
model = Model()

v_x = model.add_lookup_parameters((VOCDICT.size(), TOK_DIM))
p_x = model.add_lookup_parameters((POSDICT.size(), POS_DIM))
l_x = model.add_lookup_parameters((LEMDICT.size(), LEMMA_DIM))

e_x = model.add_lookup_parameters((VOCDICT.size(), PRETRAINED_DIM))
for wordid in pretrained_map:
    e_x.init_row(wordid, pretrained_map[wordid])

# Embedding for unknown pretrained embedding.
#u_x = model.add_lookup_parameters((1, PRETRAINED_DIM), init='glorot')
u_x = model.add_lookup_parameters((1, PRETRAINED_DIM))

w_e = model.add_parameters((LSTM_INP_DIM, PRETRAINED_DIM + INPUT_DIM))
b_e = model.add_parameters((LSTM_INP_DIM, 1))

w_i = model.add_parameters((LSTM_INP_DIM, INPUT_DIM))
b_i = model.add_parameters((LSTM_INP_DIM, 1))

builders = [
    LSTMBuilder(LSTM_DEPTH, LSTM_INP_DIM, LSTM_DIM, model),
    LSTMBuilder(LSTM_DEPTH, LSTM_INP_DIM, LSTM_DIM, model),
]

w_z = model.add_parameters((HIDDEN_DIM, 2*LSTM_DIM))
b_z = model.add_parameters((HIDDEN_DIM, 1))
w_f = model.add_parameters((2, HIDDEN_DIM))  # prediction: is a target or not.
b_f = model.add_parameters((2, 1))

###### Predict
sys.stderr.write("Reading model from {} ...\n".format(model_file_name))
model.populate(model_file_name)

def predict(instances):
    predictions = []
    length = len(instances)
    for i, instance in enumerate(instances, 1):
        if i % 5 == 0:
            sys.stderr.write("\x1b[2K\r{} / {} [{:.3f}%]".format(i, length, i*100.0/length))
        _, prediction = identify_targets(builders, instance.tokens, instance.postags, instance.lemmas)
        predictions.append(prediction)
    return predictions

def load_instances(story):
    lines = nltk.sent_tokenize(story)
    instances = [make_data_instance(line, i) for i, line in enumerate(lines)]
    return instances

text = """
How to steer a powerful unconditioned language model? In this post, we will delve into several approaches for controlled content generation with an unconditioned langage model. Note that model steerability is still an open research question. Each introduced method has certain pros & cons.
"""

instances = load_instances(text)
predictions = predict(instances)
conll = print_as_conll(instances, predictions)

