# -*- coding: utf-8 -*-
import json
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
dynet_config.set_gpu("GPU:0")

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


def print_as_conll(gold_examples, predicted_target_dict, output_filepath):
    """
    Creates a CoNLL object with predicted target and lexical unit.
    Spits out one CoNLL for each LU.
    """
    with codecs.open(output_filepath, "w", encoding="utf-8") as conll_file:
        output_text = ""
        for gold, pred in zip(gold_examples, predicted_target_dict):
            for target in sorted(pred):
                result = gold.get_predicted_target_conll(target, pred[target][0]) + "\n"
                output_text += result
        conll_file.write(output_text)
        conll_file.close()

def print_data_status(fsp_dict, vocab_str):
    sys.stderr.write("# {} = {}\n\tUnseen in dev/test = {}\n\tUnlearnt in dev/test = {}\n".format(
        vocab_str, fsp_dict.size(), fsp_dict.num_unks()[0], fsp_dict.num_unks()[1]))

##########################################
# main

# build option
optpr = OptionParser()
optpr.add_option("--mode", dest="mode", type="choice",
                 choices=["train", "test", "refresh", "predict"], default="train")
optpr.add_option("-n", "--model_name", help="Name of model directory to save model to.")
optpr.add_option("--config", type="str", metavar="FILE")
optpr.add_option("--dynet-devices", default="GPU:0")
optpr.add_option("--start_index", type=int, default=0)
optpr.add_option("--end_index", type=int, default=100)
(options, args) = optpr.parse_args()

model_dir = "logs/{}/".format(options.model_name)
model_file_name = "{}best-targetid-{}-model".format(model_dir, VERSION)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

train_conll = TRAIN_FTE
USE_DROPOUT = False

sys.stderr.write("_____________________\n")
sys.stderr.write("COMMAND: {}\n".format(" ".join(sys.argv)))
if options.mode in ["train", "refresh"]:
    sys.stderr.write("VALIDATED MODEL SAVED TO:\t{}\n".format(model_file_name))
else:
    sys.stderr.write("MODEL FOR TEST / PREDICTION:\t{}\n".format(model_file_name))
sys.stderr.write("PARSING MODE:\t{}\n".format(options.mode))
sys.stderr.write("_____________________\n\n")

##
print_data_status(VOCDICT, "Tokens1")
train_examples, _, _ = read_conll(train_conll)
print_data_status(VOCDICT, "Tokens2")
combined_train = combine_examples(train_examples)

# Need to read all LUs before locking the dictionaries.
target_lu_map, lu_names = create_target_lu_map()
print_data_status(VOCDICT, "Tokens5")
post_train_lock_dicts()
print_data_status(VOCDICT, "Tokens6")

# Read pretrained word embeddings.
pretrained_map = get_wvec_map()
print_data_status(VOCDICT, "Tokens3")
PRETRAINED_DIM = len(pretrained_map.values()[0])

lock_dicts()
print_data_status(VOCDICT, "Tokens4")
UNKTOKEN = VOCDICT.getid(UNK)

# Default configurations.
configuration = {"train": train_conll,
                 "unk_prob": 0.1,
                 "dropout_rate": 0.01,
                 "token_dim": 100,
                 "pos_dim": 100,
                 "lemma_dim": 100,
                 "lstm_input_dim": 100,
                 "lstm_dim": 100,
                 "lstm_depth": 2,
                 "hidden_dim": 100,
                 "use_dropout": USE_DROPOUT,
                 "pretrained_embedding_dim": PRETRAINED_DIM,
                 "num_epochs": 100,
                 "patience": 25,
                 "eval_after_every_epochs": 100,
                 "dev_eval_epoch_frequency": 3}
configuration_file = os.path.join(model_dir, "configuration.json")
if options.mode == "train":
    if options.config:
        config_json = open(options.config, "r")
        configuration = json.load(config_json)
    with open(configuration_file, "w") as fout:
        fout.write(json.dumps(configuration))
        fout.close()
else:
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
print_data_status(POSDICT, "POS tags")
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
u_x = model.add_lookup_parameters((1, PRETRAINED_DIM), init='glorot')

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

def load_instances(filepath):
    print("Loading Instances from {}".format(filepath))
    with codecs.open(filepath, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        instances = [make_data_instance(line, i) for i, line in enumerate(lines)]
    return instances

def predict(instances):
    predictions = []
    length = len(instances)
    for i, instance in enumerate(instances, 1):
        if i % 5 == 0:
            sys.stderr.write("\x1b[2K\r{} / {} [{:.3f}%]".format(i, length, i*100.0/length))
        _, prediction = identify_targets(builders, instance.tokens, instance.postags, instance.lemmas)
        predictions.append(prediction)
    return predictions

data_dir = "/dgxhome/czh5679/workspace/StoryNet/data"
def load_fiction_list():
    with codecs.open(os.path.join(data_dir, "fiction_list.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)["Fiction"]

    # change path
    fiction_list = [
        {
            "book_id": d[0],
            "path": os.path.join(data_dir, "bookcorpus", "segment", d[1].split("/")[-1]),
            "output_file": os.path.join(data_dir, "bookcorpus", "target_id", d[1].split("/")[-1]),
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
            "path": os.path.join(data_dir, "coda19", "segment", "{}.txt".format(d["paper_id"])),
            "output_file": os.path.join(data_dir, "coda19", "target_id", "{}.txt".format(d["paper_id"]))
        }        
        for d in data        
    ]
    return coda_list


#data_list = load_fiction_list()
data_list = load_coda_list()
data_list = data_list[options.start_index:options.end_index]
previous_time =  datetime.now()
time_info = []
for fiction_count, fiction_info in enumerate(data_list, 1):
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
        print_as_conll(instances, predictions, fiction_info["output_file"])
    except Exception as e:
        with codecs.open("target_id.error.log", 'a', encoding='utf-8') as outfile:
            outfile.write("\n\n{} - {} - {}\n".format(str(datetime.now()), str(fiction_info), str(e)))

    end_time = datetime.now()
    time_info.append((end_time-start_time).total_seconds())

