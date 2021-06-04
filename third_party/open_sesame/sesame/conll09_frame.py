# -*- coding: utf-8 -*-
from copy import deepcopy

from frame_semantic_graph import LexicalUnit, Frame, FrameElement, FrameSemParse
from housekeeping import *

VOCDICT_FRAMEID = FspDict()
LEMDICT_FRAMEID = FspDict()
POSDICT_FRAMEID = FspDict()
FRAMEDICT_FRAMEID = FspDict()
LUDICT_FRAMEID = FspDict()
LUPOSDICT_FRAMEID = FspDict()
FEDICT_FRAMEID = FspDict()
DEPRELDICT_FRAMEID = FspDict()
CLABELDICT_FRAMEID = FspDict()


class CoNLL09Element:
    """
    All the elements in a single line of a CoNLL 2009-like file.
    """

    def __init__(self, conll_line, read_depsyn=None):
        ele = conll_line.split("\t")
        lufields = ['_', '_']
        self.id = int(ele[0])
        self.form = VOCDICT_FRAMEID.addstr(ele[1].lower())
        self.nltk_lemma = LEMDICT_FRAMEID.addstr(ele[3])
        self.fn_pos = ele[4]  # Not a gold POS tag, provided by taggers used in FrameNet, ignore.
        self.nltk_pos = POSDICT_FRAMEID.addstr(ele[5])
        self.sent_num = int(ele[6])

        self.dephead = EMPTY_LABEL
        self.deprel = EMPTY_LABEL
        if read_depsyn:
            self.dephead = int(ele[9])
            self.deprel = DEPRELDICT_FRAMEID.addstr(ele[11])

        self.is_pred = (ele[12] != EMPTY_LABEL)
        if self.is_pred:
            lufields = ele[12].split(".")
        self.lu = LUDICT_FRAMEID.addstr(lufields[0])
        self.lupos = LUPOSDICT_FRAMEID.addstr(lufields[1])
        self.frame = FRAMEDICT_FRAMEID.addstr(ele[13])

        # BIOS scheme
        self.is_arg = (ele[14] != EMPTY_FE)
        self.argtype = BIO_INDEX_DICT[ele[14][0]]
        if self.is_arg:
            self.role = FEDICT_FRAMEID.addstr(ele[14][2:])
        else:
            self.role = FEDICT_FRAMEID.addstr(ele[14])

    def get_str(self, rolelabel=None, no_args=False):
        idstr = str(self.id)
        form = VOCDICT_FRAMEID.getstr(self.form)
        predicted_lemma = LEMDICT_FRAMEID.getstr(self.nltk_lemma)
        nltkpos = POSDICT_FRAMEID.getstr(self.nltk_pos)

        dephead = "_"
        deprel = "_"
        if self.dephead != EMPTY_LABEL:
            dephead = str(self.dephead)
            deprel = DEPRELDICT_FRAMEID.getstr(self.deprel)

        if self.is_pred:
            lu = LUDICT_FRAMEID.getstr(self.lu) + "." + LUPOSDICT_FRAMEID.getstr(self.lupos)
        else:
            lu = LUDICT_FRAMEID.getstr(self.lu)
        frame = FRAMEDICT_FRAMEID.getstr(self.frame)

        if rolelabel is None:
            if self.is_arg:
                rolelabel = INDEX_BIO_DICT[self.argtype] + "-" + FEDICT_FRAMEID.getstr(self.role)
            else:
                rolelabel = INDEX_BIO_DICT[self.argtype]

        if no_args:  # For Target ID / Frame ID predictions
            rolelabel = "O"

        if DEBUG_MODE:
            return idstr + form + lu + frame + rolelabel
        else:
            # ID    FORM    LEMMA   PLEMMA  POS PPOS    SENT#   PFEAT   HEAD    PHEAD   DEPREL  PDEPREL LU  FRAME ROLE
            # 0     1       2       3       4   5       6       7       8       9       10      11      12  13    14
            return "{}\t{}\t_\t{}\t{}\t{}\t{}\t_\t_\t{}\t_\t{}\t{}\t{}\t{}\n".format(
                self.id, form, predicted_lemma, self.fn_pos, nltkpos, self.sent_num, dephead, deprel, lu, frame, rolelabel).encode('utf-8')


class CoNLL09Example(FrameSemParse):
    """a single example in CoNLL 09 format which corresponds to a single frame-semantic parse structure"""

    def __init__(self, sentence, elements):
        FrameSemParse.__init__(self, sentence)
        # not in parent class
        self._elements = elements
        self.sent_num = elements[0].sent_num

        notfes = []
        self.invertedfes = {}
        for e in elements:
            if e.is_pred:
                self.add_target((e.id - 1), e.lu, e.lupos, e.frame)

            if e.role not in self.invertedfes:
                self.invertedfes[e.role] = []
            if e.argtype == SINGULAR:
                self.invertedfes[e.role].append((e.id - 1, e.id - 1))
                self.numargs += 1
            elif e.argtype == BEGINNING:
                self.invertedfes[e.role].append((e.id - 1, None))
                self.numargs += 1
            elif e.argtype == INSIDE:
                argspan = self.invertedfes[e.role].pop()
                self.invertedfes[e.role].append((argspan[0], e.id - 1))
            else:
                notfes.append(e.id - 1)

        if FEDICT_FRAMEID.getid(EMPTY_FE) in self.invertedfes:
            self.invertedfes[FEDICT_FRAMEID.getid(EMPTY_FE)] = extract_spans(notfes)

        self.modifiable = False  # true cz generally gold.

    def _get_inverted_femap(self):
        tmp = {}
        for e in self._elements:
            if e.role not in tmp:
                tmp[e.role] = []
            tmp[e.role].append(e.id - 1)

        inverted = {}
        for felabel in tmp:
            argindices = sorted(tmp[felabel])
            argranges = extract_spans(argindices)
            inverted[felabel] = argranges

        return inverted

    def get_str(self, predictedfes=None):
        mystr = ""
        if predictedfes is None:
            for e in self._elements:
                mystr += e.get_str()
        else:
            rolelabels = [EMPTY_FE for _ in self._elements]
            for feid in predictedfes:
                felabel = FEDICT_FRAMEID.getstr(feid)
                if felabel == EMPTY_FE:
                    continue
                for argspan in predictedfes[feid]:
                    if argspan[0] == argspan[1]:
                        rolelabels[argspan[0]] = INDEX_BIO_DICT[SINGULAR] + "-" + felabel
                    else:
                        rolelabels[argspan[0]] = INDEX_BIO_DICT[BEGINNING] + "-" + felabel
                    for position in xrange(argspan[0] + 1, argspan[1] + 1):
                        rolelabels[position] = INDEX_BIO_DICT[INSIDE] + "-" + felabel

            for e, role in zip(self._elements, rolelabels):
                mystr += e.get_str(rolelabel=role)

        return mystr

    def get_predicted_frame_conll(self, predicted_frame):
        """
        Get new CoNLL string, after substituting predicted frame.
        """
        new_conll_str = ""
        for e in xrange(len(self._elements)):
            field = deepcopy(self._elements[e])
            if (field.id - 1) in predicted_frame:
                field.is_pred = True
                field.lu = predicted_frame[field.id - 1][0].id
                field.lupos = predicted_frame[field.id - 1][0].posid
                field.frame = predicted_frame[field.id - 1][1].id
            else:
                field.is_pred = False
                field.lu = LUDICT_FRAMEID.getid(EMPTY_LABEL)
                field.lupos = LUPOSDICT_FRAMEID.getid(EMPTY_LABEL)
                field.frame = FRAMEDICT_FRAMEID.getid(EMPTY_LABEL)
            new_conll_str += field.get_str()
        return new_conll_str

    def get_predicted_target_conll(self, predicted_target, predicted_lu):
        """
        Get new CoNLL string, after substituting predicted target.
        """
        new_conll_str = ""
        for e in xrange(len(self._elements)):
            field = deepcopy(self._elements[e])
            if (field.id - 1) == predicted_target:
                field.is_pred = True
                field.lu = predicted_lu.id
                field.lupos = predicted_lu.posid
            else:
                field.is_pred = False
                field.lu = LUDICT_FRAMEID.getid(EMPTY_LABEL)
                field.lupos = LUPOSDICT_FRAMEID.getid(EMPTY_LABEL)
            field.frame = FRAMEDICT_FRAMEID.getid(EMPTY_LABEL)
            new_conll_str += field.get_str(no_args=True)
        return new_conll_str

    def print_internal(self, logger):
        self.print_internal_sent(logger)
        self.print_internal_frame(logger)
        self.print_internal_args(logger)

    def print_internal_sent(self, logger):
        logger.write("tokens and depparse:\n")
        for x in xrange(len(self.tokens)):
            logger.write(VOCDICT_FRAMEID.getstr(self.tokens[x]) + " ")
        logger.write("\n")

    def print_internal_frame(self, logger):
        logger.write("LU and frame: ")
        for tfpos in self.targetframedict:
            t, f = self.targetframedict[tfpos]
            logger.write(VOCDICT_FRAMEID.getstr(self.tokens[tfpos]) + ":" + \
                LUDICT_FRAMEID.getstr(t.id) + "." + LUPOSDICT_FRAMEID.getstr(t.posid) + \
                FRAMEDICT_FRAMEID.getstr(f.id) + "\n")

    def print_external_frame(self, predtf, logger):
        logger.write("LU and frame: ")
        for tfpos in predtf:
            t, f = predtf[tfpos]
            logger.write(VOCDICT_FRAMEID.getstr(self.tokens[tfpos]) + ":" + \
                LUDICT_FRAMEID.getstr(t.id) + "." + LUPOSDICT_FRAMEID.getstr(t.posid) + \
                FRAMEDICT_FRAMEID.getstr(f.id) + "\n")

    def print_internal_args(self, logger):
        logger.write("frame:" + FRAMEDICT_FRAMEID.getstr(self.frame.id).upper() + "\n")
        for fepos in self.invertedfes:
            if fepos == FEDICT_FRAMEID.getid(EMPTY_FE):
                continue
            for span in self.invertedfes[fepos]:
                logger.write(FEDICT_FRAMEID.getstr(fepos) + "\t")
                for s in xrange(span[0], span[1] + 1):
                    logger.write(VOCDICT_FRAMEID.getstr(self.tokens[s]) + " ")
                logger.write("\n")
        logger.write("\n")

    def print_external_parse(self, parse, logger):
        for fepos in parse:
            if fepos == FEDICT_FRAMEID.getid(EMPTY_FE):
                continue
            for span in parse[fepos]:
                logger.write(FEDICT_FRAMEID.getstr(fepos) + "\t")
                for s in xrange(span[0], span[1] + 1):
                    logger.write(VOCDICT_FRAMEID.getstr(self.tokens[s]) + " ")
                logger.write("\n")
        logger.write("\n")


def lock_dicts():
    VOCDICT_FRAMEID.lock()
    LEMDICT_FRAMEID.lock()
    POSDICT_FRAMEID.lock()
    FRAMEDICT_FRAMEID.lock()
    LUDICT_FRAMEID.lock()
    LUPOSDICT_FRAMEID.lock()
    FEDICT_FRAMEID.lock()
    DEPRELDICT_FRAMEID.lock()
    CLABELDICT_FRAMEID.lock()

def post_train_lock_dicts():
    VOCDICT_FRAMEID.post_train_lock()
    LEMDICT_FRAMEID.post_train_lock()
    POSDICT_FRAMEID.post_train_lock()
    FRAMEDICT_FRAMEID.post_train_lock()
    LUDICT_FRAMEID.post_train_lock()
    LUPOSDICT_FRAMEID.post_train_lock()
    FEDICT_FRAMEID.post_train_lock()
    DEPRELDICT_FRAMEID.post_train_lock()
    CLABELDICT_FRAMEID.post_train_lock()
