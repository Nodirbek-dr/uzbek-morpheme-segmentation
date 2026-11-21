import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, random
from common import graphemes
import morfessor
SEED = 20261119

def train_morf(ws, cw):
    random.seed(SEED)  # Morfessor shuffles the training data in every epoch
    model = morfessor.BaselineModel(corpusweight=cw)
    model.load_data([(1, graphemes(w)[0]) for w in ws])
    model.train_batch()
    return model

def morf_pred(model, w):
    g, st = graphemes(w); segs = model.viterbi_segment(g)[0]; pos = 0; bs = set()
    for s in segs[:-1]:
        pos += len(s); bs.add(st[pos])
    return bs

