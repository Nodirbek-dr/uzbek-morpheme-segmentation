"""Morfessor trained transductively on all unsegmented word types (no gold labels used)."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json
from baselines_lib import train_morf, morf_pred
from evalu import evaluate, SPLITS, GOLD
allw = sorted(GOLD); m = train_morf(allw, 1.0); out = {}
for split in ('random', 'root'):
    sp = SPLITS[split]
    out[split] = {'dev': evaluate({w: morf_pred(m, w) for w in sp['dev']}, sp['dev']), 'test': evaluate({w: morf_pred(m, w) for w in sp['test']}, sp['test'], boot=1000)}
    print(split, {k: round(v, 4) for k, v in out[split]['test'].items() if k in ('P', 'R', 'F1', 'acc')}, flush=True)
json.dump(out, open(RESULTS + '/res_morf_all.json', 'w'), indent=1)
