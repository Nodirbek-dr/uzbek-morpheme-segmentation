"""CRF with root-lexicon features built by jackknifing: during training, a word's features use only roots
from other root groups, so the lexicon never contains the word's own root (as at test time in the root-disjoint split)."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, time, random
from common import graphemes, components
from evalu import evaluate, SPLITS
from crf import labels
from crf_lex import feats_lex
import sklearn_crfsuite
from collections import defaultdict

split = sys.argv[1]; sp = SPLITS[split]; t0 = time.time()
recs, P, S = components()
roots_of = defaultdict(set)
for r in recs: roots_of[r['surface']].add(graphemes(r['root'])[0])
train = sp['train']
# folds by root (a word goes to the fold of its first root)
allroots = sorted({x for w in train for x in roots_of[w]}); random.Random(5).shuffle(allroots)
fold_of_root = {x: i % 5 for i, x in enumerate(allroots)}
fold = {w: fold_of_root[min(roots_of[w])] for w in train}
L_full = {x for w in train for x in roots_of[w] if len(x) >= 3}
L_fold = {k: {x for w in train if fold[w] != k for x in roots_of[w] if len(x) >= 3} for k in range(5)}
X = [feats_lex(graphemes(w)[0], L_fold[fold[w]]) for w in train]; Y = [labels(w) for w in train]
m = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.0, c2=0.3, max_iterations=150, all_possible_transitions=True); m.fit(X, Y)
def predict(ws):
    out = {}
    for w, y in zip(ws, m.predict([feats_lex(graphemes(w)[0], L_full) for w in ws])):
        st = graphemes(w)[1]; out[w] = sorted(st[i] for i, t in enumerate(y) if t == '1' and i > 0)
    return out
dev = evaluate({w: set(v) for w, v in predict(sp['dev']).items()}, sp['dev'])
pr = predict(sp['test']); test = evaluate({w: set(v) for w, v in pr.items()}, sp['test'], boot=1000)
json.dump(pr, open(RESULTS + f'/pred_crflexjk_{split}.json', 'w', encoding='utf8'), ensure_ascii=False)
res = {'split': split, 'dev': dev, 'test': test, 'train_s': time.time() - t0}
json.dump(res, open(RESULTS + f'/res_crflexjk_{split}.json', 'w'), indent=1); print(json.dumps(res), flush=True)
