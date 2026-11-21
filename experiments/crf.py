"""Supervised CRF segmenter (character-level boundary tagging over graphemes)."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, time
from common import graphemes
from evalu import evaluate, SPLITS, GOLD
import sklearn_crfsuite

def feats(g):
    n = len(g); G = '^' + g + '$'; out = []
    for i in range(n):
        f = {'b': 1.0, 'c=' + g[i]: 1.0, 'e=%d' % min(n - i, 8): 1.0}
        for k in range(1, 5):
            f['L%d=%s' % (k, G[max(0, i + 1 - k):i + 1])] = 1.0
        for k in range(1, 6):
            f['R%d=%s' % (k, G[i + 1:i + 1 + k])] = 1.0
        if n - i <= 7: f['R*=' + G[i + 1:]] = 1.0
        if i <= 5: f['L*=' + G[:i + 1]] = 1.0
        f['LR=%s|%s' % (G[i:i + 1], G[i + 1:i + 2])] = 1.0
        out.append(f)
    return out

def labels(w):
    g, st = graphemes(w); gold = sorted(GOLD[w], key=lambda s: sorted(s))[0]
    return ['1' if i > 0 and st[i] in gold else '0' for i in range(len(g))]

def predict(model, ws):
    X = [feats(graphemes(w)[0]) for w in ws]; Y = model.predict(X); out = {}
    for w, y in zip(ws, Y):
        st = graphemes(w)[1]; out[w] = sorted(st[i] for i, t in enumerate(y) if t == '1' and i > 0)
    return out

def train(ws, c1, c2):
    m = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=c1, c2=c2, max_iterations=150, all_possible_transitions=True)
    m.fit([feats(graphemes(w)[0]) for w in ws], [labels(w) for w in ws]); return m

if __name__ == '__main__':
    split, c1, c2, mode = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    sp = SPLITS[split]; t0 = time.time(); m = train(sp['train'], c1, c2)
    tag = f'{split}_{c1}_{c2}'
    dev = evaluate({w: set(v) for w, v in predict(m, sp['dev']).items()}, sp['dev'])
    out = {'split': split, 'c1': c1, 'c2': c2, 'dev': dev, 'train_s': time.time() - t0}
    if mode == 'test':
        pr = predict(m, sp['test']); out['test'] = evaluate({w: set(v) for w, v in pr.items()}, sp['test'], boot=1000)
        json.dump(pr, open(RESULTS + f'/pred_crf_{split}.json', 'w', encoding='utf8'), ensure_ascii=False)
        # predictions on dev too (for error analysis); and for all words of root test
    json.dump(out, open(RESULTS + f'/res_crf_{tag}_{mode}.json', 'w'), indent=1)
    print(json.dumps(out), flush=True)
