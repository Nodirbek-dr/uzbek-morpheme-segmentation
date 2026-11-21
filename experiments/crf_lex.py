"""CRF segmenter with root-lexicon features taken from the training part of the dictionary."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, time
from common import graphemes, components
from evalu import evaluate, SPLITS, GOLD
from crf import feats as base_feats, labels
import sklearn_crfsuite

def lexicon(train_words):
    tw = set(train_words); recs, P, S = components()
    return {graphemes(r['root'])[0] for r in recs if r['surface'] in tw and len(graphemes(r['root'])[0]) >= 3}

def feats_lex(g, L, maxlen=12):
    F = base_feats(g); n = len(g)
    for i in range(n):
        rl = max((k for k in range(3, min(maxlen, n - i) + 1) if g[i:i + k] in L), default=0)
        ll = max((k for k in range(3, min(maxlen, i) + 1) if g[i - k:i] in L), default=0)
        if rl: F[i]['RL=%d' % min(rl, 7)] = 1.0; F[i]['RLend=%d' % min(n - i - rl, 6)] = 1.0
        if ll: F[i]['LL=%d' % min(ll, 7)] = 1.0; F[i]['LLstart=%d' % min(i - ll, 4)] = 1.0
    return F

if __name__ == '__main__':
    split = sys.argv[1]; sp = SPLITS[split]; L = lexicon(sp['train']); t0 = time.time()
    m = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.0, c2=0.3, max_iterations=150, all_possible_transitions=True)
    m.fit([feats_lex(graphemes(w)[0], L) for w in sp['train']], [labels(w) for w in sp['train']])
    def predict(ws):
        Y = m.predict([feats_lex(graphemes(w)[0], L) for w in ws]); out = {}
        for w, y in zip(ws, Y):
            st = graphemes(w)[1]; out[w] = sorted(st[i] for i, t in enumerate(y) if t == '1' and i > 0)
        return out
    dev = evaluate({w: set(v) for w, v in predict(sp['dev']).items()}, sp['dev'])
    pr = predict(sp['test']); test = evaluate({w: set(v) for w, v in pr.items()}, sp['test'], boot=1000)
    json.dump(pr, open(RESULTS + f'/pred_crflex_{split}.json', 'w', encoding='utf8'), ensure_ascii=False)
    res = {'split': split, 'lexicon_size': len(L), 'dev': dev, 'test': test, 'train_s': time.time() - t0}
    json.dump(res, open(RESULTS + f'/res_crflex_{split}.json', 'w'), indent=1); print(json.dumps(res), flush=True)
