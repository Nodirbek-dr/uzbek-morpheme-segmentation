"""Rule-based affix stripping and unsupervised Morfessor Baseline."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, time, random
from common import graphemes, build_affix_sets, load_affix_forms
from evalu import evaluate, SPLITS
import morfessor

P, S = build_affix_sets(load_affix_forms())
SUFg = sorted({graphemes(s)[0] for s in S}, key=len, reverse=True)
PREg = sorted({graphemes(p)[0] for p in P}, key=len, reverse=True)
VOW = set('aeiouö')
hv = lambda x: any(c in VOW for c in x)

def affix_strip(w, m):
    g, st = graphemes(w); end = len(g); bs = []
    while True:
        for s in SUFg:
            if end - len(s) >= m and g[:end].endswith(s) and hv(g[:end - len(s)]):
                end -= len(s); bs.append(end); break
        else:
            break
    for p in PREg:
        if g.startswith(p) and end - len(p) >= m and hv(g[len(p):end]):
            bs.append(len(p)); break
    return {st[i] for i in bs if 0 < i < len(g)}

res = {'affix': {}, 'morfessor': {}}
preds = {}
for split in ('random', 'root'):
    sp = SPLITS[split]
    dev = {m: evaluate({w: affix_strip(w, m) for w in sp['dev']}, sp['dev'])['F1'] for m in (2, 3, 4, 5)}
    m = max(dev, key=dev.get)
    pr = {w: sorted(affix_strip(w, m)) for w in sp['test']}
    res['affix'][split] = {'dev_F1_by_m': dev, 'm': m, 'test': evaluate({w: set(v) for w, v in pr.items()}, sp['test'], boot=1000)}
    preds[f'affix_{split}'] = pr
    print('affix', split, m, res['affix'][split]['test'], flush=True)

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

for split in ('root', 'random'):
    sp = SPLITS[split]
    grid = (0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0) if split == 'root' else (res['morfessor']['root']['cw'],)
    best = None; dev = {}
    for cw in grid:
        t0 = time.time(); model = train_morf(sp['train'], cw)
        f = evaluate({w: morf_pred(model, w) for w in sp['dev']}, sp['dev'])['F1']; dev[cw] = f
        print('morf', split, cw, round(f, 4), round(time.time() - t0, 1), 's', flush=True)
        if best is None or f > best[0]: best = (f, cw, model)
    pr = {w: sorted(morf_pred(best[2], w)) for w in sp['test']}
    res['morfessor'][split] = {'dev_F1_by_cw': dev, 'cw': best[1], 'test': evaluate({w: set(v) for w, v in pr.items()}, sp['test'], boot=1000)}
    preds[f'morfessor_{split}'] = pr
    print('morfessor', split, res['morfessor'][split]['test'], flush=True)

json.dump(res, open(RESULTS + '/res_baselines.json', 'w'), indent=1)
json.dump(preds, open(RESULTS + '/pred_baselines.json', 'w', encoding='utf8'), ensure_ascii=False)
print('done')
