import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json
from common import components
from evalu import GOLD, SPLITS, best_gold
from collections import Counter, defaultdict
recs, P, S = components()
R = {r['root'] for r in recs}
segs_of = defaultdict(list)
for r in recs: segs_of[r['surface']].append(r['segs'])
def btype(segs, k):
    """type of the boundary before segment k"""
    s = segs[k]; prev = segs[k - 1]
    if prev in P and all(x in P for x in segs[:k]): return 'after prefix'
    if s in S: return 'before suffix'
    if s in R: return 'compound'
    return 'other'
test = SPLITS['root']['test']; out = {}
import os
systems = [('CRF', RESULTS + '/pred_crf_root.json'), ('CRF+lex(jk)', RESULTS + '/pred_crflexjk_root.json')]
if os.path.exists(RESULTS + '/res_bilstm_main.json'):
    for i, f in enumerate(json.load(open(RESULTS + '/res_bilstm_main.json'))['preds_root']): systems.append((f'BiLSTM seed{i + 1}', RESULTS + '/' + f))
for name, f in systems:
    p = {w: set(v) for w, v in json.load(open(f, encoding='utf8')).items()}
    miss, extra, gold_types = Counter(), Counter(), Counter()
    for w in test:
        g = best_gold(w, p[w])
        segs = next(sg for sg in segs_of[w] if set(__import__('itertools').accumulate(len(x) for x in sg[:-1])) == g) if g else segs_of[w][0]
        cuts = list(__import__('itertools').accumulate(len(x) for x in segs[:-1]))
        for k, c in enumerate(cuts, start=1):
            t = btype(segs, k); gold_types[t] += 1
            if c not in p[w]: miss[t] += 1
        extra['total'] += len(p[w] - g)
    wrong = [w for w in test if p[w] not in GOLD[w]]
    out[name] = {'gold_boundaries_by_type': dict(gold_types), 'missed_by_type': dict(miss),
                 'recall_by_type': {t: 1 - miss[t] / gold_types[t] for t in gold_types}, 'extra_total': extra['total'],
                 'wrong_words': len(wrong), 'missing_only': sum(all(p[w] <= g for g in GOLD[w]) for w in wrong)}
runs = [v for k, v in out.items() if k.startswith('BiLSTM seed')]
if runs:  # mean over the BiLSTM seeds
    out['BiLSTM'] = {'recall_by_type': {t: sum(r['recall_by_type'][t] for r in runs) / len(runs) for t in runs[0]['recall_by_type']},
                     'wrong_words': sum(r['wrong_words'] for r in runs) / len(runs), 'missing_only': sum(r['missing_only'] for r in runs) / len(runs),
                     'missing_only_share': sum(r['missing_only'] / r['wrong_words'] for r in runs) / len(runs), 'seeds': len(runs)}
print(json.dumps(out, indent=1))
json.dump(out, open(RESULTS + '/res_errors2.json', 'w'), indent=1)
