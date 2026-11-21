"""Error analysis: are segmentation errors concentrated in suffix chains that the dictionary segments in more than one way?"""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json
from common import components
from evalu import GOLD, SPLITS, evaluate
from collections import defaultdict
recs, P, S = components()
pat = defaultdict(set); chain_of = defaultdict(set)
for r in recs:
    if r['post']:
        ch = ''.join(r['post']); pos, b = 0, []
        for s in r['post'][:-1]:
            pos += len(s); b.append(pos)
        pat[ch].add(tuple(b)); chain_of[r['surface']].add(ch)
variable = {c for c, ps in pat.items() if len(ps) > 1}
test = SPLITS['root']['test']
inv = [w for w in test if chain_of[w] & variable]; rest = [w for w in test if w not in set(inv)]
preds = {'CRF': json.load(open(RESULTS + '/pred_crf_root.json', encoding='utf8'))}
pb = json.load(open(RESULTS + '/pred_baselines.json', encoding='utf8'))
preds['Affix stripping'] = pb['affix_root']; preds['Morfessor'] = pb['morfessor_root']
out = {'test_words': len(test), 'in_variable_chain': len(inv), 'share_in_variable': len(inv) / len(test), 'by_system': {}}
for name, p in preds.items():
    p = {w: set(v) for w, v in p.items()}
    a = evaluate(p, inv); b = evaluate(p, rest)
    wrong = [w for w in test if p[w] not in GOLD[w]]
    out['by_system'][name] = {'variable': {k: a[k] for k in ('P', 'R', 'F1', 'acc')}, 'other': {k: b[k] for k in ('P', 'R', 'F1', 'acc')},
                              'wrong_words': len(wrong), 'wrong_in_variable_share': sum(w in set(inv) for w in wrong) / len(wrong)}
    print(name, json.dumps(out['by_system'][name]), flush=True)
# error types for CRF: under- vs over-segmentation
p = {w: set(v) for w, v in preds['CRF'].items()}
under = sum(1 for w in test if p[w] not in GOLD[w] and all(p[w] <= g for g in GOLD[w]))
over = sum(1 for w in test if p[w] not in GOLD[w] and any(p[w] > g for g in GOLD[w]))
out['crf_error_types'] = {'missing_only': under, 'extra_only': over, 'wrong_words': out['by_system']['CRF']['wrong_words']}
ex = [w for w in inv if p[w] not in GOLD[w]][:15]
def seg(w, b):
    c = [0] + sorted(b) + [len(w)]; return '/'.join(w[c[i]:c[i+1]] for i in range(len(c) - 1))
out['crf_examples_variable'] = [(seg(w, p[w]), seg(w, min(GOLD[w], key=len))) for w in ex]
json.dump(out, open(RESULTS + '/res_errors.json', 'w', encoding='utf8'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != 'by_system'}, ensure_ascii=False, indent=1))
