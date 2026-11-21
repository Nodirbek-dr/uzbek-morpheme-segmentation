"""Root-disjoint test words split by whether any of their roots (first root or compound part) occurs
anywhere in the training words; scores of every segmenter on both subsets."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, glob
from collections import defaultdict
from common import components
from evalu import SPLITS, evaluate
recs, P, S = components()
R = {r['root'] for r in recs}
first, later = defaultdict(set), defaultdict(set)
for r in recs:
    first[r['surface']].add(r['root'])
    for seg in r['post']:
        if seg in R and seg not in S: later[r['surface']].add(seg)
sp = SPLITS['root']; tr = set(sp['train'])
tf = set().union(*[first[w] for w in tr]); tl = set().union(*[later[w] for w in tr])
test = sp['test']
ovs = {w for w in test if (first[w] & tl) or (later[w] & tf) or (later[w] & tl)}
groups = {'all': test, 'no_overlap': [w for w in test if w not in ovs], 'overlap': [w for w in test if w in ovs]}
B = json.load(open(RESULTS + '/pred_baselines.json'))
systems = {'affix': B['affix_root'], 'morfessor': B['morfessor_root'],
           'crf': json.load(open(RESULTS + '/pred_crf_root.json')), 'crf_lex_jackknife': json.load(open(RESULTS + '/pred_crflexjk_root.json'))}
main = json.load(open(RESULTS + '/res_bilstm_main.json')) if glob.glob(RESULTS + '/res_bilstm_main.json') else None
if main:
    for i, f in enumerate(main['preds_root']): systems[f'bilstm_seed{i + 1}'] = json.load(open(RESULTS + '/' + f))
out = {'n': {g: len(ws) for g, ws in groups.items()},
       'compound_share': {g: sum(bool(later[w]) for w in ws) / len(ws) for g, ws in groups.items()},
       'first_as_later_in_train': sum(bool(first[w] & tl) for w in test)}
for name, pred in systems.items():
    for g, ws in groups.items():
        out[f'{name}|{g}'] = evaluate(pred, ws, boot=1000 if g != 'all' else 0)
        e = out[f'{name}|{g}']
        print(f"{name:18} {g:10} n={len(ws):5} F1={e['F1']:.3f} Acc={100 * e['acc']:.1f}", flush=True)
if main:  # mean over the BiLSTM seeds
    for g in groups:
        runs = [out[k] for k in out if k.startswith('bilstm_seed') and k.endswith('|' + g)]
        out[f'bilstm|{g}'] = {m: sum(r[m] for r in runs) / len(runs) for m in ('P', 'R', 'F1', 'acc')} | {'seeds': len(runs)}
        print(f"bilstm mean         {g:10} F1={out[f'bilstm|{g}']['F1']:.3f} Acc={100 * out[f'bilstm|{g}']['acc']:.1f}", flush=True)
json.dump(out, open(RESULTS + '/res_overlap_subsets.json', 'w'), indent=1)
