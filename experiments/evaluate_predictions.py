"""Recompute the scores of Table 2 from the saved predictions in results/ (no model training needed).
The BiLSTM row of Table 2 is the mean over its three seeds, which are printed one by one as well."""
import sys, json, pathlib, statistics
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from evalu import evaluate, SPLITS

def load(name):
    return json.load(open(ROOT / 'results' / name, encoding='utf8'))

base = load('pred_baselines.json'); bl = load('res_bilstm_main.json')
systems = {
    'Affix stripping': {s: base[f'affix_{s}'] for s in ('random', 'root')},
    'Morfessor Baseline': {s: base[f'morfessor_{s}'] for s in ('random', 'root')},
    'CRF': {s: load(f'pred_crf_{s}.json') for s in ('random', 'root')},
    'CRF + root lexicon': {'random': load('pred_crflex_random.json'), 'root': load('pred_crflexjk_root.json')},
}
for i in range(len(bl['preds_root'])):
    systems[f'BiLSTM seed {i + 1}'] = {'random': load(bl['preds_random'][i]), 'root': load(bl['preds_root'][i])}
print(f"{'System':22s} {'split':7s}     P      R     F1    Acc")
res = {}
for name, preds in systems.items():
    for split, pred in preds.items():
        r = evaluate({w: set(v) for w, v in pred.items()}, SPLITS[split]['test']); res[name, split] = r
        print(f"{name:22s} {split:7s} {r['P']:.3f}  {r['R']:.3f}  {r['F1']:.3f}  {100 * r['acc']:.1f}")
for split in ('random', 'root'):
    rs = [res[k] for k in res if k[0].startswith('BiLSTM') and k[1] == split]
    m = {k: statistics.mean(r[k] for r in rs) for k in ('P', 'R', 'F1', 'acc')}
    print(f"{'BiLSTM mean':22s} {split:7s} {m['P']:.3f}  {m['R']:.3f}  {m['F1']:.3f}  {100 * m['acc']:.1f}")
