"""Paired bootstrap over test words for differences in boundary F1 and word accuracy between segmenters.
The same resampled word sets are scored for both systems, so the interval reflects the difference itself."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json
import numpy as np
from evalu import SPLITS, counts

B, SEED = 10000, 7
def arrays(pred, ws):
    rows = np.array(counts(pred, ws), dtype=np.int64)  # tp, fp, fn, exact
    return rows
def f1(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn)
def compare(pa, pb, ws):
    a, b = arrays(pa, ws), arrays(pb, ws); n = len(ws); rng = np.random.default_rng(SEED)
    dF, dA = [], []
    for _ in range(B // 1000):
        idx = rng.integers(0, n, size=(1000, n))
        sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
        dF.append(f1(sa[:, 0], sa[:, 1], sa[:, 2]) - f1(sb[:, 0], sb[:, 1], sb[:, 2])); dA.append((sa[:, 3] - sb[:, 3]) / n)
    dF, dA = np.concatenate(dF), np.concatenate(dA)
    ta, tb = a.sum(axis=0), b.sum(axis=0)
    obs_F = f1(*ta[:3]) - f1(*tb[:3]); obs_A = (ta[3] - tb[3]) / n
    p = lambda d, obs: float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {'dF1': float(obs_F), 'dF1_ci': [float(np.percentile(dF, 2.5)), float(np.percentile(dF, 97.5))], 'p_F1': p(dF, obs_F),
            'dAcc': float(obs_A), 'dAcc_ci': [float(np.percentile(dA, 2.5)), float(np.percentile(dA, 97.5))], 'p_acc': p(dA, obs_A), 'n': n, 'B': B}

if __name__ == '__main__':
    M = json.load(open(RESULTS + '/res_bilstm_main.json'))
    load = lambda f: {w: set(v) for w, v in json.load(open(RESULTS + '/' + f, encoding='utf8')).items()}
    base = {'root': {'CRF': load('pred_crf_root.json'), 'CRF+lexJK': load('pred_crflexjk_root.json')},
            'random': {'CRF': load('pred_crf_random.json'), 'CRF+lex': load('pred_crflex_random.json')}}
    out = {}
    for split in ('root', 'random'):
        S = dict(base[split]); lex = 'CRF+lexJK' if split == 'root' else 'CRF+lex'
        pairs = [(lex, 'CRF')]
        for i, f in enumerate(M['preds_' + split]):
            S[f'BiLSTM seed{i + 1}'] = load(f); pairs += [(f'BiLSTM seed{i + 1}', 'CRF'), (f'BiLSTM seed{i + 1}', lex)]
        ws = SPLITS[split]['test']
        for a, b in pairs:
            r = compare(S[a], S[b], ws); out[f'{split}|{a} - {b}'] = r
            print(f"{split:6} {a:14} - {b:10} dF1={r['dF1']:+.4f} CI [{r['dF1_ci'][0]:+.4f}, {r['dF1_ci'][1]:+.4f}] p={r['p_F1']:.4f} | "
                  f"dAcc={100 * r['dAcc']:+.2f} CI [{100 * r['dAcc_ci'][0]:+.2f}, {100 * r['dAcc_ci'][1]:+.2f}] p={r['p_acc']:.4f}", flush=True)
    json.dump(out, open(RESULTS + '/res_paired_bootstrap.json', 'w'), indent=1)
