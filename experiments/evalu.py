import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import json, random
GOLD = {w: [set(g) for g in gs] for w, gs in json.load(open(str(ROOT / 'benchmark' / 'gold.json'), encoding='utf8')).items()}
SPLITS = json.load(open(str(ROOT / 'benchmark' / 'splits.json'), encoding='utf8'))

def best_gold(w, p):
    return min(GOLD[w], key=lambda g: (len(p - g) + len(g - p), -len(p & g)))

def counts(pred, ws):
    rows = []
    for w in ws:
        p = set(pred[w]); g = best_gold(w, p)
        rows.append((len(p & g), len(p - g), len(g - p), int(p == g)))
    return rows

def summarize(rows):
    tp = sum(r[0] for r in rows); fp = sum(r[1] for r in rows); fn = sum(r[2] for r in rows)
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {'P': P, 'R': R, 'F1': F, 'acc': sum(r[3] for r in rows) / len(rows), 'tp': tp, 'fp': fp, 'fn': fn, 'n': len(rows)}

def evaluate(pred, ws, boot=0, seed=7):
    rows = counts(pred, ws); res = summarize(rows)
    if boot:
        rng = random.Random(seed); F, A = [], []
        for _ in range(boot):
            s = summarize([rows[rng.randrange(len(rows))] for _ in rows]); F.append(s['F1']); A.append(s['acc'])
        F.sort(); A.sort(); lo, hi = int(0.025 * boot), int(0.975 * boot) - 1
        res['F1_ci'] = (F[lo], F[hi]); res['acc_ci'] = (A[lo], A[hi])
    return res
