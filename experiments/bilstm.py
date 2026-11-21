"""Character-level BiLSTM boundary tagger (JAX), the neural baseline of the chapter.

Each word is a sequence of graphemes (o‘, g‘, sh, ch count as one letter, as for the other
segmenters); the model predicts for every position after the first whether a new morph
begins there. Training labels are those of the CRF (crf.labels).

usage: python bilstm.py SPLIT HIDDEN LAYERS DROPOUT SEED dev|test
"""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, time
import numpy as np
import jax, jax.numpy as jnp, optax
from common import graphemes
from evalu import evaluate, SPLITS
from crf import labels

EMB, LR, BATCH, MAX_EPOCHS, PATIENCE = 64, 2e-3, 64, 40, 5
BUCKETS = (8, 12, 16, 20, 24, 32, 48, 64)

def bucket(n): return next(b for b in BUCKETS if b >= n)

def build_vocab(ws):
    chars = sorted({c for w in ws for c in graphemes(w)[0]})
    return {c: i + 2 for i, c in enumerate(chars)}  # 0 = padding, 1 = unknown

def encode(ws, vocab, with_labels=True):
    data = []
    for w in ws:
        g = graphemes(w)[0]
        x = [vocab.get(c, 1) for c in g]
        y = [int(t) for t in labels(w)] if with_labels else [0] * len(g)
        data.append((w, x, y))
    return data

def batches(data, rng=None):
    order = list(range(len(data)))
    if rng is not None: rng.shuffle(order)
    groups = {}
    for i in order: groups.setdefault(bucket(len(data[i][1])), []).append(i)
    out = []
    for T, idx in groups.items():
        for k in range(0, len(idx), BATCH):
            sel = idx[k:k + BATCH]; B = len(sel)
            X = np.zeros((B, T), np.int32); Y = np.zeros((B, T), np.float32); M = np.zeros((B, T), np.float32)
            for r, i in enumerate(sel):
                n = len(data[i][1]); X[r, :n] = data[i][1]; Y[r, :n] = data[i][2]; M[r, :n] = 1.0
            out.append((sel, X, Y, M))
    if rng is not None: rng.shuffle(out)
    return out

def init_params(key, V, H, L):
    keys = iter(jax.random.split(key, 4 + 4 * L))
    p = {'emb': jax.random.normal(next(keys), (V, EMB)) * 0.1, 'layers': []}
    d_in = EMB
    for _ in range(L):
        layer = {}
        for dirn in ('f', 'b'):
            s = 1.0 / np.sqrt(H)
            b = jnp.zeros(4 * H).at[H:2 * H].set(1.0)  # forget-gate bias 1
            layer[dirn] = {'Wx': jax.random.uniform(next(keys), (d_in, 4 * H), minval=-s, maxval=s),
                           'Wh': jax.random.uniform(next(keys), (H, 4 * H), minval=-s, maxval=s), 'b': b}
        p['layers'].append(layer); d_in = 2 * H
    s = 1.0 / np.sqrt(d_in)
    p['Wo'] = jax.random.uniform(next(keys), (d_in, 1), minval=-s, maxval=s); p['bo'] = jnp.zeros(1)
    return p

def run_dir(pd, xs, ms):
    """xs: (T, B, D); ms: (T, B, 1). Padding positions leave the state unchanged."""
    H = pd['Wh'].shape[0]; B = xs.shape[1]
    def step(carry, inp):
        h, c = carry; x, m = inp
        z = x @ pd['Wx'] + h @ pd['Wh'] + pd['b']
        i, f, g, o = jnp.split(z, 4, axis=-1)
        c_new = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h_new = jax.nn.sigmoid(o) * jnp.tanh(c_new)
        c = m * c_new + (1 - m) * c; h = m * h_new + (1 - m) * h
        return (h, c), h
    h0 = jnp.zeros((B, H)); _, hs = jax.lax.scan(step, (h0, h0), (xs, ms))
    return hs

def forward(p, X, M, key, rate, train):
    def drop(a, k):
        if not train or rate == 0: return a
        keep = jax.random.bernoulli(k, 1 - rate, a.shape)
        return jnp.where(keep, a / (1 - rate), 0.0)
    keys = jax.random.split(key, len(p['layers']) + 1)
    h = jnp.transpose(p['emb'][X], (1, 0, 2)); ms = jnp.transpose(M, (1, 0))[:, :, None]
    h = drop(h, keys[0])
    for li, layer in enumerate(p['layers']):
        fw = run_dir(layer['f'], h, ms)
        bw = run_dir(layer['b'], h[::-1], ms[::-1])[::-1]
        h = drop(jnp.concatenate([fw, bw], -1), keys[li + 1])
    logits = (h @ p['Wo'])[..., 0] + p['bo'][0]
    return jnp.transpose(logits, (1, 0))  # (B, T)

def make_fns(rate, opt):
    def loss_fn(p, X, Y, M, key):
        logits = forward(p, X, M, key, rate, True)
        mask = M.at[:, 0].set(0.0)  # the first letter never starts a new morph
        ll = optax.sigmoid_binary_cross_entropy(logits, Y)
        return (ll * mask).sum() / mask.sum()
    @jax.jit
    def train_step(p, s, X, Y, M, key):
        loss, g = jax.value_and_grad(loss_fn)(p, X, Y, M, key)
        upd, s = opt.update(g, s, p); return optax.apply_updates(p, upd), s, loss
    @jax.jit
    def probs(p, X, M):
        return jax.nn.sigmoid(forward(p, X, M, jax.random.PRNGKey(0), 0.0, False))
    return train_step, probs

def predict(p, probs_fn, data):
    out = {}
    for sel, X, Y, M in batches(data):
        P = np.asarray(probs_fn(p, X, M))
        for r, i in enumerate(sel):
            w = data[i][0]; st = graphemes(w)[1]; n = len(data[i][1])
            out[w] = sorted(st[t] for t in range(1, n) if P[r, t] > 0.5)
    return out

def train(split, H, L, rate, seed, log=True):
    sp = SPLITS[split]; vocab = build_vocab(sp['train'])
    tr, dv = encode(sp['train'], vocab), encode(sp['dev'], vocab)
    p = init_params(jax.random.PRNGKey(seed), len(vocab) + 2, H, L)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR)); s = opt.init(p)
    train_step, probs_fn = make_fns(rate, opt)
    rng = np.random.default_rng(seed); key = jax.random.PRNGKey(seed + 1)
    best, best_p, best_ep, hist, wait = -1.0, p, 0, [], 0
    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.time(); tot = 0.0; nb = 0
        for sel, X, Y, M in batches(tr, rng):
            key, k = jax.random.split(key)
            p, s, loss = train_step(p, s, X, Y, M, k); tot += float(loss); nb += 1
        dev = evaluate(predict(p, probs_fn, dv), sp['dev'])
        hist.append({'epoch': ep, 'loss': tot / nb, 'dev_F1': dev['F1'], 'dev_acc': dev['acc'], 'sec': time.time() - t0})
        if log: print(f"{split} H={H} L={L} dr={rate} seed={seed} ep={ep} loss={tot / nb:.4f} devF1={dev['F1']:.4f} acc={dev['acc']:.4f} {time.time() - t0:.0f}s", flush=True)
        if dev['F1'] > best: best, best_p, best_ep, wait = dev['F1'], p, ep, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    return best_p, probs_fn, vocab, {'best_dev_F1': best, 'best_epoch': best_ep, 'history': hist}

if __name__ == '__main__':
    split, H, L, rate, seed, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    p, probs_fn, vocab, info = train(split, H, L, rate, seed)
    sp = SPLITS[split]; tag = f'{split}_H{H}_L{L}_d{rate}_s{seed}'
    out = {'split': split, 'H': H, 'L': L, 'dropout': rate, 'seed': seed, **info}
    out['dev'] = evaluate(predict(p, probs_fn, encode(sp['dev'], vocab)), sp['dev'])
    if mode == 'test':
        pr = predict(p, probs_fn, encode(sp['test'], vocab))
        out['test'] = evaluate({w: set(v) for w, v in pr.items()}, sp['test'], boot=1000)
        json.dump(pr, open(RESULTS + f'/pred_bilstm_{tag}.json', 'w', encoding='utf8'), ensure_ascii=False)
    json.dump(out, open(RESULTS + f'/res_bilstm_{tag}_{mode}.json', 'w'), indent=1)
    print('RESULT', json.dumps({k: v for k, v in out.items() if k != 'history'}), flush=True)
