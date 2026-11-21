"""Morphological alignment of LLM tokenizers and in-domain subword models with gold morpheme boundaries."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, re, os, statistics
from evalu import GOLD, SPLITS, evaluate
import tiktoken, sentencepiece as spm
import tiktoken_ext.openai_public as op

TOK = str(ROOT / 'tokenizers')
_orig = op.load_tiktoken_bpe
def _local(url, expected_hash=None):          # same files, verified against the official SHA-256
    return _orig(f'{TOK}/tiktoken_rs/' + url.rsplit('/', 1)[1], expected_hash=expected_hash)
op.load_tiktoken_bpe = _local

def char_bounds(text, piece_bytes):
    """Token boundaries inside the word (text = ' ' + word) as word-relative char offsets.
    A boundary that falls inside a multi-byte character is returned as offset + 0.5."""
    off, at = 0, {}
    for i, ch in enumerate(text):
        at[off] = i; off += len(ch.encode('utf8'))
    at[off] = len(text)
    starts = sorted(at)
    pos, out = 0, set()
    for pb in piece_bytes[:-1]:
        pos += len(pb)
        if pos in at:
            c = at[pos] - 1
            if 0 < c < len(text) - 1: out.add(c)
        else:
            prev = max(s for s in starts if s < pos); c = at[prev] - 1
            out.add(c + 0.5)
    return out

class Tik:
    def __init__(self, name): self.enc = tiktoken.Encoding(**getattr(op, name)())
    def __call__(self, w):
        text = ' ' + w; ids = self.enc.encode_ordinary(text)
        bts = [self.enc.decode_single_token_bytes(t) for t in ids]
        assert b''.join(bts) == text.encode('utf8')
        return char_bounds(text, bts), len(ids)

class SP:
    def __init__(self, path, add_space): self.sp = spm.SentencePieceProcessor(model_file=path); self.add_space = add_space
    def __call__(self, w):
        ids = self.sp.encode((' ' + w) if self.add_space else w); bts = []
        for i in ids:
            p = self.sp.id_to_piece(i); m = re.fullmatch(r'<0x([0-9A-Fa-f]{2})>', p)
            bts.append(bytes([int(m.group(1), 16)]) if m else p.replace('▁', ' ').encode('utf8'))
        text = ' ' + w
        if b''.join(bts) != text.encode('utf8'):
            raise ValueError((w, [self.sp.id_to_piece(i) for i in ids]))
        return char_bounds(text, bts), len(ids)

def train_sp(kind, vocab, words, tag):
    os.makedirs(RESULTS + '/sp', exist_ok=True); inp = RESULTS + '/sp/train_words.txt'
    open(inp, 'w', encoding='utf8').write('\n'.join(words) + '\n')
    prefix = RESULTS + f'/sp/{tag}'
    spm.SentencePieceTrainer.train(input=inp, model_prefix=prefix, vocab_size=vocab, model_type=kind,
                                   character_coverage=1.0, add_dummy_prefix=True, hard_vocab_limit=False,
                                   bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1, minloglevel=2)
    return SP(prefix + '.model', add_space=False)

VAR = {'dictionary': {}, 'modifier': {'‘': 'ʻ', '’': 'ʼ'}, 'ascii': {'‘': "'", '’': "'"}}
def variant(w, v): return ''.join(VAR[v].get(c, c) for c in w)

def audit(tok, ws, v='dictionary'):
    pred, fert, intra_letter, n_letter = {}, [], 0, 0
    for w in ws:
        b, n = tok(variant(w, v)); pred[w] = b; fert.append(n)
        # split inside the letters o‘ / g‘ (boundary right after o or g, before the apostrophe)
        letter_pos = [i + 1 for i in range(len(w) - 1) if w[i] in 'og' and w[i + 1] == '‘']
        if letter_pos:
            n_letter += 1; intra_letter += any(p in b for p in letter_pos)
    r = evaluate(pred, ws)
    clean = sum(all(x in min(GOLD[w], key=lambda g: len(set(pred[w]) ^ g)) for x in pred[w]) for w in ws) / len(ws)
    r.update({'fertility': statistics.mean(fert), 'clean_words': clean,
              'o_g_letter_split_share': intra_letter / n_letter if n_letter else None, 'words_with_o_g': n_letter})
    return r, pred

if __name__ == '__main__':
    sp_root = SPLITS['root']; test = sp_root['test']
    toks = {'GPT-4 (cl100k_base)': Tik('cl100k_base'), 'GPT-4o (o200k_base)': Tik('o200k_base'),
            'Llama 2 (32k SentencePiece)': SP(f'{TOK}/llama2/tokenizer.model', add_space=False),
            'Gemma (256k SentencePiece)': SP(f'{TOK}/gemma/tokenizer.model', add_space=True)}
    res = {'test_words': len(test), 'gold_morphs_per_word': statistics.mean(1 + len(min(GOLD[w], key=len)) for w in test), 'llm': {}, 'in_domain': {}, 'variants': {}, 'all_words': {}}
    for name, t in toks.items():
        r, _ = audit(t, test); res['llm'][name] = r; print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    # apostrophe variants on the test words that contain an apostrophe
    apo = [w for w in test if '‘' in w or '’' in w]
    res['variants']['n_words'] = len(apo)
    for name, t in toks.items():
        res['variants'][name] = {}
        for v in VAR:
            r, _ = audit(t, apo, v); res['variants'][name][v] = {'fertility': r['fertility'], 'P': r['P'], 'R': r['R'], 'F1': r['F1']}
        print('variants', name, {v: round(x['fertility'], 3) for v, x in res['variants'][name].items()}, flush=True)
    # all word types (LLM tokenizers are not trained on the benchmark)
    allw = sorted(GOLD)
    for name, t in toks.items():
        r, _ = audit(t, allw); res['all_words'][name] = {k: r[k] for k in ('P', 'R', 'F1', 'acc', 'fertility', 'clean_words', 'o_g_letter_split_share')}
    # in-domain subword models trained on the training words of the root-disjoint split
    for kind in ('bpe', 'unigram'):
        for vocab in (1000, 4000, 16000):
            tag = f'{kind}{vocab}'
            try:
                t = train_sp(kind, vocab, sp_root['train'], tag)
            except Exception as e:
                print('skip', tag, e); continue
            r, _ = audit(t, test); r['vocab_actual'] = t.sp.get_piece_size(); res['in_domain'][tag] = r
            print(tag, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    json.dump(res, open(RESULTS + '/res_tok.json', 'w'), indent=1)
    print('done')
