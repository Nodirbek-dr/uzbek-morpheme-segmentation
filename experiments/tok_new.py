"""Tokenizer audit extended to newer LLM tokenizers (Llama 3, Llama 4, Mistral Tekken),
plus the decomposition of word-internal cuts for every LLM tokenizer.

Tokenizer files:
  Llama 3 and Llama 4: tokenizer.model from the llama-models 0.3.0 wheel (PyPI), loaded as in its tokenizer.py
  Tekken: tekken_240911.json from the mistral-common 1.13.1 wheel (PyPI), loaded as in its tekken.py
"""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, base64, statistics
import tiktoken
from tokaudit import Tik, SP, audit, VAR, TOK
from evalu import SPLITS, GOLD

LLAMA3_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
LLAMA4_PAT = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

def bpe_file(path):
    ranks = {}
    for line in open(path, 'rb'):
        if line.strip():
            tok, rank = line.split(); ranks[base64.b64decode(tok)] = int(rank)
    return ranks

class Enc(Tik):
    def __init__(self, enc): self.enc = enc

def llama(ver):
    path = f'{TOK}/llama{ver}/tokenizer.model'
    return Enc(tiktoken.Encoding(name=f'llama{ver}', pat_str=LLAMA3_PAT if ver == 3 else LLAMA4_PAT,
                                 mergeable_ranks=bpe_file(path), special_tokens={}))

def tekken():
    d = json.load(open(f'{TOK}/tekken/tekken_240911.json', encoding='utf8')); cfg = d['config']
    inner = cfg['default_vocab_size'] - cfg['default_num_special_tokens']
    ranks = {base64.b64decode(x['token_bytes']): x['rank'] for x in d['vocab'][:inner]}
    assert set(ranks.values()) == set(range(inner))
    return Enc(tiktoken.Encoding(name='tekken', pat_str=cfg['pattern'], mergeable_ranks=ranks, special_tokens={})), inner

def decompose(tok, ws):
    """Classify every word-internal cut: on a gold boundary, inside the letters o‘/g‘, inside a UTF-8 character, elsewhere."""
    c = dict(total=0, on_morpheme=0, inside_o_g_letter=0, inside_utf8_char=0, inside_other=0)
    for w in ws:
        b, _ = tok(w); g = min(GOLD[w], key=lambda s: len(set(b) ^ s))
        letter = {i + 1 for i in range(len(w) - 1) if w[i] in 'og' and w[i + 1] == '‘'}
        for x in b:
            c['total'] += 1
            if x != int(x): c['inside_utf8_char'] += 1
            elif x in g: c['on_morpheme'] += 1
            elif x in letter: c['inside_o_g_letter'] += 1
            else: c['inside_other'] += 1
    c['share_on'] = c['on_morpheme'] / c['total']; c['share_letter'] = c['inside_o_g_letter'] / c['total']
    return c

if __name__ == '__main__':
    test = SPLITS['root']['test']; apo = [w for w in test if '‘' in w or '’' in w]; allw = sorted(GOLD)
    tk, inner = tekken()
    toks = {'GPT-4 (cl100k_base)': Tik('cl100k_base'), 'GPT-4o (o200k_base)': Tik('o200k_base'),
            'Llama 2 (32k SentencePiece)': SP(f'{TOK}/llama2/tokenizer.model', add_space=False),
            'Gemma (256k SentencePiece)': SP(f'{TOK}/gemma/tokenizer.model', add_space=True),
            'Llama 3 (128k BPE)': llama(3), 'Llama 4 (200k BPE)': llama(4), 'Mistral Tekken (131k BPE)': tk}
    sizes = {'Llama 3 (128k BPE)': len(toks['Llama 3 (128k BPE)'].enc._mergeable_ranks) if hasattr(toks['Llama 3 (128k BPE)'].enc, '_mergeable_ranks') else None}
    res = {'test_words': len(test), 'tekken_inner_vocab': inner, 'vocab': {}, 'llm': {}, 'variants': {'n_words': len(apo)}, 'all_words': {}, 'decomp': {}}
    for name, t in toks.items():
        if hasattr(t, 'enc'): res['vocab'][name] = t.enc.n_vocab
        r, _ = audit(t, test); res['llm'][name] = r
        res['decomp'][name] = decompose(t, test)
        res['variants'][name] = {}
        for v in VAR:
            rv, _ = audit(t, apo, v)
            res['variants'][name][v] = {'fertility': rv['fertility'], 'F1': rv['F1'], 'letter_split': rv['o_g_letter_split_share']}
        ra, _ = audit(t, allw); res['all_words'][name] = {k: ra[k] for k in ('P', 'R', 'F1', 'fertility')}
        print(f"{name:28} P={r['P']:.3f} R={r['R']:.3f} F1={r['F1']:.3f} fert={r['fertility']:.2f} clean={r['clean_words']:.3f} "
              f"letter={r['o_g_letter_split_share']:.3f} on={res['decomp'][name]['share_on']:.3f} allF1={ra['F1']:.3f} vocab={res['vocab'].get(name)}", flush=True)
    # does the gpt-oss encoding share the BPE ranks of o200k_base?
    import tiktoken_ext.openai_public as op
    if hasattr(op, 'o200k_harmony'):
        a, b = op.o200k_base(), op.o200k_harmony()
        res['o200k_harmony_same_ranks'] = a['mergeable_ranks'] == b['mergeable_ranks'] and a['pat_str'] == b['pat_str']
        print('o200k_harmony same ranks and pattern as o200k_base:', res['o200k_harmony_same_ranks'])
    json.dump(res, open(RESULTS + '/res_tok_new.json', 'w'), indent=1)
    print('done')
