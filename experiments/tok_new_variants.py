"""Apostrophe variants for all seven LLM tokenizers on the test words with o‘/g‘ (same word set as tok2.py)."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json
from tokaudit import Tik, SP, audit, VAR, TOK
from tok_new import llama, tekken
from evalu import SPLITS
test = SPLITS['root']['test']; apo = [w for w in test if '‘' in w]
toks = {'GPT-4 (cl100k_base)': Tik('cl100k_base'), 'GPT-4o (o200k_base)': Tik('o200k_base'),
        'Llama 2 (32k SentencePiece)': SP(f'{TOK}/llama2/tokenizer.model', add_space=False),
        'Gemma (256k SentencePiece)': SP(f'{TOK}/gemma/tokenizer.model', add_space=True),
        'Llama 3 (128k BPE)': llama(3), 'Llama 4 (200k BPE)': llama(4), 'Mistral Tekken (131k BPE)': tekken()[0]}
out = {'n_words_with_o_g': len(apo), 'letter_split': {}, 'fertility_o_g': {}}
for name, t in toks.items():
    out['letter_split'][name] = {}; out['fertility_o_g'][name] = {}
    for v in VAR:
        r, _ = audit(t, apo, v); out['letter_split'][name][v] = r['o_g_letter_split_share']; out['fertility_o_g'][name][v] = r['fertility']
    print(f"{name:28}", {v: (round(out['fertility_o_g'][name][v], 2), round(out['letter_split'][name][v], 3)) for v in VAR}, flush=True)
json.dump(out, open(RESULTS + '/res_tok_new_variants.json', 'w'), indent=1)
