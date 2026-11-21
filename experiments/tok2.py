import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, statistics
import sentencepiece as spm
from tokaudit import Tik, SP, audit, VAR, TOK
from evalu import SPLITS, GOLD
test = SPLITS['root']['test']; train = SPLITS['root']['train']
apo = [w for w in test if '‘' in w]
toks = {'GPT-4 (cl100k_base)': Tik('cl100k_base'), 'GPT-4o (o200k_base)': Tik('o200k_base'),
        'Llama 2 (32k SentencePiece)': SP(f'{TOK}/llama2/tokenizer.model', add_space=False),
        'Gemma (256k SentencePiece)': SP(f'{TOK}/gemma/tokenizer.model', add_space=True)}
out = {'n_words_with_o_g': len(apo), 'letter_split': {}, 'fertility_o_g': {}}
for name, t in toks.items():
    out['letter_split'][name] = {}; out['fertility_o_g'][name] = {}
    for v in VAR:
        r, _ = audit(t, apo, v); out['letter_split'][name][v] = r['o_g_letter_split_share']; out['fertility_o_g'][name][v] = r['fertility']
    print(name, out['letter_split'][name], {k: round(x, 2) for k, x in out['fertility_o_g'][name].items()}, flush=True)
# in-domain models that may merge across the apostrophe (no Unicode-script splitting)
open(RESULTS + '/sp/train_words.txt', 'w', encoding='utf8').write('\n'.join(train) + '\n')
out['in_domain_noscript'] = {}
for kind in ('bpe', 'unigram'):
    for vocab in (4000, 16000):
        tag = f'{kind}{vocab}_noscript'; prefix = RESULTS + f'/sp/{tag}'
        spm.SentencePieceTrainer.train(input=RESULTS + '/sp/train_words.txt', model_prefix=prefix, vocab_size=vocab, model_type=kind,
                                       character_coverage=1.0, add_dummy_prefix=True, hard_vocab_limit=False, split_by_unicode_script=False,
                                       bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1, minloglevel=2)
        t = SP(prefix + '.model', add_space=False); r, _ = audit(t, test); out['in_domain_noscript'][tag] = r
        print(tag, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items() if k in ('P', 'R', 'F1', 'fertility', 'clean_words', 'o_g_letter_split_share')}, flush=True)
json.dump(out, open(RESULTS + '/res_tok2.json', 'w'), indent=1)
