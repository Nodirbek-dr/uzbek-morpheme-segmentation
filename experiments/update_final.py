"""Merge the new results (seven tokenizers, BiLSTM tagger, overlap subsets) into FINAL_NUMBERS.json."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import json, os, statistics, sys
from evalu import GOLD, SPLITS
F = json.load(open(RESULTS + '/FINAL_NUMBERS.json')); T = F['tokenizers']
N = json.load(open(RESULTS + '/res_tok_new.json')); V = json.load(open(RESULTS + '/res_tok_new_variants.json'))
for name in ('Llama 3 (128k BPE)', 'Llama 4 (200k BPE)', 'Mistral Tekken (131k BPE)'):
    T['llm_test'][name] = N['llm'][name]; T['llm_all_words'][name] = N['all_words'][name]
T['decomp'] = N['decomp']; T['vocab'] = N['vocab']; T['tekken_inner_vocab'] = N['tekken_inner_vocab']
T['o200k_harmony_same_ranks'] = N.get('o200k_harmony_same_ranks')
T['letter_split'] = V['letter_split']; T['fertility_o_g'] = V['fertility_o_g']; T['n_words_with_o_g'] = V['n_words_with_o_g']
if os.path.exists(RESULTS + '/res_bilstm_main.json'):
    B = json.load(open(RESULTS + '/res_bilstm_main.json')); F['segmentation']['bilstm'] = B['summary']
    test = SPLITS['root']['test']; per_seed = []
    for f in B['preds_root']:
        pr = json.load(open(RESULTS + '/' + f)); per_seed.append(statistics.mean(1 + len(pr[w]) for w in test))
    T['bilstm_morphs_per_word'] = statistics.mean(per_seed); T['bilstm_morphs_per_word_per_seed'] = per_seed
if os.path.exists(RESULTS + '/res_overlap_subsets.json'):
    F['overlap_subsets'] = json.load(open(RESULTS + '/res_overlap_subsets.json'))
json.dump(F, open(RESULTS + '/FINAL_NUMBERS.json', 'w'), indent=1)
print('FINAL_NUMBERS updated; tokenizers:', len(T['llm_test']), '| bilstm' if 'bilstm' in F['segmentation'] else '')
