import json
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

F = json.load(open(ROOT / 'results' / 'FINAL_NUMBERS.json'))
T = F['tokenizers']; L = T['llm_test']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
C = {'llm': '#2a78d6', 'dom': '#eb6834', 'seg': '#1baf7a'}      # validated slots 1-3 (light, all-pairs)
M = {'llm': 'o', 'dom': 's', 'seg': '^'}                         # shape = secondary encoding for grayscale print
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
                     'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
seg = F['segmentation']
pts = [  # group, label, x, y, label position in data coordinates (None = offset), offset/ha
    ('llm', 'GPT-4', L['GPT-4 (cl100k_base)'], (4.70, 0.288)),
    ('llm', 'GPT-4o / GPT-5', L['GPT-4o (o200k_base)'], (3.62, 0.300)),
    ('llm', 'Llama 2', L['Llama 2 (32k SentencePiece)'], (5.02, 0.392)),
    ('llm', 'Llama 3', L['Llama 3 (128k BPE)'], (4.50, 0.392)),
    ('llm', 'Llama 4', L['Llama 4 (200k BPE)'], (3.80, 0.392)),
    ('llm', 'Tekken', L['Mistral Tekken (131k BPE)'], (4.26, 0.272)),
    ('llm', 'Gemma', L['Gemma (256k SentencePiece)'], (4.42, 0.445)),
    ('dom', 'BPE 4k', T['in_domain_noscript']['bpe4000_noscript'], (3.39, 0.402)),
    ('dom', 'BPE 16k', T['in_domain_noscript']['bpe16000_noscript'], (2.98, 0.300)),
    ('dom', 'Unigram 4k', T['in_domain_noscript']['unigram4000_noscript'], (4.29, 0.556)),
    ('dom', 'Unigram 16k', T['in_domain_noscript']['unigram16000_noscript'], (3.45, 0.556)),
]
sup = [('CRF', T['crf_morphs_per_word'], seg['crf']['root']['F1'], (2.45, 0.800)),
       ('CRF + root lexicon', T['crflexjk_morphs_per_word'], seg['crf_lex_jackknife']['root']['F1'], (3.05, 0.950))]
if 'bilstm' in seg:
    sup.append(('BiLSTM tagger', T['bilstm_morphs_per_word'], seg['bilstm']['root']['F1'], (3.27, 0.815)))
fig, ax = plt.subplots(figsize=(4.6, 3.3), dpi=600)
ax.set_axisbelow(True); ax.grid(axis='both', color=GRID, linewidth=0.5)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
gold = T['gold_morphs_per_word_test']
ax.axvline(gold, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
ax.text(gold + 0.05, 0.69, f'gold segmentation:\n{gold:.2f} morphemes per word', color=INK2, fontsize=8, va='center', ha='left')
lead = dict(arrowstyle='-', color=INK2, linewidth=0.5, shrinkA=0, shrinkB=3.5)
box = dict(boxstyle='square,pad=0.12', fc='white', ec='none')
def mark(g, x, y):
    ax.plot(x, y, marker=M[g], color=C[g], markersize=6.5, markeredgecolor='white', markeredgewidth=0.9, linestyle='none', zorder=3)
def label(text, x, y, pos):
    ax.annotate(text, (x, y), xytext=pos, textcoords='data', ha='center', va='center', color=INK, fontsize=8, zorder=4, bbox=box,
                arrowprops=lead)
for g, lab, r, pos in pts:
    mark(g, r['fertility'], r['F1']); label(lab, r['fertility'], r['F1'], pos)
for lab, x, y, pos in sup:
    mark('seg', x, y); label(lab, x, y, pos)
ax.set_xlim(2.3, 5.3); ax.set_ylim(0.25, 0.97)
ax.set_xlabel('Tokens or segments per word', color=INK); ax.set_ylabel('Boundary F1 against gold morphemes', color=INK)
handles = [Line2D([], [], marker=M[k], color=C[k], markeredgecolor='white', markeredgewidth=0.9, markersize=6.5, linestyle='none') for k in ('llm', 'dom', 'seg')]
ax.legend(handles, ['LLM tokenizers', 'In-domain subword models', 'Supervised segmenters'], loc='upper right', frameon=False,
          fontsize=8, handletextpad=0.3, borderaxespad=0.2, labelcolor=INK)
fig.tight_layout(pad=0.3)
fig.savefig(ROOT / 'figures' / 'fig1_tokenizer_alignment.png', dpi=600, facecolor='white')
print('saved')
