"""Benchmark construction: word types, random split and root-disjoint split (80/10/10)."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, random
from common import components
from collections import defaultdict, Counter

SEED = 20261119
recs, P, S = components()
types = {}
for r in recs:
    t = types.setdefault(r['surface'], {'golds': set(), 'roots': set()})
    t['golds'].add(r['b']); t['roots'].add(r['root'])
words = sorted(types)
n = len(words)

# random split over word types
w = words[:]; random.Random(SEED).shuffle(w)
a, b = round(0.8 * n), round(0.9 * n)
rand = {'train': sorted(w[:a]), 'dev': sorted(w[a:b]), 'test': sorted(w[b:])}

# root-disjoint split: union of words that share a root (homographs can link two roots)
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry: parent[rx] = ry
for word, t in types.items():
    for root in t['roots']:
        union(('w', word), ('r', root))
groups = defaultdict(list)
for word in words:
    groups[find(('w', word))].append(word)
glist = sorted(groups.values(), key=lambda g: g[0])
random.Random(SEED).shuffle(glist)
tr, dv, te = [], [], []
for g in glist:
    if len(tr) < 0.8 * n: tr += g
    elif len(dv) < 0.1 * n: dv += g
    else: te += g
root = {'train': sorted(tr), 'dev': sorted(dv), 'test': sorted(te)}

def roots_of(ws): return set().union(*[types[x]['roots'] for x in ws])
info = {
    'components': len(recs), 'word_types': n,
    'types_with_two_or_more_patterns': sum(len(t['golds']) > 1 for t in types.values()),
    'gold_boundaries_per_type_mean': sum(len(min(t['golds'])) for t in types.values()) / n,
    'root_groups': len(glist), 'largest_group': max(len(g) for g in glist),
    'random': {k: len(v) for k, v in rand.items()},
    'root': {k: len(v) for k, v in root.items()},
    'root_overlap_random_test_train': len(roots_of(rand['test']) & roots_of(rand['train'])) / len(roots_of(rand['test'])),
    'root_overlap_root_test_train': len(roots_of(root['test']) & roots_of(root['train'])) / max(1, len(roots_of(root['test']))),
    'test_words_with_root_seen_in_train_random': sum(bool(types[x]['roots'] & roots_of(rand['train'])) for x in rand['test']) / len(rand['test']),
}
# segments per type and length
lens = Counter(len(min(t['golds'])) + 1 for t in types.values())
info['segments_per_type_hist'] = dict(sorted(lens.items()))
info['chars_per_type_mean'] = sum(len(x) for x in words) / n
json.dump({'random': rand, 'root': root}, open(str(ROOT / 'benchmark' / 'splits.json'), 'w', encoding='utf8'), ensure_ascii=False)
json.dump({w_: sorted(map(list, t['golds'])) for w_, t in types.items()}, open(str(ROOT / 'benchmark' / 'gold.json'), 'w', encoding='utf8'), ensure_ascii=False)
json.dump(info, open(RESULTS + '/split_info.json', 'w'), indent=1)

def seg_str(word, b):
    cuts = [0] + list(b) + [len(word)]
    return '/'.join(word[cuts[i]:cuts[i+1]] for i in range(len(cuts) - 1))
for name, sp in (('random', rand), ('root', root)):
    for part, ws in sp.items():
        with open(str(ROOT / 'benchmark') + f'/{name}_{part}.tsv', 'w', encoding='utf8') as fh:
            for word in ws:
                for g in sorted(types[word]['golds']):
                    fh.write(f'{word}\t{seg_str(word, g)}\n')
print(json.dumps(info, indent=1))
