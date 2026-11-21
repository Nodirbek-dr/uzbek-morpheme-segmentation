"""Root overlap left by the root-disjoint split, which groups word types by their first root only.

1. How many root-disjoint test words have a first root that occurs as a later (compound) part in training, and vice versa.
2. What grouping by every root, compound parts included, would do: the size of the largest group.
"""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json
from collections import defaultdict, Counter
from common import components
from evalu import SPLITS
recs, P, S = components()
R = {r['root'] for r in recs}
first, later = defaultdict(set), defaultdict(set)
for r in recs:
    first[r['surface']].add(r['root'])
    for seg in r['post']:
        if seg in R and seg not in S: later[r['surface']].add(seg)
sp = SPLITS['root']; tr = set(sp['train'])
train_first = set().union(*[first[w] for w in tr]); train_later = set().union(*[later[w] for w in tr])
test = sp['test']
a = [w for w in test if first[w] & train_later]
b = [w for w in test if later[w] & train_first]
ab = [w for w in test if (first[w] & train_later) or (later[w] & train_first) or (later[w] & train_later)]
print('test', len(test), 'first root occurs as later compound part in train:', len(a), round(100*len(a)/len(test),1),
      '| later part is a first root in train:', len(b), round(100*len(b)/len(test),1), '| any root overlap:', len(ab), round(100*len(ab)/len(test),1))
# grouping by all roots (first roots and compound parts)
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry: parent[rx] = ry
words = set(first)
for w in words:
    for root in first[w] | later[w]: union(('w', w), ('r', root))
sizes = sorted(Counter(find(('w', w)) for w in words).values(), reverse=True)
print('grouping by all roots:', len(sizes), 'groups; largest', sizes[0], 'word types =', round(100*sizes[0]/len(words), 1), '% of', len(words))
json.dump({'n_test': len(test), 'first_as_later_in_train': len(a), 'later_as_first_in_train': len(b), 'any_overlap': len(ab),
           'all_roots_groups': len(sizes), 'all_roots_largest_group': sizes[0], 'n_types': len(words)}, open(RESULTS + '/res_overlap.json', 'w'), indent=1)
