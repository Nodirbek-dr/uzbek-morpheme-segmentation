"""Shared helpers: loading the electronic edition and building benchmark items.
Loading, normalization, tokenization and affix-set functions follow pipeline.py of
https://github.com/Nodirbek-dr/uzbek-morpheme-segmentation (Algorithm 1)."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import json, re
from collections import defaultdict

BASE = str(ROOT / 'data' / 'base.json')
AFF = str(ROOT / 'data' / 'affixes.json')
MODIFIER, GLOTTAL = '‘', '’'

def load_base(path=BASE):
    raw = json.load(open(path, encoding='utf8'))
    return [(x.strip(), None) if isinstance(x, str) else (str(x[0]).strip(), str(x[1]).strip()) for x in raw]

def load_affix_forms(path=AFF):
    return [str(r[0]).strip() for r in json.load(open(path, encoding='utf8'))]

def normalize(s):
    out = []
    for i, ch in enumerate(s):
        if ch == "'":
            prev = s[i-1].lower() if i > 0 else ''
            out.append(MODIFIER if prev in ('o', 'g') else GLOTTAL)
        else:
            out.append(ch)
    return ''.join(out)

def tokenize(segmented):
    s = segmented.replace('(', '').replace(')', '')
    comps = []
    for comp in s.split('-'):
        comp = comp.strip()
        if not comp:
            continue
        segs = [x for x in comp.split('/') if x != '']
        if segs:
            comps.append(segs)
    return comps

def boundary_pattern(comp):
    pos, out = 0, []
    for seg in comp[:-1]:
        pos += len(seg); out.append(pos)
    return tuple(out)

def expand_suffix_variants(form):
    variants = set()
    for alt in form.split('/'):
        alt = alt.strip(); alts = []
        m = re.match(r'^(.*?)\s*\(([^()]*)\)\s*$', alt)
        if m and re.search(r'(^|,)\s*-', m.group(2)):
            alt, alt_text = m.group(1), m.group(2)
            alts = [a.strip().strip('-').strip() for a in alt_text.split(',') if a.strip()]
        main = alt.strip().lstrip('-').strip().rstrip(',').strip()
        vs = {''}
        for piece in re.split(r'(\([^)]*\))', main):
            if not piece: continue
            if piece.startswith('(') and piece.endswith(')'):
                inner = piece[1:-1]; vs = {v + inner for v in vs} | set(vs)
            else:
                vs = {v + piece for v in vs}
        variants |= {v for v in vs if v}; variants |= {a for a in alts if a}
    return variants

def build_affix_sets(forms):
    P = {f.rstrip('-').strip() for f in forms if f.endswith('-')}
    S = set()
    for f in {f for f in forms if not f.endswith('-')}:
        S |= expand_suffix_variants(f)
    return P, S

def components():
    """One record per hyphen-separated component of every entry."""
    entries = load_base(); P, S = build_affix_sets(load_affix_forms())
    recs = []
    for eid, (seg, note) in enumerate(entries):
        for comp in tokenize(normalize(seg)):
            k = 0
            while k < len(comp) - 1 and comp[k] in P:
                k += 1
            recs.append({'eid': eid, 'segs': comp, 'surface': ''.join(comp),
                         'b': boundary_pattern(comp), 'root': comp[k], 'post': comp[k+1:]})
    return recs, P, S

# graphemes: o‘, g‘, sh, ch are single letters of the Uzbek Latin alphabet
DIGRAPHS = {'o‘': 'ö', 'g‘': 'ğ', 'sh': 'ş', 'ch': 'ç',
            'O‘': 'Ö', 'G‘': 'Ğ'}

def graphemes(w):
    """Return (mapped string, list of original char offsets of each grapheme start)."""
    out, starts, i = [], [], 0
    while i < len(w):
        two = w[i:i+2]
        if two in DIGRAPHS:
            out.append(DIGRAPHS[two]); starts.append(i); i += 2
        else:
            out.append(w[i]); starts.append(i); i += 1
    return ''.join(out), starts
