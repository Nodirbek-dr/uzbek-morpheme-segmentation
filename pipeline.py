#!/usr/bin/env python3
"""Reference implementation of Algorithm 1 and of the measures of the paper

    Morpheme Segmentation Data for Uzbek: A Reproducible Processing Algorithm
    and Statistical Validation of Segmentation Consistency

Input
-----
base.json     the base array of the electronic edition: a JSON list whose items
              are either a segmented string ("root/affix/affix") or a pair
              [string, note]
affixes.json  the affix appendix: a JSON list of rows; the first field of a row
              (or the field named "form") is the affix form, written with a
              trailing hyphen for prefixes ("be-") and a leading hyphen for
              suffixes ("-la", "-(i)b", "-gan (-kan, -qan)")

Both arrays are the two JavaScript arrays of the electronic edition exported
as JSON (step 1 of Algorithm 1, extraction).

Output (written to --out, default ./output)
------
entries.csv       one row per entry: id, surface string, segmented string, note type
segments.csv      one row per minimal segment: entry id, component index,
                  segment index, segment, label
notes.csv         entry id, note, note type (underlying form / comment)
index.csv         prefix index: string, boundary pattern, supporting entry ids
results.json      every number reported in Tables 1-3 and in the text
results.md        the same numbers as a readable report

Run
---
    python pipeline.py base.json affixes.json --out output
    python pipeline.py base.json affixes.json --out output --figure fig1.png

Only the standard library and NumPy are required; Matplotlib is needed for the
figure and SciPy, if installed, is used for Fisher's exact test (a built-in
hypergeometric implementation is used otherwise).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# step 1: extraction
# ----------------------------------------------------------------------------

def load_base(path):
    """Return a list of (segmented_string, note_or_None)."""
    with open(path, encoding="utf8") as fh:
        raw = json.load(fh)
    entries = []
    for item in raw:
        if isinstance(item, str):
            entries.append((item.strip(), None))
        elif isinstance(item, (list, tuple)) and item:
            entries.append((str(item[0]).strip(), (str(item[1]).strip() if len(item) > 1 and item[1] is not None else None)))
        elif isinstance(item, dict):
            word = item.get("word") or item.get("entry") or item.get("w") or next(iter(item.values()))
            note = item.get("note") or item.get("n")
            entries.append((str(word).strip(), (str(note).strip() if note else None)))
        else:
            raise ValueError(f"unexpected base item: {item!r}")
    return entries


def load_affix_forms(path):
    """Return the list of affix forms (one per appendix row, duplicates kept)."""
    with open(path, encoding="utf8") as fh:
        raw = json.load(fh)
    forms = []
    for row in raw:
        if isinstance(row, str):
            forms.append(row.strip())
        elif isinstance(row, (list, tuple)):
            forms.append(str(row[0]).strip())
        elif isinstance(row, dict):
            form = row.get("form") or row.get("affix") or next(iter(row.values()))
            forms.append(str(form).strip())
        else:
            raise ValueError(f"unexpected appendix row: {row!r}")
    return forms


# ----------------------------------------------------------------------------
# step 2: normalization
# ----------------------------------------------------------------------------

MODIFIER = "‘"      # letter modifier of o' and g'
GLOTTAL = "’"       # glottal-stop mark


def normalize(s):
    """Map straight apostrophes (U+0027) to U+2018 after o/g and to U+2019 elsewhere.
    No other character-level change is made."""
    out = []
    for i, ch in enumerate(s):
        if ch == "'":
            prev = s[i - 1].lower() if i > 0 else ""
            out.append(MODIFIER if prev in ("o", "g") else GLOTTAL)
        else:
            out.append(ch)
    return "".join(out)


# ----------------------------------------------------------------------------
# step 3: tokenization
# ----------------------------------------------------------------------------

def tokenize(segmented):
    """Return a list of components; each component is a list of minimal segments.
    Parentheses are removed, hyphens separate components, slashes separate segments."""
    s = segmented.replace("(", "").replace(")", "")
    components = []
    for comp in s.split("-"):
        comp = comp.strip()
        if not comp:
            continue
        segs = [x for x in comp.split("/") if x != ""]
        if segs:
            components.append(segs)
    return components


def surface(component):
    return "".join(component)


def boundary_pattern(component):
    """Vector of cut positions (|s1|, |s1 s2|, ..., |s1 ... s_{n-1}|)."""
    pos, out = 0, []
    for seg in component[:-1]:
        pos += len(seg)
        out.append(pos)
    return tuple(out)


# ----------------------------------------------------------------------------
# step 4: typing (prefix set P, suffix variant set S, allomorph set A, root set R)
# ----------------------------------------------------------------------------

def expand_suffix_variants(form):
    """'-(i)b' -> {'ib', 'b'};  '-gan (-kan, -qan)' -> {'gan', 'kan', 'qan'}."""
    form = form.strip()
    main, alts = form, []
    m = re.match(r"^(.*?)\s+\((.*)\)\s*$", form)
    if m:
        main, alt_text = m.group(1), m.group(2)
        alts = [a.strip().strip("-").strip() for a in alt_text.split(",") if a.strip()]
    main = main.strip().lstrip("-").strip()
    variants = {""}
    # expand optional groups written inside the form, e.g. (i)b
    for piece in re.split(r"(\([^)]*\))", main):
        if not piece:
            continue
        if piece.startswith("(") and piece.endswith(")"):
            inner = piece[1:-1]
            variants = {v + inner for v in variants} | set(variants)
        else:
            variants = {v + piece for v in variants}
    variants = {v for v in variants if v}
    variants |= {a for a in alts if a}
    return variants


def build_affix_sets(forms):
    prefix_forms = sorted({f.rstrip("-").strip() for f in forms if f.endswith("-")})
    suffix_forms_raw = sorted({f for f in forms if not f.endswith("-")})
    suffix_variants = set()
    for f in suffix_forms_raw:
        suffix_variants |= expand_suffix_variants(f)
    return set(prefix_forms), suffix_variants, {
        "appendix_rows": len(forms),
        "distinct_forms": len(set(forms)),
        "prefix_forms": len(prefix_forms),
        "suffix_forms": len(suffix_forms_raw),
        "suffix_variant_strings": len(suffix_variants),
    }


# ----------------------------------------------------------------------------
# step 5: note typing
# ----------------------------------------------------------------------------

def note_type(note):
    if note is None:
        return None
    return "underlying" if (" " not in note.strip() and "+" in note) else "comment"


# ----------------------------------------------------------------------------
# the full algorithm
# ----------------------------------------------------------------------------

def run(entries, affix_forms):
    P, S, affix_info = build_affix_sets(affix_forms)

    # extraction + normalization + tokenization
    parsed = []
    apostrophes_mapped = 0
    for eid, (seg_string, note) in enumerate(entries):
        apostrophes_mapped += seg_string.count("'")
        norm = normalize(seg_string)
        comps = tokenize(norm)
        parsed.append({
            "id": eid,
            "segmented": norm,
            "surface": "-".join(surface(c) for c in comps),
            "components": comps,
            "note": note,
            "note_type": note_type(note),
            "hyphenated": len(comps) > 1,
            "parentheses": "(" in seg_string,
        })

    # first pass: prefix / root identification, root set R
    R = set()
    for e in parsed:
        e["roles"] = []
        for comp in e["components"]:
            if len(comp) > 1 and comp[0] in P:
                prefix, root, post = comp[0], comp[1], comp[2:]
            else:
                prefix, root, post = None, comp[0], comp[1:]
            R.add(root)
            e["roles"].append((prefix, root, post))

    # allomorph set A from the underlying-form notes (segment-by-segment alignment)
    A = set()
    alternations = Counter()
    aligned_notes = 0
    for e in parsed:
        if e["note_type"] != "underlying" or len(e["components"]) != 1:
            continue
        under = normalize(e["note"]).split("+")
        surf = e["components"][0]
        if len(under) != len(surf):
            continue
        aligned_notes += 1
        for s_seg, u_seg in zip(surf, under):
            if s_seg != u_seg:
                alternations[(u_seg, s_seg)] += 1
                if u_seg in S and s_seg not in S:
                    A.add(s_seg)

    # second pass: typing of post-root segments
    labels = []          # (eid, comp_idx, seg_idx, segment, label)
    post_root_tokens = Counter()
    class_of_form = {}
    for e in parsed:
        for ci, (comp, (prefix, root, post)) in enumerate(zip(e["components"], e["roles"])):
            si = 0
            if prefix is not None:
                labels.append((e["id"], ci, si, prefix, "prefix")); si += 1
            labels.append((e["id"], ci, si, root, "root")); si += 1
            for seg in post:
                if seg in S:
                    lab = "affix"
                elif seg in A:
                    lab = "noted allomorph"
                elif seg in R:
                    lab = "compound constituent"
                else:
                    lab = "other"
                labels.append((e["id"], ci, si, seg, lab)); si += 1
                post_root_tokens[seg] += 1
                class_of_form[seg] = lab

    # step 6: indexing of proper initial portions
    index = defaultdict(lambda: defaultdict(set))   # string -> pattern -> {entry ids}
    for e in parsed:
        for comp in e["components"]:
            n = len(comp)
            for k in range(1, n):
                portion = comp[:k]
                index[surface(portion)][boundary_pattern(portion)].add(e["id"])

    return {
        "parsed": parsed, "labels": labels, "index": index,
        "P": P, "S": S, "A": A, "R": R,
        "affix_info": affix_info, "alternations": alternations,
        "aligned_notes": aligned_notes, "apostrophes_mapped": apostrophes_mapped,
        "post_root_tokens": post_root_tokens, "class_of_form": class_of_form,
    }


# ----------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------

def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def chi2_2x2(a, b, c, d):
    """Pearson chi-square (df = 1) on [[a, b], [c, d]]; returns (chi2, p, min expected)."""
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    if n == 0 or 0 in (r1, r2, c1, c2):
        return (float("nan"), float("nan"), 0.0)
    exp = [r1 * c1 / n, r1 * c2 / n, r2 * c1 / n, r2 * c2 / n]
    chi = sum((o - e) ** 2 / e for o, e in zip((a, b, c, d), exp))
    p = math.erfc(math.sqrt(chi / 2))
    return (chi, p, min(exp))


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test p-value for [[a, b], [c, d]]."""
    try:
        from scipy.stats import fisher_exact as _fe
        return float(_fe([[a, b], [c, d]])[1])
    except Exception:
        pass
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def logp(x):
        return (math.lgamma(r1 + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
                + math.lgamma(n - r1 + 1) - math.lgamma(c1 - x + 1) - math.lgamma(n - r1 - c1 + x + 1)
                - (math.lgamma(n + 1) - math.lgamma(c1 + 1) - math.lgamma(n - c1 + 1)))
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    p_obs = logp(a)
    total = 0.0
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp <= p_obs + 1e-9:
            total += math.exp(lp)
    return min(1.0, total)


def loglog_fit(freqs):
    """Least-squares fit log f = alpha + beta log r over ranks with f >= 2."""
    import numpy as np
    f = np.array(sorted(freqs, reverse=True), dtype=float)
    r = np.arange(1, len(f) + 1, dtype=float)
    keep = f >= 2
    x, y = np.log(r[keep]), np.log(f[keep])
    beta, alpha = np.polyfit(x, y, 1)
    yhat = alpha + beta * x
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(alpha), float(beta), float(r2), int(keep.sum())


# ----------------------------------------------------------------------------
# measures
# ----------------------------------------------------------------------------

PAIRS = [("la", "n"), ("la", "sh"), ("la", "t"), ("lash", "tir"), ("chi", "lik")]


def measures(state):
    parsed, labels, index = state["parsed"], state["labels"], state["index"]
    S, R = state["S"], state["R"]
    res = {}

    # ---- profile of the base (Table 1) ----
    N = len(parsed)
    surfaces = Counter(e["surface"] for e in parsed)
    homograph_strings = [s for s, c in surfaces.items() if c >= 2]
    seg_strings = Counter(e["segmented"] for e in parsed)
    exact_duplicates = sum(c - 1 for c in seg_strings.values() if c > 1)
    diff_seg = 0
    for s in homograph_strings:
        segs = {e["segmented"] for e in parsed if e["surface"] == s}
        if len(segs) > 1:
            diff_seg += 1
    seg_counts = [sum(len(c) for c in e["components"]) for e in parsed]
    hist = Counter(seg_counts)
    notes = [e for e in parsed if e["note"] is not None]
    prefixed = sum(1 for e in parsed if e["roles"] and e["roles"][0][0] is not None)
    H = set(surfaces)
    roots = Counter(root for e in parsed for (_, root, _) in e["roles"])
    res["table1"] = {
        "entries": N,
        "distinct_word_strings": len(surfaces),
        "homograph_strings": len(homograph_strings),
        "homograph_entries": sum(surfaces[s] for s in homograph_strings),
        "homographs_with_different_segmentation": diff_seg,
        "exact_duplicate_entries": exact_duplicates,
        "entries_with_note": len(notes),
        "underlying_form_notes": sum(1 for e in notes if e["note_type"] == "underlying"),
        "comment_notes": sum(1 for e in notes if e["note_type"] == "comment"),
        "hyphenated_entries": sum(1 for e in parsed if e["hyphenated"]),
        "entries_with_parentheses": sum(1 for e in parsed if e["parentheses"]),
        "minimal_segments_total": sum(seg_counts),
        "segments_per_entry_mean": sum(seg_counts) / N,
        "segments_per_entry_median": sorted(seg_counts)[N // 2],
        "segments_per_entry_max": max(seg_counts),
        "segments_histogram": {str(k): v for k, v in sorted(hist.items())},
        "entries_with_one_segment": hist.get(1, 0),
        "entries_with_three_or_more": sum(v for k, v in hist.items() if k >= 3),
        "entries_with_listed_prefix": prefixed,
        "prefix_form_counts_top": Counter(e["roles"][0][0] for e in parsed if e["roles"][0][0]).most_common(6),
        "root_tokens": sum(roots.values()),
        "distinct_roots": len(roots),
        "hapax_roots": sum(1 for c in roots.values() if c == 1),
        "roots_not_headwords": sum(1 for r in roots if r not in H),
        "most_productive_roots": roots.most_common(6),
        "aligned_underlying_forms": state["aligned_notes"],
        "most_frequent_alternations": [(f"{u} > {s}", c) for (u, s), c in state["alternations"].most_common(5)],
        "straight_apostrophes_mapped": state["apostrophes_mapped"],
        "appendix": state["affix_info"],
        "all_prefix_forms_attested": all(p in {e["roles"][0][0] for e in parsed} for p in state["P"]),
    }

    # ---- affix inventory (Table 2) ----
    tokens = state["post_root_tokens"]
    cls = state["class_of_form"]
    by_class = defaultdict(lambda: [0, 0])
    for form, c in tokens.items():
        by_class[cls[form]][0] += 1
        by_class[cls[form]][1] += c
    total_tokens = sum(tokens.values())
    affix_forms = {f: c for f, c in tokens.items() if cls[f] == "affix"}
    ranked = sorted(affix_forms.values(), reverse=True)
    cum = [sum(ranked[:k]) / sum(ranked) for k in (10, 20, 50, 100)] if ranked else []
    alpha, beta, r2, nfit = loglog_fit(list(tokens.values()))
    res["table2"] = {
        "post_root_tokens": total_tokens,
        "distinct_forms": len(tokens),
        "classes": {k: {"forms": v[0], "tokens": v[1], "share": v[1] / total_tokens} for k, v in by_class.items()},
        "top_affix_forms": Counter(affix_forms).most_common(10),
        "coverage_top_10_20_50_100": cum,
        "affix_forms_occurring_once": sum(1 for c in affix_forms.values() if c == 1),
        "powerlaw_fit_all_post_root": {"alpha": alpha, "beta": beta, "R2": r2, "ranks_used": nfit},
        "examples_by_class": {k: [f for f, _ in Counter({f: c for f, c in tokens.items() if cls[f] == k}).most_common(5)] for k in by_class},
    }

    # ---- consistency and the stem-attestation test (Table 3) ----
    chains = defaultdict(Counter)           # chain string -> pattern -> count
    suffixed_components = 0
    comp_records = []                       # (eid, prefix, root, post)
    for e in parsed:
        for (prefix, root, post) in e["roles"]:
            if post:
                suffixed_components += 1
                chains["".join(post)][tuple(len(p) for p in post)] += 1
                comp_records.append((e["id"], prefix, root, post))
    variable = {c for c, pats in chains.items() if len(pats) >= 2}
    covered = sum(sum(chains[c].values()) for c in variable)
    big = [c for c, pats in chains.items() if sum(pats.values()) >= 10]
    big_variable = [c for c in big if c in variable]

    def attested(stem, eid):
        if stem in H:
            return True
        pats = index.get(stem)
        if not pats:
            return False
        return any(i != eid for ids in pats.values() for i in ids)

    table3 = []
    for x, y in PAIRS:
        fused_str = x + y
        split_n = split_att = fused_n = fused_att = 0
        for eid, prefix, root, post in comp_records:
            head = (prefix or "") + root
            matched = None
            for i, seg in enumerate(post):
                if seg == x and i + 1 < len(post) and post[i + 1] == y:
                    matched = ("split", head + "".join(post[:i]) + x); break
                if seg == fused_str:
                    matched = ("fused", head + "".join(post[:i]) + x); break
            if matched is None:
                continue
            kind, stem = matched
            att = attested(stem, eid)
            if kind == "split":
                split_n += 1; split_att += att
            else:
                fused_n += 1; fused_att += att
        a, b, c, d = split_att, split_n - split_att, fused_att, fused_n - fused_att
        chi, p, min_exp = chi2_2x2(a, b, c, d)
        table3.append({
            "pair": f"{x}/{y} against {fused_str}",
            "split_components": split_n, "split_attested": split_att,
            "split_rate": (split_att / split_n if split_n else None),
            "fused_components": fused_n, "fused_attested": fused_att,
            "fused_rate": (fused_att / fused_n if fused_n else None),
            "chi2": chi, "p": p, "min_expected": min_exp,
            "fisher_p": fisher_exact(a, b, c, d) if split_n and fused_n else None,
        })
    res["table3"] = {
        "suffixed_components": suffixed_components,
        "distinct_suffix_chains": len(chains),
        "variable_chains": len(variable),
        "share_components_in_variable_chains": covered / suffixed_components if suffixed_components else None,
        "chains_with_10_or_more": len(big),
        "variable_among_them": len(big_variable),
        "chain_level_examples": {c: dict((("/".join(map(str, pat))), n) for pat, n in chains[c].items())
                                 for c in ("chilik", "lanmoq", "lashmoq", "lanish", "lashtirish") if c in chains},
        "rows": table3,
    }

    # ---- internal evidence beyond the headword list (Sect. 3.4) ----
    own_pattern = {}
    for e in parsed:
        if len(e["components"]) == 1:
            own_pattern.setdefault(e["surface"], set()).add(boundary_pattern(e["components"][0]))
    in_index = [h for h in H if h in index]
    agree = disagree = ties = 0
    for h in in_index:
        pats = index[h]
        counts = {pat: len(ids) for pat, ids in pats.items()}
        best = max(counts.values())
        winners = [pat for pat, c in counts.items() if c == best]
        if len(winners) > 1:
            ties += 1
        elif winners[0] in own_pattern.get(h, set()):
            agree += 1
        else:
            disagree += 1
    latent = [s for s in index if s not in H]
    with_boundary = sum(1 for s in latent if any(pat for pat in index[s]))
    support = {s: len(set().union(*index[s].values())) for s in latent}
    single = sum(1 for s in latent if support[s] == 1)
    multi = [s for s in latent if support[s] >= 2]
    unanimous = conflicts_majority = conflicts_tie = 0
    for s in multi:
        counts = {pat: len(ids) for pat, ids in index[s].items()}
        if len(counts) == 1:
            unanimous += 1
        else:
            best = max(counts.values())
            if sum(1 for c in counts.values() if c == best) > 1:
                conflicts_tie += 1
            else:
                conflicts_majority += 1
    lo, hi = wilson(unanimous, len(multi)) if multi else (None, None)
    lo2, hi2 = wilson(agree, agree + disagree) if (agree + disagree) else (None, None)
    res["section34"] = {
        "index_distinct_strings": len(index),
        "headwords_inside_other_entries": len(in_index),
        "headwords_share": len(in_index) / len(H),
        "headword_agreement": {"agree": agree, "disagree": disagree, "ties": ties,
                               "rate": (agree / (agree + disagree) if (agree + disagree) else None),
                               "wilson95": (lo2, hi2)},
        "latent_strings": len(latent),
        "latent_with_boundary": with_boundary,
        "latent_single_roots": len(latent) - with_boundary,
        "support_one": single,
        "support_two_or_more": len(multi),
        "unanimous": unanimous, "unanimous_rate": (unanimous / len(multi) if multi else None),
        "unanimous_wilson95": (lo, hi),
        "conflicts_resolved_by_majority": conflicts_majority,
        "ties": conflicts_tie,
        "analyzable_strings": len(H) + len(latent),
        "coverage_gain": len(latent) / len(H),
    }
    return res


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------

def write_tables(state, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "entries.csv"), "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh); w.writerow(["id", "surface", "segmented", "note_type"])
        for e in state["parsed"]:
            w.writerow([e["id"], e["surface"], e["segmented"], e["note_type"] or ""])
    with open(os.path.join(out, "segments.csv"), "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh); w.writerow(["entry_id", "component", "position", "segment", "label"])
        w.writerows(state["labels"])
    with open(os.path.join(out, "notes.csv"), "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh); w.writerow(["entry_id", "note", "note_type"])
        for e in state["parsed"]:
            if e["note"] is not None:
                w.writerow([e["id"], e["note"], e["note_type"]])
    with open(os.path.join(out, "index.csv"), "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh); w.writerow(["string", "boundary_pattern", "entry_ids"])
        for s in sorted(state["index"]):
            for pat, ids in state["index"][s].items():
                w.writerow([s, "/".join(map(str, pat)), " ".join(map(str, sorted(ids)))])


def pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def write_report(res, out):
    t1, t2, t3, s4 = res["table1"], res["table2"], res["table3"], res["section34"]
    lines = ["# Results", "", "## Table 1. Profile of the base", ""]
    for k, v in t1.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Table 2. Classes of post-root segments", "",
              f"- post-root tokens: {t2['post_root_tokens']}; distinct forms: {t2['distinct_forms']}"]
    for k, v in t2["classes"].items():
        lines.append(f"- {k}: forms {v['forms']}, tokens {v['tokens']}, share {pct(v['share'], 2)}; examples {t2['examples_by_class'][k]}")
    lines.append(f"- ten most frequent affix forms: {t2['top_affix_forms']}")
    lines.append(f"- coverage of top 10/20/50/100 affix forms: {[pct(c) for c in t2['coverage_top_10_20_50_100']]}")
    lines.append(f"- affix forms occurring once: {t2['affix_forms_occurring_once']}")
    fit = t2["powerlaw_fit_all_post_root"]
    lines.append(f"- power-law fit over all post-root forms (ranks with f >= 2, n = {fit['ranks_used']}): slope {fit['beta']:.2f}, R2 {fit['R2']:.2f}")
    lines += ["", "## Table 3. Stem-attestation test", "",
              f"- suffixed components: {t3['suffixed_components']}; distinct suffix chains: {t3['distinct_suffix_chains']}; "
              f"variable chains: {t3['variable_chains']} covering {pct(t3['share_components_in_variable_chains'])} of suffixed components; "
              f"chains with >= 10 components: {t3['chains_with_10_or_more']}, variable among them: {t3['variable_among_them']}",
              f"- chain-level counts: {t3['chain_level_examples']}", "",
              "| pair | split n | split attested | fused n | fused attested | chi2 | p | Fisher p |", "|---|---|---|---|---|---|---|---|"]
    for r in t3["rows"]:
        lines.append(f"| {r['pair']} | {r['split_components']} | {pct(r['split_rate'])} | {r['fused_components']} | {pct(r['fused_rate'])} | "
                     f"{r['chi2']:.2f} | {r['p']:.2e} | {r['fisher_p'] if r['fisher_p'] is None else format(r['fisher_p'], '.2e')} |")
    lines += ["", "## Section 3.4. Internal evidence", ""]
    for k, v in s4.items():
        lines.append(f"- {k}: {v}")
    with open(os.path.join(out, "results.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(lines) + "\n")


def make_figure(res, state, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    hist = res["table1"]["segments_histogram"]
    tokens = state["post_root_tokens"]; cls = state["class_of_form"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ks = sorted(int(k) for k in hist)
    ax1.bar(ks, [hist[str(k)] for k in ks], color="#2f3e5f")
    for k in ks:
        ax1.text(k, hist[str(k)], f"{hist[str(k)]:,}", ha="center", va="bottom", fontsize=7)
    ax1.set_xlabel("Minimal segments per entry"); ax1.set_ylabel("Entries"); ax1.set_title("(a) Segments per entry", fontsize=9)
    allf = np.array(sorted(tokens.values(), reverse=True), dtype=float)
    aff = np.array(sorted((c for f, c in tokens.items() if cls[f] == "affix"), reverse=True), dtype=float)
    ax2.loglog(np.arange(1, len(allf) + 1), allf, ".", color="#9fb3d9", ms=3, label=f"all post-root segments (n = {len(allf):,})")
    ax2.loglog(np.arange(1, len(aff) + 1), aff, ".", color="#2f3e5f", ms=3, label=f"affix forms in the table (n = {len(aff)})")
    fit = res["table2"]["powerlaw_fit_all_post_root"]
    r = np.arange(1, len(allf) + 1)
    ax2.loglog(r, np.exp(fit["alpha"]) * r ** fit["beta"], "-", color="#c0504d", lw=1, label=f"power-law fit to all segments, slope {fit['beta']:.2f}")
    ax2.set_xlabel("Rank"); ax2.set_ylabel("Frequency"); ax2.set_title("(b) Rank-frequency of post-root segments", fontsize=9)
    ax2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=300)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"); ap.add_argument("affixes")
    ap.add_argument("--out", default="output"); ap.add_argument("--figure", default=None)
    args = ap.parse_args(argv)
    entries = load_base(args.base)
    forms = load_affix_forms(args.affixes)
    state = run(entries, forms)
    res = measures(state)
    write_tables(state, args.out)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    write_report(res, args.out)
    if args.figure:
        make_figure(res, state, args.figure)
    print(open(os.path.join(args.out, "results.md"), encoding="utf8").read())


if __name__ == "__main__":
    main()
