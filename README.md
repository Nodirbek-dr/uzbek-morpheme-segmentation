# Morpheme segmentation data for Uzbek: processing algorithm and statistics

Code and derived tables for the paper

> Khabibillayev N., Rahmanova G., Kholova M., Khabibillayeva G., Ibragimova U.
> *Morpheme Segmentation Data for Uzbek: A Reproducible Processing Algorithm and
> Statistical Validation of Segmentation Consistency.* MIP: Computing-V 2026.

The source is the electronic edition of the dictionary of Uzbek word structure
(https://soztarkibi.uz/), whose 43,706 entries are divided into roots and
affixes by explicit boundary marks. `pipeline.py` implements Algorithm 1 of the
paper (extraction, normalization, tokenization, segment typing, note typing,
prefix indexing) and computes every number reported in Tables 1-3, Fig. 1 and
the text.

## Contents

| File | Purpose |
|---|---|
| `pipeline.py` | Algorithm 1, the measures of Sect. 2.3, the statistical tests, the output tables and Fig. 1 |
| `stats_from_table3.py` | Recomputes the chi-square and Fisher tests of Table 3 from the published counts alone, without the data |
| `tests/sample_base.json`, `tests/sample_affixes.json` | A small synthetic sample in the input format, for a smoke test |
| `data/` | Input arrays exported from the electronic edition (see `data/README.md`) |
| `output/` | Derived tables written by the pipeline (`entries.csv`, `segments.csv`, `notes.csv`, `index.csv`, `results.json`, `results.md`) |

## Input format

The electronic edition stores the base as a JavaScript array of 43,706
elements and the affix appendix as a second array of 590 rows. Export both as
JSON:

* `data/base.json`: a list whose items are either a segmented string
  (`"root/affix/affix"`) or a pair `["root/affix", "note"]`;
* `data/affixes.json`: a list of rows whose first field (or a field named
  `form`) is the affix form, written with a trailing hyphen for prefixes
  (`"be-"`) and a leading hyphen for suffixes (`"-la"`, `"-(i)b"`,
  `"-gan (-kan, -qan)"`).

## Run

```bash
pip install -r requirements.txt
python pipeline.py data/base.json data/affixes.json --out output --figure fig1.png
```

`output/results.md` lists the values in the order of the paper. The smoke test
runs on the synthetic sample:

```bash
python pipeline.py tests/sample_base.json tests/sample_affixes.json --out /tmp/smoke
```

## What the pipeline computes

* Table 1: entries, distinct word strings, homographs, duplicates, notes,
  hyphenated and parenthesized entries, segments per entry, prefixes, roots.
* Table 2: classes of post-root segments (affix form of the appendix, noted
  allomorph, compound constituent, other), cumulative coverage of the most
  frequent affix forms, least-squares power-law fit of the rank-frequency
  series over the ranks with frequency >= 2.
* Table 3: the stem-attestation test for the pairs la/n, la/sh, la/t, lash/tir
  and chi/lik with the entry under test excluded from the evidence
  (leave-one-out), the attestation rates of the split and the fused readings,
  Pearson's chi-square test of independence (df = 1) and Fisher's exact test.
* Sect. 3.4: the prefix index, headwords that occur inside other entries and
  their agreement with the majority internal pattern, latent strings, support,
  unanimity, majority and ties, coverage gain, with 95% Wilson score
  intervals for the two validation rates.

## Conventions implemented

* Straight apostrophes (U+0027) are mapped to U+2018 after `o`/`g` and to
  U+2019 elsewhere; no other character is changed.
* Parentheses are removed, hyphens separate components, slashes separate
  minimal segments.
* A component whose first segment is a prefix form of the appendix and has
  more than one segment is read as prefix + root; otherwise the first segment
  is the root.
* Post-root segments are labeled in this order of precedence: affix (a variant
  string of the appendix), noted allomorph (a surface segment that differs from
  an affix in an aligned underlying-form note), compound constituent (a
  segment that is a root elsewhere in the base), other.
* A note is an underlying form when it is a single token containing `+`,
  otherwise a comment.
* The prefix index maps every proper initial portion of a component to its
  boundary pattern and the identifier of its entry.

## License

Code: MIT (see `LICENSE`). The dictionary itself remains the work of its
compilers; the derived tables are published for verification of the paper.
