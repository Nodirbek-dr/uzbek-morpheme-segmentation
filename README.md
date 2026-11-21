# Uzbek morpheme segmentation: dictionary data, benchmark, and experiments

Code, data, and a morpheme segmentation benchmark for the chapter

> Khabibillayev N., Rahmanova G., Kholova M., Khabibillayeva G.
> *Morpheme Segmentation Data for Uzbek: An Open Benchmark, Segmentation
> Baselines, and the Morphological Alignment of Language Model Tokenizers.*
> In preparation.

The source is the electronic edition of the Dictionary of Uzbek Word Structure
(https://soztarkibi.uz/), whose 43,706 entries are divided into roots and
affixes by explicit boundary marks. The repository has two parts:

* `pipeline.py` processes the electronic edition (extraction, normalization,
  tokenization, segment typing, note typing, prefix indexing) and computes the
  statistics of the source base in `output/`, including the stem-attestation
  tests of Sect. 3.1 of the chapter. The table numbers inside
  `output/results.md` follow an earlier version of the manuscript.
* `benchmark/`, `experiments/`, `results/`, and `figures/` hold the benchmark
  splits built from the processed base, the segmentation and tokenizer
  experiments, their results, and Fig. 1 of the chapter. See
  `benchmark/README.md` for the splits, the systems, and the commands that
  reproduce every number of the chapter.

## Contents

| File | Purpose |
|---|---|
| `pipeline.py` | The processing algorithm, the statistics of the base, the stem-attestation tests, the output tables and `output/fig1.png` |
| `stats_from_table3.py` | Recomputes the chi-square and Fisher tests of the stem-attestation table from the published counts alone, without the data |
| `tests/sample_base.json`, `tests/sample_affixes.json` | A small synthetic sample in the input format, for a smoke test |
| `data/` | The two arrays of the electronic edition, exported from its HTML on 12 November 2026 (`base.json`, 43,706 items; `affixes.json`, 590 rows) |
| `output/` | Derived tables written by the pipeline (`entries.csv`, `segments.csv`, `notes.csv`, `index.csv`, `results.json`, `results.md`) and `fig1.png` (segments per entry; rank-frequency of post-root segments) |
| `benchmark/` | The benchmark: gold segmentations of 43,787 word types, the random and root-disjoint splits, and their documentation |
| `experiments/` | The segmentation systems (affix stripping, Morfessor, CRF, BiLSTM tagger) and the tokenizer audit |
| `results/` | Results and predictions of every experiment reported in the chapter |
| `figures/` | Fig. 1 of the chapter and the script that draws it |

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

`output/results.md` lists the statistics of the base. The smoke test runs on
the synthetic sample:

```bash
python pipeline.py tests/sample_base.json tests/sample_affixes.json --out /tmp/smoke
```

## Reproduction

`output/results.md` in this repository is the output of the command above on
the arrays in `data/`. It contains: 43,706 entries and 43,612 distinct strings; 125,935
minimal segments; 3,489 prefixed entries; 11,496 roots; 330 affix forms with
68,464 tokens (89.42%); the power-law slope -1.45 (R2 = 0.99); 6,318 suffix
chains, 240 of them variable (26.1% of suffixed components); the five rows of
the stem-attestation test with their chi-square and Fisher tests; 216
headwords inside other entries (183 of 215 in agreement); 32,740 latent
strings, 12,705 of 12,831 unanimous.

## What the pipeline computes

* Profile of the base: entries, distinct word strings, homographs, duplicates, notes,
  hyphenated and parenthesized entries, segments per entry, prefixes, roots.
* Affix inventory: classes of post-root segments (affix form of the appendix, noted
  allomorph, compound constituent, other), cumulative coverage of the most
  frequent affix forms, least-squares power-law fit of the rank-frequency
  series over the ranks with frequency >= 2.
* Consistency: the stem-attestation test for the pairs la/n, la/sh, la/t, lash/tir
  and chi/lik with the entry under test excluded from the evidence
  (leave-one-out), the attestation rates of the split and the fused readings,
  Pearson's chi-square test of independence (df = 1) and Fisher's exact test.
* Internal evidence: the prefix index, headwords that occur inside other entries and
  their agreement with the majority internal pattern, latent strings, support,
  unanimity, majority and ties, coverage gain, with 95% Wilson score
  intervals for the two validation rates.

## Conventions implemented

* Straight apostrophes (U+0027) are mapped to U+2018 after `o`/`g` and to
  U+2019 elsewhere; no other character is changed.
* Parentheses are removed, hyphens separate components, slashes separate
  minimal segments.
* Leading segments that are prefix forms of the appendix are labeled prefixes
  as long as at least one segment remains (a component may open with two
  prefixes); the first remaining segment is the root.
* Post-root segments are labeled in this order of precedence: affix (a variant
  string of the appendix), noted allomorph (a surface segment that differs from
  an affix in an aligned underlying-form note), compound constituent (a
  segment that is a root elsewhere in the base), other.
* A note is an underlying form when it is a single token containing `+`,
  otherwise a comment.
* The prefix index maps every proper initial portion of a component to its
  boundary pattern and the identifier of its entry.
* In the stem-attestation test a pair such as la/n against lan is counted
  where it opens the suffix chain; the intermediate stem is the root followed
  by the first piece (a listed prefix is not part of it), and it counts as
  attested when it is a headword or an initial portion of another entry,
  the entry under test excluded. Counts for the pair at any position are kept
  in `results.json` under `pairs_at_any_position`.
* The appendix row `-i, (-si)` carries a stray comma; the parser strips it, so
  `i` is read as an affix variant.

## License

Code: MIT (see `LICENSE`). The dictionary itself remains the work of its
compilers; the derived tables and the benchmark are published for research
and for verification of the chapter.
