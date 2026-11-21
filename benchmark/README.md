# Uzbek morpheme segmentation benchmark

Benchmark, experiment code, and results for the chapter

> Khabibillayev N., Rahmanova G., Kholova M., Khabibillayeva G. *Morpheme Segmentation Data for Uzbek:
> An Open Benchmark, Segmentation Baselines, and the Morphological Alignment of Language Model Tokenizers.*
> Chapter in preparation for *Artificial Intelligence Adoption in Central Asia* (Springer, Intelligent Systems Reference Library).

The benchmark is derived from the electronic edition of the Dictionary of Uzbek Word Structure
(https://soztarkibi.uz/), whose arrays are stored in `data/` (exported 12 November 2026).

## Files

| Path | Content |
|---|---|
| `benchmark/{random,root}_{train,dev,test}.tsv` | one line per word type and gold analysis: `word<TAB>segmentation`, morphs separated by `/` |
| `benchmark/splits.json`, `benchmark/gold.json` | the same splits and all gold analyses in JSON |
| `experiments/` | benchmark construction, segmenters, tokenizer audit, error analysis, evaluation |
| `results/` | all scores (`FINAL_NUMBERS.json` collects the values reported in the chapter) and model predictions |
| `figures/` | Fig. 1 of the chapter and the script that draws it |

Word types: 43,787 (117 of them with two or more gold analyses; a prediction is scored against the closest one).

| Split | Train | Dev | Test | Test words whose first root occurs in training |
|---|---|---|---|---|
| random | 35,030 | 4,378 | 4,379 | 86.3% |
| root-disjoint | 35,030 | 4,416 | 4,341 | 0% |

The root-disjoint split assigns whole groups of word types that share a first root (the root that opens the word after any prefixes) to one part, so no test word shares its first root with a training word. Roots inside compounds are not used for grouping, because they would merge 57.8% of the vocabulary into one group. 1,716 root-disjoint test words (39.5%) therefore share a root with the training data through a compound, 1,328 of them because their first root occurs as a later compound part in training (`experiments/overlap.py`, `results/res_overlap.json`). These words are harder, not easier, for the supervised models; `experiments/overlap_subsets.py` scores every segmenter on both subsets (`results/res_overlap_subsets.json`).

## Segmentation results (test sets)

`python experiments/evaluate_predictions.py` recomputes these values from the saved predictions.

| System | Random P / R / F1 / Acc | Root-disjoint P / R / F1 / Acc |
|---|---|---|
| Affix stripping (rule-based) | 0.527 / 0.732 / 0.613 / 17.7 | 0.508 / 0.721 / 0.596 / 15.7 |
| Morfessor Baseline | 0.674 / 0.573 / 0.619 / 30.0 | 0.468 / 0.556 / 0.508 / 11.0 |
| Morfessor, all 43,787 unsegmented types | 0.773 / 0.537 / 0.634 / 34.9 | 0.762 / 0.534 / 0.628 / 34.9 |
| CRF | 0.947 / 0.926 / 0.936 / 80.0 | 0.918 / 0.828 / 0.871 / 63.8 |
| CRF + root lexicon | 0.954 / 0.931 / 0.942 / 81.5 | 0.941 / 0.836 / 0.885 / 67.2 |
| BiLSTM tagger (mean of three seeds) | 0.958 / 0.959 / 0.959 / 87.6 | 0.905 / 0.880 / 0.892 / 69.1 |

The CRF + root lexicon row uses the full training lexicon on the random split and the jackknifed lexicon on the root-disjoint split; the development sets preferred these variants (`results/res_crflex_*.json`, `results/res_crflexjk_*.json`). The transductive Morfessor row is in `results/res_morf_all.json`.

The BiLSTM tagger (`experiments/bilstm.py`, JAX) has 2 layers with 128 units per direction and dropout 0.3, chosen on the root-disjoint development set among three settings (`results/res_bilstm_grid.json`). Over three seeds (20261119 to 20261121), its test F1 ranged from 0.957 to 0.960 on the random split and from 0.889 to 0.897 on the root-disjoint split (`results/res_bilstm_main.json`). Paired bootstrap tests between the supervised segmenters are in `results/res_paired_bootstrap.json`.

P, R, and F1 refer to morpheme boundaries; Acc is word accuracy in %.

## Tokenizer audit (root-disjoint test set, 4,341 words)

| Tokenizer | P | R | F1 | Tokens per word | Words without a cut inside a morpheme | Cuts on a morpheme boundary |
|---|---|---|---|---|---|---|
| GPT-4 (cl100k_base) | 0.255 | 0.512 | 0.341 | 4.59 | 3.9% | 25.5% |
| GPT-4o (o200k_base) | 0.275 | 0.468 | 0.346 | 4.04 | 6.9% | 27.5% |
| Llama 2 (32k SentencePiece) | 0.248 | 0.539 | 0.340 | 4.89 | 2.7% | 24.8% |
| Gemma (256k SentencePiece) | 0.338 | 0.588 | 0.429 | 4.11 | 7.7% | 33.8% |
| Llama 3 (128k BPE) | 0.263 | 0.508 | 0.346 | 4.46 | 4.6% | 26.3% |
| Llama 4 (200k BPE) | 0.274 | 0.464 | 0.345 | 4.03 | 7.0% | 27.4% |
| Mistral Tekken (131k BPE) | 0.262 | 0.475 | 0.338 | 4.24 | 5.9% | 26.2% |

The gold segmentation has 2.79 morphemes per word. Results on all 43,787 word types, apostrophe variants (U+2018, U+02BB, U+0027), and the in-domain SentencePiece models are in `results/res_tok*.json`.

## Reproducing the experiments

```bash
pip install -r experiments/requirements.txt
python experiments/splits.py            # rebuilds the splits (identical output, seed 20261119)
python experiments/baselines.py         # affix stripping and Morfessor (seeded, about 6 min)
python experiments/crf.py root 0.0 0.3 test
python experiments/crf.py random 0.0 0.3 test
python experiments/crf_lex.py random    # CRF with the full root lexicon
python experiments/crf_lexjk.py root    # CRF with the jackknifed root lexicon
python experiments/bilstm_driver.py     # BiLSTM grid and seeds (CPU, a few hours)
python experiments/bilstm_summary.py
python experiments/tokaudit.py          # GPT-4, GPT-4o, Llama 2, Gemma and in-domain SentencePiece models
python experiments/tok2.py              # apostrophe variants and in-domain models without script splitting
python experiments/tok_new.py           # Llama 3, Llama 4, Mistral Tekken and the decomposition of cuts
python experiments/tok_new_variants.py  # apostrophe variants for all seven LLM tokenizers
python experiments/erran.py; python experiments/erran2.py
python experiments/morf_all.py         # Morfessor on all unsegmented word types
python experiments/overlap.py; python experiments/overlap_subsets.py
python experiments/paired_boot.py       # paired bootstrap tests between the supervised segmenters
python experiments/update_final.py     # collects the new results in results/FINAL_NUMBERS.json
python figures/make_fig1.py
```

The tokenizer audit needs seven tokenizer files, which are not redistributed here. Place them in `tokenizers/`:

* `tokenizers/tiktoken_rs/cl100k_base.tiktoken` and `o200k_base.tiktoken` (for example from the `assets` folder of
  https://github.com/zurawiki/tiktoken-rs); the scripts check them against the SHA-256 values in the tiktoken source;
* `tokenizers/llama2/tokenizer.model` (the Llama 2 SentencePiece model, 32,000 pieces);
* `tokenizers/gemma/tokenizer.model` (the Gemma SentencePiece model, 256,000 pieces, from https://github.com/google/gemma_pytorch);
* `tokenizers/llama3/tokenizer.model` and `tokenizers/llama4/tokenizer.model` (from `llama_models/llama3/` and
  `llama_models/llama4/` in the wheel `llama_models-0.3.0-py3-none-any.whl` on PyPI);
* `tokenizers/tekken/tekken_240911.json` (from `mistral_common/data/` in the wheel `mistral_common-1.13.1-py3-none-any.whl` on PyPI).

SHA-256 checksums of the files used in the chapter:

| File | SHA-256 |
|---|---|
| `cl100k_base.tiktoken` | `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7` |
| `o200k_base.tiktoken` | `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d` |
| `llama2/tokenizer.model` | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` |
| `gemma/tokenizer.model` | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` |
| `llama3/tokenizer.model` | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` |
| `llama4/tokenizer.model` | `d0bdbaf59b0762c8c807617e2d8ea51420eb1b1de266df2495be755c8e0ed6ed` |
| `tekken/tekken_240911.json` | `1948e2d48b0e7377f1bb5f1210f1ae5f984934e75713fc07e2452729b8365316` |

Software versions: Python 3.13, Morfessor 2.0.6, sklearn-crfsuite 0.5.0, JAX 0.12.1, Optax 0.2.9, SentencePiece 0.2.3, tiktoken 0.14.0.
