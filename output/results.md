# Results

## Table 1. Profile of the base

- entries: 43706
- distinct_word_strings: 43612
- homograph_strings: 90
- homograph_entries: 184
- homographs_with_different_segmentation: 88
- exact_duplicate_entries: 5
- entries_with_note: 2249
- underlying_form_notes: 2009
- comment_notes: 240
- hyphenated_entries: 2042
- entries_with_parentheses: 141
- minimal_segments_total: 125935
- segments_per_entry_mean: 2.881412163089736
- segments_per_entry_median: 3
- segments_per_entry_max: 8
- segments_histogram: {'1': 361, '2': 17453, '3': 15359, '4': 8500, '5': 1652, '6': 345, '7': 27, '8': 9}
- entries_with_one_segment: 361
- entries_with_three_or_more: 25892
- entries_with_listed_prefix: 3489
- prefix_form_counts_top: [('be', 502), ('ser', 231), ('no', 188), ('radio', 179), ('avto', 165), ('foto', 141)]
- prefix_tokens: 3622
- root_tokens: 45749
- distinct_roots: 11496
- hapax_roots: 5116
- roots_not_headwords: 11104
- most_productive_roots: [('oq', 156), ('qora', 154), ('suv', 120), ('kam', 114), ('graf', 112), ('gul', 109)]
- aligned_underlying_forms: 1125
- most_frequent_alternations: [('la > lo', 536), ('illa > illo', 113), ('ber > ver', 36), ('a > o', 32), ('q > g‘', 21)]
- straight_apostrophes_mapped: 9
- appendix: {'appendix_rows': 590, 'distinct_forms': 410, 'prefix_forms': 117, 'suffix_forms': 293, 'suffix_variant_strings': 364}
- all_prefix_forms_attested: True

## Table 2. Classes of post-root segments

- post-root tokens: 76564; distinct forms: 2332
- affix: forms 330, tokens 68464, share 89.42%; examples ['lik', 'moq', 'la', 'ish', 'li']
- compound constituent: forms 1664, tokens 6428, share 8.40%; examples ['xona', 'gul', 'noma', 'bosh', 'bop']
- noted allomorph: forms 37, tokens 1072, share 1.40%; examples ['lo', 'illo', 'o', 'ol', 'g‘']
- other: forms 301, tokens 600, share 0.78%; examples ['baxsh', 'zoda', 'garlik', 'taxlit', 'yi']
- ten most frequent affix forms: [('lik', 8501), ('moq', 6373), ('la', 4709), ('ish', 3893), ('li', 2946), ('lar', 2682), ('tir', 2614), ('t', 2293), ('sh', 2046), ('chi', 1767)]
- coverage of top 10/20/50/100 affix forms: ['55.2%', '71.5%', '88.0%', '95.5%']
- affix forms occurring once: 25
- power-law fit over all post-root forms (ranks with f >= 2, n = 1253): slope -1.45, R2 0.99

## Table 3. Stem-attestation test

- suffixed components: 42899; distinct suffix chains: 6318; variable chains: 240 covering 26.1% of suffixed components; chains with >= 10 components: 454, variable among them: 82
- chain-level counts: {'chilik': {'3/3': 718, '6': 176}, 'lanmoq': {'2/1/3': 542, '3/3': 317, '1/1/1/3': 1}, 'lashmoq': {'4/3': 276, '2/2/3': 321, '1/1/2/3': 1}, 'lanish': {'3/3': 215, '2/1/3': 342, '1/1/1/3': 1}, 'lashtirish': {'2/2/3/3': 66, '7/3': 4, '4/3/3': 120}}

| pair | split n | split attested | fused n | fused attested | chi2 | p | Fisher p |
|---|---|---|---|---|---|---|---|
| la/n against lan | 1273 | 90.9% | 811 | 2.8% | 1563.77 | 0.00e+00 | 0.00e+00 |
| la/sh against lash | 1148 | 85.5% | 867 | 2.4% | 1365.04 | 8.32e-299 | 0.00e+00 |
| la/t against lat | 801 | 95.1% | 24 | 0.0% | 298.98 | 5.49e-67 | 1.38e-29 |
| lash/tir against lashtir | 306 | 81.4% | 9 | 0.0% | 34.95 | 3.38e-09 | 4.94e-07 |
| chi/lik against chilik | 718 | 4.6% | 180 | 3.9% | 0.17 | 6.81e-01 | 8.40e-01 |

## Section 3.4. Internal evidence

- index_distinct_strings: 32956
- headwords_inside_other_entries: 216
- headwords_share: 0.004952765293955792
- headword_agreement: {'agree': 183, 'disagree': 32, 'ties': 1, 'rate': 0.8511627906976744, 'wilson95': (0.7974405282228281, 0.8925566997791357)}
- latent_strings: 32740
- latent_with_boundary: 22154
- latent_single_roots: 10586
- support_one: 19909
- support_two_or_more: 12831
- unanimous: 12705
- unanimous_rate: 0.9901800327332242
- unanimous_wilson95: (0.9883210806238126, 0.9917455637752686)
- conflicts_resolved_by_majority: 91
- ties: 35
- analyzable_strings: 76352
- coverage_gain: 0.7507108135375584
