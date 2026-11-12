# Input data

The two arrays of the electronic edition of the dictionary of Uzbek word
structure (https://soztarkibi.uz/), exported as JSON from the page's script
(`const DATA = [...]` and `const AFFIX = [...]`) on 12 November 2026:

* `base.json`: the base array (43,706 items; a segmented string, or a pair of a
  string and a note);
* `affixes.json`: the affix appendix (590 rows; the first field is the affix
  form).

The pipeline does not modify these files. The derived tables in `output/` are
produced from them by `python pipeline.py data/base.json data/affixes.json --out output`.
