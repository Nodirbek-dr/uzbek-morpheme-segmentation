#!/usr/bin/env python3
"""Recompute the last column of Table 3 from the published counts.

Each row gives the number of components with the split reading and with the
fused reading and the share of them whose intermediate stem is attested
elsewhere in the base. The attested counts are recovered from the rounded
percentages (for these group sizes every percentage determines one integer),
and Pearson's chi-square test of independence (df = 1) and Fisher's exact test
are run on the 2 x 2 table (reading x attestation). Wilson 95% intervals for the
two validation rates of Sect. 3.4 are printed as well.

    python stats_from_table3.py
"""
import math

from pipeline import chi2_2x2, fisher_exact, wilson

TABLE3 = [
    # pair, split n, split attested %, fused n, fused attested %
    ("la/n against lan", 1273, 90.9, 811, 2.5),
    ("la/sh against lash", 1148, 85.2, 867, 2.1),
    ("la/t against lat", 801, 95.1, 24, 0.0),
    ("lash/tir against lashtir", 306, 81.0, 9, 0.0),
    ("chi/lik against chilik", 718, 4.6, 180, 3.9),
]


def recover_count(n, pct):
    """The integer k with round(100 k / n, 1) == pct; raises if it is not unique."""
    hits = [k for k in range(n + 1) if round(100 * k / n, 1) == pct]
    if len(hits) != 1:
        raise ValueError(f"percentage {pct} of {n} does not identify one count: {hits}")
    return hits[0]


def main():
    print(f"{'pair':26} {'split':>12} {'fused':>12} {'chi2':>9} {'p':>10} {'Fisher p':>10}")
    for pair, ns, ps, nf, pf in TABLE3:
        a = recover_count(ns, ps); b = ns - a
        c = recover_count(nf, pf); d = nf - c
        chi, p, min_exp = chi2_2x2(a, b, c, d)
        fp = fisher_exact(a, b, c, d)
        print(f"{pair:26} {a:>5}/{ns:<6} {c:>5}/{nf:<6} {chi:9.2f} {p:10.2e} {fp:10.2e}"
              + ("   (expected count < 5)" if min_exp < 5 else ""))
    for label, k, n in (("unanimous latent strings", 12703, 12831), ("headword agreement", 183, 215)):
        lo, hi = wilson(k, n)
        print(f"{label}: {k}/{n} = {100*k/n:.1f}%  95% Wilson CI {100*lo:.1f}-{100*hi:.1f}%")


if __name__ == "__main__":
    main()
