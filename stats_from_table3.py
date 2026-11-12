#!/usr/bin/env python3
"""Recompute the last column of Table 3 from the published counts.

Each row gives the number of components with the split reading and with the
fused reading and the number of them whose intermediate stem is attested
elsewhere in the base. Pearson's chi-square test of independence (df = 1) and Fisher's exact test
are run on the 2 x 2 table (reading x attestation). Wilson 95% intervals for the
two validation rates of Sect. 3.4 are printed as well.

    python stats_from_table3.py
"""
import math

from pipeline import chi2_2x2, fisher_exact, wilson

TABLE3 = [
    # pair, split components, split attested, fused components, fused attested (Table 3 of the paper)
    ("la/n against lan", 1273, 1157, 811, 23),
    ("la/sh against lash", 1148, 982, 867, 21),
    ("la/t against lat", 801, 762, 24, 0),
    ("lash/tir against lashtir", 306, 249, 9, 0),
    ("chi/lik against chilik", 718, 33, 180, 7),
]


def main():
    print(f"{'pair':26} {'split':>12} {'fused':>12} {'chi2':>9} {'p':>10} {'Fisher p':>10}")
    for pair, ns, a, nf, c in TABLE3:
        b, d = ns - a, nf - c
        chi, p, min_exp = chi2_2x2(a, b, c, d)
        fp = fisher_exact(a, b, c, d)
        print(f"{pair:26} {a:>5}/{ns:<6} {c:>5}/{nf:<6} {chi:9.2f} {p:10.2e} {fp:10.2e}"
              + ("   (expected count < 5)" if min_exp < 5 else ""))
    for label, k, n in (("unanimous latent strings", 12705, 12831), ("headword agreement", 183, 215)):
        lo, hi = wilson(k, n)
        print(f"{label}: {k}/{n} = {100*k/n:.1f}%  95% Wilson CI {100*lo:.1f}-{100*hi:.1f}%")


if __name__ == "__main__":
    main()
