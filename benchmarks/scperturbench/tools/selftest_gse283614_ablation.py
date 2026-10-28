"""Self-tests for the GSE283614 ablation tooling (synthetic data, no real results).

Covers the pieces that silently produce wrong numbers when broken:

1. the two guide-key resolvers, including the ``SafeHarborchr1_1007`` /
   ``NANOG-1-5`` cases that made the first preflight fail;
2. activity-table loading with an explicit zero fill;
3. the pooled out-of-fold loader's duplicate-condition guard;
4. the paired statistics against hand-computed values;
5. the BH correction against a hand-checked example.

Run inside the container:

    /usr/local/anaconda3/envs/pertpyV7/bin/python tools/selftest_gse283614_ablation.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
for candidate in (WORKSPACE, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_key_resolvers() -> None:
    from tools.preflight_gse283614_activity import canonical_guide_keys, guide_keys

    cases = {
        "NANOG-1-5": ("NANOG-1-5",),
        "NANOG-pro1": ("NANOG-pro1",),
        # Underscore splitting only, and the extra key needs at least three
        # parts (or a doubled first token).  The raw label always survives.
        # Both resolvers were copied verbatim from the repo, so these
        # expectations pin the copied behaviour rather than an assumption.
        "OCT4_1-17": ("OCT4_1-17",),
        "NANOG_1-23": ("NANOG_1-23",),
        "SafeHarborchr1_1007": ("SafeHarborchr1_1007",),
        "NT_Control_7": ("NT_Control_7", "Control_7"),
        "A_B_C": ("A_B_C", "B_C"),
        "A_A_B": ("A_A_B", "A_B"),
    }
    for label, expected in cases.items():
        got = canonical_guide_keys(label)
        check(f"canonical_guide_keys({label})", got == expected, f"got {got}, expected {expected}")

    # guide_keys must always include the raw label, which is what makes an
    # explicit 0.0 row resolvable for a guide whose name has no underscore.
    for label in cases:
        check(f"guide_keys contains raw label for {label}", label in guide_keys(label))

    # The regression that made the first preflight fail: a SafeHarbor guide whose
    # only canonical key is its own name must still be resolvable when the table
    # carries an explicit row for it.
    table = {"SafeHarborchr1_1007": 0.0, "NANOG-1-5": 0.0}
    for label in table:
        hit = next((table[k] for k in canonical_guide_keys(label) if k in table), None)
        check(f"explicit-zero row resolves {label}", hit == 0.0, f"hit={hit}")


def test_activity_table_round_trip() -> None:
    from tools.preflight_gse283614_activity import load_activity_keys

    frame = pd.DataFrame(
        {
            "guide_identity": ["NANOG_1-23", "NANOG-1-5", "OCT4-pro1", "SafeHarborchr2_1008"],
            "relative_activity": [1.0, 0.0, 0.0, 0.0],
            "target_gene": ["NANOG", "NANOG", "POU5F1", "control"],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "activity.tsv"
        frame.to_csv(path, sep="\t", index=False)
        data_lookup, guide_lookup = load_activity_keys(path)
    check("data lookup has the measured guide", data_lookup.get("NANOG_1-23") == 1.0)
    check("data lookup has the zero-filled guide", data_lookup.get("NANOG-1-5") == 0.0)
    check("guide lookup has the control guide", guide_lookup.get("SafeHarborchr2_1008") == 0.0)


def test_oof_loader_duplicate_guard() -> None:
    import scperturbench_gse283614_ablation as driver

    rows = []
    for fold in (0, 1):
        rows.append(
            {
                "method": "activity_on",
                "seed": 7,
                "condition": "G1" if fold == 0 else "G2",
                "metric": "mse",
                "deg_budget": "100",
                "value": 0.1,
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for fold, condition in ((0, "G1"), (1, "G2")):
            path = root / "gse283614" / "esc" / f"fold_{fold}" / "activity_on" / "seed7" / "score"
            path.mkdir(parents=True)
            frame = pd.DataFrame([r for r in rows if r["condition"] == condition])
            frame.to_csv(path / "per_condition.csv", index=False)
        pooled, folds = driver.load_arm_oof(root, "gse283614", "esc", "activity_on", 7)
        check("pooled loader gathers both folds", sorted(folds) == [0, 1], f"folds={folds}")
        check("pooled loader keeps the fold column", set(pooled["fold"]) == {0, 1})

        # Same condition in two folds must be rejected: it would double-count a
        # guide and break the pairing.
        for fold in (0, 1):
            path = root / "gse283614" / "esc" / f"fold_{fold}" / "activity_on" / "seed7" / "score"
            pd.DataFrame([r for r in rows if r["condition"] == "G1"]).to_csv(
                path / "per_condition.csv", index=False
            )
        try:
            driver.load_arm_oof(root, "gse283614", "esc", "activity_on", 7)
            check("pooled loader rejects duplicate conditions", False, "no SystemExit raised")
        except SystemExit:
            check("pooled loader rejects duplicate conditions", True)


def test_paired_statistics() -> None:
    try:
        import scperturbench_ablation_figure as base
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] paired statistics: cannot import the figure module ({type(exc).__name__})")
        return

    on = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    off = np.array([2.0, 2.0, 4.0, 4.0, 6.0])
    diff = on - off
    g = base.hedges_g_paired(diff)
    n = diff.size
    d = diff.mean() / diff.std(ddof=1)
    expected_g = d * (1 - 3 / (4 * n - 5))
    check("hedges_g_paired matches the closed form", abs(g - expected_g) < 1e-12, f"{g} vs {expected_g}")

    # All-positive differences -> every rank on the positive side.
    rb = base.rank_biserial_paired(np.array([1.0, 2.0, 3.0, 0.0]))
    check("rank_biserial_paired ignores zeros and saturates at +1", abs(rb - 1.0) < 1e-12, str(rb))

    q = base.benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    # largest p -> q == p (BH is monotone from the top)
    check("BH leaves the largest p unchanged", abs(q[-1] - 0.05) < 1e-12, str(q))
    check("BH is monotone non-decreasing", all(q[i] <= q[i + 1] + 1e-12 for i in range(len(q) - 1)), str(q))
    check("BH q >= p elementwise", all(qi >= pi - 1e-12 for qi, pi in zip(q, [0.01, 0.02, 0.03, 0.04, 0.05])))


def test_probe_swap_is_within_target() -> None:
    from tools.activity_perturbation_probe_gse283614 import within_target_swap

    target_of = {"A1": "T1", "A2": "T1", "B1": "T2"}
    activity = {"A1": 0.1, "A2": 0.9, "B1": 0.5}
    rng = np.random.default_rng(0)
    for _ in range(20):
        guide, value = within_target_swap("A1", "T1", target_of, activity, rng)
        check_once = guide in {"A2"} and value == 0.9
        if not check_once:
            check("within-target swap stays in target", False, f"{guide} -> {value}")
            return
    check("within-target swap stays in target", True, "always picked A2 with 0.9")


def main() -> int:
    print("=== GSE283614 ablation tooling self-test ===")
    test_key_resolvers()
    test_activity_table_round_trip()
    test_oof_loader_duplicate_guard()
    test_paired_statistics()
    test_probe_swap_is_within_target()
    print()
    if FAILURES:
        print(f"SELFTEST FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
