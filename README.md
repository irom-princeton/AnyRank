# License
The code is provided under a Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. Under the license, the code is provided royalty free for non-commercial purposes only.

# Installation
```bash
git clone git@github.com:irom-princeton/AnyRank.git
cd AnyRank
pip install -e .
```

**NOTE:** Conda's `environment.yml` file is deprecated. For developers that need editable pip installation, we recommend setting up a separate python virtualenv.

# Inference modes

AnyRank forwards an `inference_mode` option to the mirrored sequential tests
provided by `nscore` and `sequentialized-barnard-tests`.

- `comparison`: treats equality as belonging to the null or baseline side and
  uses the full child-test alpha.
- `ranking`: treats either strict ordering error as relevant at an exact tie,
  uses alpha/2 for each child test, and is the conservative two-sided ranking
  mode.
- `ranking_no_ties`: uses ranking semantics with the guarantee restricted to
  the parameter space where `p_0 != p_1`, and uses the full child-test alpha.

AnyRank defaults to `ranking_no_ties` because it reports rankings and has no
baseline policy. In contrast, `statistical-comparison-core` and
`sequentialized-barnard-tests` default to `comparison` to preserve behavior for
existing A/B comparison callers. Numerically, `ranking_no_ties` and
`comparison` are identical; the default is a declaration of AnyRank's intended
semantics rather than a behavior change. Only `ranking` changes results.

Switching to `ranking` halves the effective per-pair alpha, which can lengthen
decision times and leave more pairs unresolved within a fixed budget. For STEP
comparisons, it can also trigger one-time policy synthesis for each derived
alpha value. See `statistical_comparison_core.MirroredTestMixin` for the full
statistical treatment of these modes.

# Reproducing paper results
```bash
python scripts/lbm_individual_test.py                                        # LBM experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena4           # RoboArena4 experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena7           # RoboArena7 experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena4_wm_prior  # RoboArena4 experiments with world-model priors
python scripts/roboarena_individual_test.py  --exp_name roboarena7_wm_prior  # RoboArena7 experiments with world-model priors
```
