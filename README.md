# License
The code is provided under the MIT license. It depends on [`statistical-comparison-core`](https://github.com/TRI-ML/statistical_comparison_core) and [`sequentialized-barnard-tests`](https://github.com/TRI-ML/sequentialized_barnard_tests), which are separately licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. They are not included in this repository. Users should review their license statements before usage.

# Installation
```bash
git clone git@github.com:irom-princeton/AnyRank.git
cd AnyRank
pip install -e .
```

**NOTE:** Conda's `environment.yml` file is deprecated. For developers who need editable pip installation, we recommend setting up a separate Python virtualenv.

# Inference modes

AnyRank forwards an `inference_mode` option to the mirrored sequential tests
provided by `nscore` (>=0.1.2) and `sequentialized-barnard-tests` (>=0.0.7).
The following options are available:

- `ranking`: defines a ranking problem where exact ties in pairwise comparisons make either
  strict ordering wrong. Less sample efficient but preserves family-wise error rate (FWER)
  guarantee at `alpha` even if some policies have exactly the same mean performance score.
- `ranking_no_ties` (default): defines a ranking problem with the FWER guarantee restricted to
  the parameter space where exact ties are not considered. This is more sample efficient and
  the default choice made in the paper.

The full statistical definitions live in [`statistical_comparison_core.MirroredTestMixin`](https://github.com/TRI-ML/statistical_comparison_core/blob/v0.2.1/statistical_comparison_core/base.py#L163) (>=0.2.1).

# Reproducing paper results
```bash
python scripts/lbm_individual_test.py                                        # LBM experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena4           # RoboArena4 experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena7           # RoboArena7 experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena4_wm_prior  # RoboArena4 experiments with world-model priors
python scripts/roboarena_individual_test.py  --exp_name roboarena7_wm_prior  # RoboArena7 experiments with world-model priors
```
