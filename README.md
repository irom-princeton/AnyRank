# License
The code is provided under a Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. Under the license, the code is provided royalty free for non-commercial purposes only.

# Installation
```bash
git clone git@github.com:irom-princeton/AnyRank.git
cd AnyRank
pip install -e .
```

**NOTE:** Conda's `environment.yml` file is deprecated. For developers that need editable pip installation, we recommend setting up a separate python virtualenv.


# Reproducing paper results
```bash
python scripts/lbm_individual_test.py                                        # LBM experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena4           # RoboArena4 experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena7           # RoboArena7 experiments
python scripts/roboarena_individual_test.py  --exp_name roboarena4_wm_prior  # RoboArena4 experiments with world-model priors
python scripts/roboarena_individual_test.py  --exp_name roboarena7_wm_prior  # RoboArena7 experiments with world-model priors
```