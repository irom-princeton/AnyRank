PYTHONPATH="/home/dasnyder/Documents/GitHub/multi_hypothesis_test/":"${PYTHONPATH}"
export PYTHONPATH

# # Run on progress-as-preference case
python scripts/savings_graphical_wm_sequential_artificial.py

# Get TTD Level sets (check hypothesis)
# python scripts/find_ttd_invariance.py
# python scripts/find_ttd_invariance_via_SPRT.py
# python scripts/find_ttd_invariance_via_SPRT_v2.py
# python scripts/investigate_ttd_from_minmax_sprt.py
