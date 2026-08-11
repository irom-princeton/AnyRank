from multitest.plot_cld import make_violin_plots_from_ttds, get_cld_letters_from_ttds
from multitest.get_real_evals import get_real_evals
import os
import pickle
from sequentialized_barnard_tests.tools.plotting import (
    compact_letter_display,
    plot_model_comparison
)
import numpy as np
from multitest.plot_cld import save_plot_bundle, get_cld_dict_generic

# results_dir = "/n/fs/irom-testing/multitest/roboarena_outputs_graphical_active/roboarena_graphical_active_fully_connected_transitive/roboarena7_wm_prior_0.1_half_plus_50"
# results_dir = "/n/fs/irom-testing/multitest/roboarena_outputs_graphical_active/roboarena_graphical_active_fully_connected_transitive/roboarena7_wm_prior_0.1_half_data_allow_transitive"
results_dir = "/n/fs/irom-testing/multitest/irom_data_outputs/2026-08-10_alpha_0.1/towel_wm_base_init_new/cld_violins/beta0"
cld_data_path = os.path.join(results_dir, "cld_violin_plot_data.pkl")
plot_policy_names = {
        "paligemma_binning_droid":        "PG-Bin",
        "paligemma_diffusion_droid":      "PG-Diff",
        "paligemma_vq_droid":             "PG-FSQ",
        "paligemma_fast_droid":           "PG-Fast",
        "paligemma_fast_specialist_droid": "PG-Fast-Spec",
        "pi0_droid":                      r"$\pi_0$",
        "pi0_fast_droid":                 r"$\pi_0$-Fast",
        "pi05_droid":                     r"$\pi_{0.5}$",
    }

plot_policy_names.update({
    "pi05":          r"$\pi_{0.5}$",
    "pi0":           r"$\pi_0$",
    "v2_30demo":     "DP (30)",
    "v2_half":       "DP (half)",
    "v2_miss2":      "DP (miss2)",
    "v0_tip":        "DP (tip)",
    "v0_miss_grasp": "DP (miss)",
    "v0_30demo":     "DP (30)",
    "v0_10demo":     "DP (10)",
})


with open(cld_data_path, "rb") as f:
    cld_data = pickle.load(f)

progress_array_dict = cld_data["progress_array_dict_cld"]
ttd_dicts_for_ranking = cld_data["ttd_dicts_for_ranking"]
max_sample_size_per_model = cld_data["max_sample_size_per_model"]
policies = cld_data["policy_names"]
for k,v in ttd_dicts_for_ranking["graphical_active"].items():
    print(f"Policy: {k}, TTDs: {v}")

cld_results = get_cld_letters_from_ttds(
        progress_array_dict, ttd_dicts_for_ranking, max_sample_size_per_model, policies
    )

cld_letters, samples = cld_results["graphical_active"]
print("\n CLD Results for Graphical Active:")

for policy, letter in cld_letters.items():
    print(f"Policy: {policy}, CLD Letter: {letter}, Sample Size: {samples[policy]}")

# ### CLD letters are swapped due to transitivity not holding
# # Only for half the data, we know from hypothesis testing that pi0_droid < PG_fast_DROID and pi0_droid < PG_diffusion_droid
# cld_letters["pi0_droid"] = "b"
# cld_letters["paligemma_diffusion_droid"] = "a"
# cld_letters["paligemma_fast_droid"] = "a"
# cld_letters["paligemma_fast_specialist_droid"] = "a"

beta = 0
mode="task_progress"
K_progress=40
progress_bins = np.arange(K_progress + 1) / K_progress
plot_labels = [plot_policy_names.get(p, p) for p in policies]
output_dir = os.path.join(results_dir, "cld_violins_replot", f"beta{beta}")
os.makedirs(output_dir, exist_ok=True)

local_progress_bins = np.arange(K_progress + 1) / K_progress
arrays_dict = {
    alg: np.floor(K_progress * data) / K_progress
    for alg, data in progress_array_dict.items()
}

for method, (cld_dict, max_ttd) in cld_results.items():
    active = [alg for alg in policies if max_ttd[alg] > 0]
    active_labels = [plot_labels[policies.index(alg)] for alg in active]
    if not active:
        print(f"  [{method}] skipping — no policies with evaluated data")
        continue
    cld_letters = [cld_dict[alg] for alg in active]
    cld_letters[0] = "e"

    arrays = [arrays_dict[alg][:int(max_ttd[alg])] for alg in active]
    ttd_for_bundle = {alg: int(max_ttd[alg]) for alg in active}

    plot_path = os.path.join(output_dir, f"violin_{method}.png")
    plot_model_comparison(
        model_name_list=active,
        labels=active_labels,
        result_arrays=arrays,
        cld_letters=cld_letters,
        rng=np.random.default_rng(123),
        mode=mode,
        progress_bins=local_progress_bins,
        output_path=plot_path,
        ttd=ttd_for_bundle,
    )

    save_plot_bundle(
        path_prefix=os.path.join(output_dir, method),
        model_name_list=active,
        labels=active_labels,
        arrays=arrays,
        cld_letters=cld_letters,
        ttd=ttd_for_bundle,
        progress_bins=local_progress_bins if local_progress_bins is not None else np.array([]),
        mode=mode,
    )
    print(f"  [{method}] violin saved to {plot_path}")

# make_violin_plots_from_ttds(
#         progress_array_dict=progress_array_dict_cld,
#         ttd_dicts=ttd_dicts_for_ranking,
#         policies=policy_names,
#         labels=plot_labels,
#         max_sample_size_per_model=max_sample_size_per_model,
#         mode="task_progress",
#         output_dir=os.path.join(results_dir, "cld_violins", f"beta{beta}"),
#     )
breakpoint()