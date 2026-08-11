from multitest.plot_cld import make_violin_plots_from_ttds, get_cld_letters_from_ttds
import os
import numpy as np
from multitest.plot_cld import save_plot_bundle, get_cld_dict_generic

filename = "/n/fs/irom-testing/multitest/irom_data_outputs/2026-08-10_alpha_0.1/towel_wm_base_init_new/cld_violins/beta0/graphical_active.npz"
data = np.load(filename, allow_pickle=True)

breakpoint()