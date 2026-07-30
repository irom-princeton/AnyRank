import numpy as np
from sequentialized_barnard_tests.tools.plotting import compare_success_and_get_cld

model_name_list = [
    "Single Task",
    "LBM zeroshot",
    "LBM finetuned",
]

# Taken from Fig2A_HW_Seen_Nominal
success_array_list = [
    np.array([False,False,False,False,True,False,False,False,False,False,False,True,False,False,True,True,False,False,False,True,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]),
    np.array([False,False,False,False,False,False,False,False,False,True,False,True,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,True,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]),
    np.array([True,False,True,True,True,False,False,True,True,True,False,True,False,False,True,True,True,False,False,True,False,True,False,False,True,False,False,False,False,False,False,False,True,False,False,False,False,False,True,False,False,False,False,False,False,True,False,False,False,False]),
]

print([len(array) for array in success_array_list])

compare_success_and_get_cld(
    model_name_list,
    success_array_list,
    global_confidence_level=0.90,
    max_sample_size_per_model=50,  # 50 rollouts per policy
    shuffle=False,
    verbose=True,
)