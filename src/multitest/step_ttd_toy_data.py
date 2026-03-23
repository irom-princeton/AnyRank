'''
Apurva Badithela, Jan 10, 2026
To test the upper bound on expected efficiency gains from state of the art 
multitest correction methods, we create a toy dataset for policy comparison.
This dataset comprises of policies of different mean performances, and the
associated time to decision for each hypothesis test for a pair of policies. This 
time to decision is computed from an SPRT procedure in the STEP paper. 

The main idea is to order hypotheses tests by: i) their expected time to decision, and ii) the 
difference in mean performances of the policies being compared. If we provide multitest correction
algorithms with the optimal ordering, how many trials can we expect to save compared to naive 
Bonferroni correction?

Resources: 
[1] The STEP paper: https://arxiv.org/pdf/2503.10966
[2] SPRT procedure: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
[3] The time-to-decision numbers are taken from the figure given in the folder data/step_ttd_toy_data/
'''
import numpy as np
import os


def get_step_ttd_toy_data():
    p0 = 0.1 * np.arange(10) + 0.05
    p1 = (0.1 * np.arange(10) + 0.05)[::-1]
    ttd_data = np.array([
        [ 2.77,  4.17,  4.31,  6.32,  8.33, 11.37, 18.00, 31.84, 51.81,  0.00],
        [ 4.15,  6.03,  8.35, 11.78, 16.57, 26.05, 43.07, 60.53,  0.00,  0.00],
        [ 4.35,  8.00, 12.43, 18.93, 30.53, 47.56, 64.37,  0.00,  0.00,  0.00],
        [ 6.40, 11.43, 19.25, 31.94, 49.13, 66.65,  0.00,  0.00,  0.00,  0.00],
        [ 8.42, 16.96, 30.75, 48.88, 64.51,  0.00,  0.00,  0.00,  0.00,  0.00],
        [11.40, 28.02, 46.91, 65.73,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        [17.73, 42.14, 63.75,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        [30.49, 60.95,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        [52.70,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
    ])
    policy_pairs = [(p0i, p1i) for p0i in p0 for p1i in p1]    

    policy_ttd_data = {}
    for i, p0i in enumerate(p0):
        for j, p1i in enumerate(p1):
            if p1i > p0i:
                ttd = ttd_data[j, i]
                policy_ttd_data[(p0i, p1i)] = ttd
    return policy_ttd_data

def make_toy_data(num_policies=10, Nmax=500):
    policies = np.linspace(0.05, 0.95, num_policies)
    policy_data = np.zeros((Nmax, num_policies))
    for i, mu in enumerate(policies):
        policy_data[:, i] = np.random.binomial(1, mu, Nmax)
    return policy_data
    
if __name__ == "__main__":
    policy_ttd_data = get_step_ttd_toy_data()
    for key in policy_ttd_data:
        print("Policy pair: ", key, " Time to decision: ", policy_ttd_data[key])
