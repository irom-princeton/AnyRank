import numpy as np 
from sequentialized_barnard_tests import MirroredStepTest, StepTest
from sequentialized_barnard_tests.base import Decision, Hypothesis
from scipy.special import comb
from tqdm import tqdm

if __name__ == "__main__":
    np.random.seed(42)
    n_policies=6

    # Construct an exchange matrix (will be useful later)
    exchange_matrix = np.zeros((n_policies, n_policies))
    for i in range(n_policies):
        exchange_matrix[i, n_policies-1-i] = 1.
    
    equispaced = True
    if equispaced:
        p_vec = np.arange(n_policies) / n_policies + (1. / (2. * n_policies))
    else:
        fixed_gap_width = 0.5 / n_policies
        rand_gap_widths = np.random.rand(n_policies)
        rand_gap_widths *= ((0.5 * (n_policies-1) / (n_policies*np.sum(rand_gap_widths))))
        p_vec = np.cumsum(fixed_gap_width + rand_gap_widths)
    
    print(f"There are {n_policies} policies, with success rates: ")
    print(p_vec)
    print()

    # Define run parameters
    Nmax = 500
    n_runs = 400

    DATA = np.zeros((Nmax, n_policies))

    # Verify the number of comparisons
    number_comparisons = int(comb(n_policies, 2))
    print("Full number of comparisons: ")
    print(number_comparisons)
    print()

    # Initialize the tests
    base_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, 0.05)
    bonferroni_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, 0.005) # why is it this and not alpha/num_hypotheses?

    # Initialize the data storage 
    times_to_decision_base = np.zeros((n_runs, n_policies, n_policies))
    times_to_decision_bonferroni = np.zeros((n_runs, n_policies, n_policies))

    correctness_base = np.zeros((n_runs, n_policies, n_policies))
    correctness_bonferroni = np.zeros((n_runs, n_policies, n_policies))

    full_ranking_base = np.ones(n_runs)
    full_ranking_bonferroni = np.ones(n_runs)

    # Loop over the number of independent (meta-) runs
    for n in tqdm(range(n_runs)):
        for i in range(p_vec.shape[0]):
            DATA[:, i] = np.random.binomial(1, p_vec[i], Nmax)

        for i in range(n_policies-1):
            for j in range(i+1, n_policies):
                data0 = DATA[:, i]
                data1 = DATA[:, j]

                base_step_result = base_step_test.run_on_sequence(data0, data1)
                bonferroni_step_result = bonferroni_step_test.run_on_sequence(data0, data1)

                if base_step_result.decision is Decision.AcceptAlternative:
                    correctness_base[n, i, j] = 1
                if bonferroni_step_result.decision is Decision.AcceptAlternative:
                    correctness_bonferroni[n, i, j] = 1
                times_to_decision_base[n, i, j] = base_step_result.info["Time"]
                times_to_decision_bonferroni[n, i, j] = bonferroni_step_result.info["Time"]
    
        for i in range(n_policies-1):
            for j in range(i+1, n_policies):
                if correctness_base[n, i, j] < 0.5:
                    full_ranking_base[n] = 0.
                if correctness_bonferroni[n, i, j] < 0.5:
                    full_ranking_bonferroni[n] = 0.
    
    average_times_to_decision_base = np.mean(times_to_decision_base, axis=0)
    average_times_to_decision_bonferroni = np.mean(times_to_decision_bonferroni, axis=0)
    complexity_critical_base = 0
    complexity_critical_bonferroni = 0
    complexity_total_base = np.sum(average_times_to_decision_base)
    complexity_total_bonferroni = np.sum(average_times_to_decision_bonferroni)

    complexity_precise_base = np.zeros(n_policies)
    complexity_precise_bonferroni = np.zeros(n_policies)

    for i in range(n_policies - 1):
        complexity_critical_base += average_times_to_decision_base[i, i+1]
        complexity_critical_bonferroni += average_times_to_decision_bonferroni[i, i+1]
    
    average_times_to_decision_base = np.matmul(average_times_to_decision_base, exchange_matrix)
    average_times_to_decision_bonferroni = np.matmul(average_times_to_decision_bonferroni, exchange_matrix)

    max_row_base = np.max(average_times_to_decision_base, axis=1)
    max_row_bonferroni = np.max(average_times_to_decision_bonferroni, axis=1)

    max_column_base = np.max(average_times_to_decision_base, axis=0)
    max_column_bonferroni = np.max(average_times_to_decision_bonferroni, axis=0)

    for i in range(n_policies):
        complexity_precise_base[i] = np.maximum(max_row_base[n_policies-i-1], max_column_base[i])
        complexity_precise_bonferroni[i] = np.maximum(max_row_bonferroni[n_policies-i-1], max_column_bonferroni[i])
    
    print("Average times to decision (base): ")
    print(average_times_to_decision_base)

    print()
    print("Average times to decision Bonferroni: ")
    print(average_times_to_decision_bonferroni)

    print()
    print()
    print("Full Ranking Correctness (Base) ", np.mean(full_ranking_base))
    print("Full Ranking Correctness (Bonferroni) ", np.mean(full_ranking_bonferroni))
    
    print()
    print("Total naive sample complexity (Base): ", complexity_total_base)
    print("Total naive sample complexity (Bonferroni): ", complexity_total_bonferroni)
    print()
    print("Precise sample complexity (Base): ", complexity_precise_base)
    print("Precise sample complexity (Bonferroni): ", complexity_precise_bonferroni)
    print()
    print("Total precise sample complexity (Base): ", np.sum(complexity_precise_base))
    print("Total precise sample complexity (Bonferroni): ", np.sum(complexity_precise_bonferroni))
    print()
    print("Fraction of total complexity in 'critical' comparisons (Base): ", complexity_critical_base / complexity_total_base)
    print("Fraction of total complexity in 'critical' comparisons (Bonferroni): ", complexity_critical_bonferroni / complexity_total_bonferroni)