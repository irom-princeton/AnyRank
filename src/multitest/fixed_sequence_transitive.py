### Transitive reasoning with fixed sequence testing
### Apurva Badithela

import os
import copy
from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.nonparametric_nsm import MirroredContinuousNsmTest
from sequentialized_barnard_tests.step import MirroredStepTest
from sequentialized_barnard_tests.nsm_graphical import ContinuousNsmTest, MirroredContinuousNsmTest_AlphaAdaptive
import numpy as np
from multitest.transivity import transitive_closure_relations, transitive_decision_times

class FixedSequenceTransitive:
    def __init__(self, num_policies, null_hypotheses, subfolder = "fixed_sequence_transitive", total_alpha=0.05):
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.total_alpha = total_alpha # total alpha is the family-wise error rate we want to control, so we will be working with 1 - total_alpha as the error budget to allocate
        self.num_policies=num_policies # number of policies
        self.num_hypotheses = len(null_hypotheses) # number of hypotheses
        self.save_dir = os.path.join(repo_dir, "outputs", subfolder)
        os.makedirs(self.save_dir, exist_ok=True)

    def initialize_graphical_nsm_test(self, Nmax, alpha):
        nonparametric_nsm_test = MirroredContinuousNsmTest_AlphaAdaptive(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(2)
            )
        return nonparametric_nsm_test

    def initialize_bonferroni_test(self, Nmax, alpha, num_hypotheses):
        bonferroni_step_test = MirroredContinuousNsmTest(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha/num_hypotheses, c=np.arange(2)
            )
        return bonferroni_step_test
    
    def initialize_graphical_nsm_test_partial_progress(self, Nmax, alpha):
        nonparametric_nsm_test = MirroredContinuousNsmTest_AlphaAdaptive(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(11)/10
            )
        return nonparametric_nsm_test

    def initialize_bonferroni_test_partial_progress(self, Nmax, alpha, num_hypotheses):
        bonferroni_step_test = MirroredContinuousNsmTest(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha/num_hypotheses, c=np.arange(11)/10
            )
        return bonferroni_step_test

    def run_test_on_data(self,test, data0, data1):
        result = test.run_on_sequence(data0, data1)
        return result

    def adjust_transivity(self, rejected_hypotheses, decision_times, hypotheses_correct, num_policies): 
        updated_decision_times = copy.deepcopy(decision_times)
        updated_hypotheses_correct = copy.deepcopy(hypotheses_correct)       
        timings = transitive_decision_times(decision_times, hypotheses_correct, num_policies)
        for hyp, values in timings.items():
            if values["direction"] != 0:
                if hyp not in rejected_hypotheses:
                    rejected_hypotheses.append(hyp)
                updated_decision_times[hyp] = values["earliest_time"]
                updated_hypotheses_correct[hyp] = values["direction"]
        return rejected_hypotheses, updated_decision_times, updated_hypotheses_correct

    def save_rejected_hypotheses(self, rejected_hypotheses, decision_times, hypotheses_correct, timestep):
        # Save .txt file of rejected hypotheses and their decision times
        save_path = os.path.join(self.save_dir, f"rejected_hypotheses_{timestep}.txt")
        with open(save_path, "w") as f:
            f.write("Rejected Hypotheses: Decision Times, Correctness:\n")
            for hypothesis in rejected_hypotheses:
                f.write(f"{hypothesis}: {decision_times[hypothesis]}, {hypotheses_correct[hypothesis]}\n")
        print(f"Saved rejected hypotheses and decision times at timestep {timestep} to {save_path}")

    def fixed_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=False, allow_transitive=True):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        hypotheses_correct = {hypothesis_policy_indices: 0.0 for hypothesis_policy_indices in ordered_hypotheses_policy_indices} # 1 if we correctly rejected, -1 if we incorrectly rejected, 0 if not rejected
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            valid = ~np.isnan(data0) & ~np.isnan(data1)
            data0 = data0[valid]
            data1 = data1[valid]

            if allow_transitive:
                hyp_correct_at_curr_step = {hyp: hypotheses_correct[hyp] for hyp in rejected_hypotheses}
                closure = transitive_closure_relations(rejected_hypotheses, hyp_correct_at_curr_step, n_policies=policy_data.shape[1])
                
                if hypothesis_policy_indices in closure["implied"]: 
                    print("Skipping hypothesis ", hypothesis_policy_indices, " since it's implied by previously rejected hypotheses via transitivity")
                    if hypothesis_policy_indices not in rejected_hypotheses:
                        rejected_hypotheses.append(hypothesis_policy_indices)
                        hypotheses_correct[hypothesis_policy_indices] = 1.0
                        decision_times[hypothesis_policy_indices] = 0.0 # we can consider the decision time for this implied rejection to be 0 since we can infer it immediately based on the previously rejected hypotheses and the graph structure
                    continue
                elif (p1_index, p0_index) in closure["implied"]: 
                    print("Skipping hypothesis ", hypothesis_policy_indices, " since it's implied by previously rejected hypotheses via transitivity")
                    if hypothesis_policy_indices not in rejected_hypotheses:
                        rejected_hypotheses.append(hypothesis_policy_indices)
                        hypotheses_correct[hypothesis_policy_indices] = -1.0
                        decision_times[hypothesis_policy_indices] = 0.0 # we can consider the decision time for this implied rejection to be 0 since we can infer it immediately based on the previously rejected hypotheses and the graph structure
                    continue

            if not bernoulli:
                bonferroni_nsm_test = self.initialize_bonferroni_test_partial_progress(Nmax, alpha, 1)
            else:
                bonferroni_nsm_test = self.initialize_bonferroni_test(Nmax, alpha, 1)

            len_common = min(len(data0), len(data1))
            data0 = data0[:len_common]
            data1 = data1[:len_common]

            result = self.run_test_on_data(bonferroni_nsm_test, data0, data1)
            
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                hypotheses_correct[hypothesis_policy_indices] = 1.0
                if "Time" in result.info:
                    decision_times[hypothesis_policy_indices] = result.info["Time"]
                else:
                    decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            elif result.decision == Decision.AcceptNull:
                rejected_hypotheses.append(hypothesis_policy_indices) # if we accept null, we can also infer the opposite direction
                hypotheses_correct[hypothesis_policy_indices] = -1.0
                if "Time" in result.info:
                    decision_times[hypothesis_policy_indices] = result.info["Time"]
                else:
                    decision_times[hypothesis_policy_indices] = result.info["result_for_null"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = Nmax
                break
            
            # Save rejected hypotheses and their decision times at each step
            self.save_rejected_hypotheses(rejected_hypotheses, decision_times, hypotheses_correct, timestep=i)

        # Note that we break after the first non-rejection to mimic a fixed-sequence testing procedure, where we only move on to the next hypothesis if we rejected the previous one. This is different from the weighted_bonferroni_multitest where we will test all hypotheses regardless of rejections, but just with different alpha levels.
        if allow_transitive:
            rejected_decision_times = {hyp: decision_times[hyp] for hyp in rejected_hypotheses}
            rejected_hyp_correct = {hyp: hypotheses_correct[hyp] for hyp in rejected_hypotheses}
            updated_rejected_hypotheses, updated_decision_times, updated_hypotheses_correct = self.adjust_transivity(rejected_hypotheses, rejected_decision_times,  rejected_hyp_correct, policy_data.shape[1])
            for new_hyp in updated_rejected_hypotheses:
                if new_hyp not in rejected_hypotheses:
                    rejected_hypotheses.append(new_hyp)
                    decision_times[new_hyp] = updated_decision_times[new_hyp]
                    hypotheses_correct[new_hyp] = updated_hypotheses_correct[new_hyp]
                if decision_times[new_hyp] == "N/A":
                    decision_times[new_hyp] = updated_decision_times[new_hyp]
                    hypotheses_correct[new_hyp] = updated_hypotheses_correct[new_hyp]
                elif updated_decision_times[new_hyp] < decision_times[new_hyp]:
                    decision_times[new_hyp] = updated_decision_times[new_hyp]
                    hypotheses_correct[new_hyp] = updated_hypotheses_correct[new_hyp]
        self.save_rejected_hypotheses(rejected_hypotheses, decision_times, hypotheses_correct, timestep="final")
        return rejected_hypotheses, decision_times, hypotheses_correct

def generate_partial_data(real_mean, N):
    # Generate N samples from a Gaussian distribution with real mean
    data = np.random.normal(loc=real_mean, scale=1.0, size=N)
    data = np.clip(data, 0, 1) # ensure all data is between min and max to avoid issues with the mirrored test which requires data to be in a certain range
    return data

def test_inseparable():
    # Experiment to test if fixed sequence will correctly count for hypotheses that 
    # rejected 
    num_policies = 5
    Nmax = 1000
    alpha=0.1
    subfolder = "fixed_sequence_transitive_inseparable_policies"
    # Five policies:
    real_means = {
        "pi0": 0.3,
        "pi1": 0.3,
        "pi2": 0.6,
        "pi3": 0.8,
        "pi4": 1.0
    }

    # Incorrect means for pi0 an pi1:
    sim_means = {
        "pi0": 0.28,
        "pi1": 0.32,
        "pi2": 0.6,
        "pi3": 0.82,
        "pi4": 1.0
    }
    return num_policies, Nmax, alpha, subfolder, real_means, sim_means

def test_fully_separable():
    num_policies = 5
    Nmax = 1000
    alpha=0.1
    subfolder = "fixed_sequence_transitive_separable_policies"
    # Five policies:
    real_means = {
        "pi0": 0.2,
        "pi1": 0.4,
        "pi2": 0.6,
        "pi3": 0.8,
        "pi4": 1.0
    }

    # Incorrect means for pi0 an pi1:
    sim_means = {
        "pi0": 0.4,
        "pi1": 0.2,
        "pi2": 0.6,
        "pi3": 0.8,
        "pi4": 1.0
    }
    return num_policies, Nmax, alpha, subfolder, real_means, sim_means

def test_fully_separable_example_2():
    num_policies = 5
    Nmax = 1000
    alpha=0.1
    subfolder = "fixed_sequence_transitive_separable_policies_example_2"

    # Same as fully separable 1 but real policies are not in increasing order of means.
    # Five policies:
    real_means = {
        "pi0": 0.4,
        "pi1": 0.2,
        "pi2": 0.6,
        "pi3": 0.8,
        "pi4": 1.0
    }

    # Incorrect means for pi0 an pi1:
    sim_means = {
        "pi0": 0.4,
        "pi1": 0.2,
        "pi2": 0.6,
        "pi3": 0.8,
        "pi4": 1.0
    }
    return num_policies, Nmax, alpha, subfolder, real_means, sim_means

if __name__ == "__main__":
    experiment = "inseparable" # options: "fully_separable", "fully_separable_example_2", "inseparable"

    if experiment == "fully_separable":
        num_policies, Nmax, alpha, subfolder, real_means, sim_means = test_fully_separable()
    elif experiment == "fully_separable_example_2":
        num_policies, Nmax, alpha, subfolder, real_means, sim_means = test_fully_separable_example_2()
    else:
        num_policies, Nmax, alpha, subfolder, real_means, sim_means = test_inseparable()

    policies = list(real_means.keys())
    policy_data = [generate_partial_data(real_means[p], Nmax) for p in policies]
    policy_data = np.array(policy_data).T # shape (Nmax, num_policies)
    
    real_sim_means = [(real_means[p], sim_means[p]) for p in policies]
    null_hypotheses = [
        (sim_means[policies[i]], sim_means[policies[j]])
        for i in range(num_policies) for j in range(i + 1, num_policies)
    ]
    null_hypotheses_policy_indices = [
        (i, j) for i in range(num_policies) for j in range(i + 1, num_policies)
    ]

    print("Ordered policy pairs (by time to decision on null hypothesis mu0 < mu1): ")
    hyp_diffs = [abs(mu1 - mu0) for (mu0, mu1) in null_hypotheses]
    ordered_hypotheses = sorted(zip(null_hypotheses, null_hypotheses_policy_indices,hyp_diffs), key=lambda x: x[2], reverse=True) 
    policy_indices = {i: f"pi{i}" for i in range(num_policies)}
    ordered_null_hypotheses = [hyp[0] for hyp in ordered_hypotheses]
    ordered_null_hypotheses_policy_indices = [hyp[1] for hyp in ordered_hypotheses]
    
    #######
    ### Hypothesis order:
    fixed_test = FixedSequenceTransitive(num_policies, ordered_null_hypotheses_policy_indices, subfolder=subfolder, total_alpha=alpha)

    hypotheses_order_filepath = os.path.join("outputs", subfolder, "hypotheses_order.txt")
    with open(hypotheses_order_filepath, "w") as f:
        f.write("Null Hypotheses (mu0, mu1) and their corresponding policy indices:\n")
        for hyp, hyp_indices in zip(ordered_null_hypotheses, ordered_null_hypotheses_policy_indices):
            f.write(f"{hyp}: {hyp_indices}\n")
    print(f"Saved null hypotheses and their corresponding policy indices to {hypotheses_order_filepath}")

    #######
    ### Fixed sequence:
    rejected_hypotheses, decision_times, hypotheses_correct = fixed_test.fixed_multitest(ordered_null_hypotheses_policy_indices, policy_data=policy_data, Nmax=Nmax, alpha=alpha, bernoulli=False, allow_transitive=True)
