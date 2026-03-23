############################################
# Implementation of fixed sequence multitest
# Given a list of policies, need to construct all possible hypotheses
# Evaluate each hypothesis in a fixed sequence manner
# Then, testing each hypothesis amounts to arriving at a decision at some time step
# This decision can be obtained sequentially as data arrives with partial credit evaluations as well.
############################################

import numpy as np
from sequentialized_barnard_tests import StepTest
from sequentialized_barnard_tests.base import Decision, Hypothesis

def initialize_bonferroni_test(Nmax, alpha, num_hypotheses):
    bonferroni_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha/num_hypotheses)
    return bonferroni_step_test

def initialize_step_test(Nmax, alpha):
    step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha)
    return step_test

def run_test_on_data(test, data0, data1):
    result = test.run_on_sequence(data0, data1)
    return result

def fixed_sequence_multitest(ordered_hypotheses, alpha=0.05):
    '''
    Given an ordered list of hypotheses, perform fixed sequence multitest
    Return the list of rejected hypotheses
    '''
    rejected_hypotheses = []
    for i, hypothesis in enumerate(ordered_hypotheses):
        # TODO: Perform test for the current hypothesis at level alpha
        # where is this currently being done in STEP?
        p_value = hypothesis.test()
        if p_value < alpha:
            rejected_hypotheses.append(hypothesis)
        else:
            break  # Stop testing further hypotheses
    return rejected_hypotheses

def fixed_sequence_multitest_step(ordered_hypotheses_policy_indices, policy_data, Nmax, alpha=0.05):
    '''
    Given an ordered list of hypotheses, perform fixed sequence multitest using sequential tests
    Return the list of rejected hypotheses along with their decision times
    '''
    rejected_hypotheses = []
    decision_times = {}
    for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
        step_test = initialize_step_test(Nmax, alpha)
        p0_index, p1_index = hypothesis_policy_indices
        data0 = policy_data[:, p0_index]
        data1 = policy_data[:, p1_index]
        print(f"Testing hypothesis {i+1}/{len(ordered_hypotheses_policy_indices)}: Policy pair indices {hypothesis_policy_indices}"
              f" with data means {np.mean(data0):.2f} vs {np.mean(data1):.2f}")
        result = run_test_on_data(step_test, data0, data1)
        
        if result.decision == Decision.AcceptAlternative:
            rejected_hypotheses.append(hypothesis_policy_indices)
            decision_times[hypothesis_policy_indices] = result.info["Time"]
        else:
            decision_times[hypothesis_policy_indices] = Nmax
            break  # Stop testing further hypotheses
    return rejected_hypotheses, decision_times

def bonferroni_multitest(ordered_hypotheses_policy_indices, policy_data, Nmax, alpha):
    '''
    Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
    Return the list of rejected hypotheses along with their decision times
    '''
    rejected_hypotheses = []
    decision_times = {}
    num_hypotheses = len(ordered_hypotheses_policy_indices)
    bonferroni_step_test = initialize_bonferroni_test(Nmax, alpha, num_hypotheses)
    for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
        p0_index, p1_index = hypothesis_policy_indices
        data0 = policy_data[:, p0_index]
        data1 = policy_data[:, p1_index]
        result = run_test_on_data(bonferroni_step_test, data0, data1)
        if result.decision == Decision.AcceptAlternative:
            rejected_hypotheses.append(hypothesis_policy_indices)
            decision_times[hypothesis_policy_indices] = result.info["Time"]
        else:
            decision_times[hypothesis_policy_indices] = Nmax
    return rejected_hypotheses, decision_times