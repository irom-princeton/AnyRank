from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.nonparametric_nsm import MirroredContinuousNsmTest
from sequentialized_barnard_tests.step import MirroredStepTest
from sequentialized_barnard_tests.nsm_graphical import ContinuousNsmTest, MirroredContinuousNsmTest_AlphaAdaptive
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from math import comb
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import copy

class SequentialGraphicalTest:
    def __init__(self, num_policies, null_hypotheses, total_alpha=0.05):
        self.total_alpha = total_alpha # total alpha is the family-wise error rate we want to control, so we will be working with 1 - total_alpha as the error budget to allocate
        self.num_policies=num_policies # number of policies
        self.num_hypotheses = len(null_hypotheses) # number of hypotheses

        # default setting for all nodes and edges:
        # self.G[i,j] indicates the directed edge from hypothesis Hi to Hj
        self.G : Sequence[Sequence[float]] = np.zeros((self.num_hypotheses, self.num_hypotheses))
        # default:
        # evenly split weight across all hypotheses:
        for i in range(self.num_hypotheses):
            for j in range(self.num_hypotheses):
                if i != j:
                    self.G[i,j] = 1/(self.num_hypotheses-1)

        self.alpha : Sequence[float]=np.array([self.total_alpha/self.num_hypotheses for _ in range(self.num_hypotheses)]) # equal split
        self.null_nypotheses = null_hypotheses
        self.hypothesis_labels = [f"H{i}\nμ₀={mu0:.2f}, μ₁={mu1:.2f}" for i, (mu0, mu1) in enumerate(null_hypotheses)]
        self.check_params()

    def set_params(self, G: Optional = None, alpha:Optional = None, total_alpha:Optional = None):
        if G is not None:
            self.G = np.asarray(G)
        if alpha is not None:
            self.alpha = np.asarray(alpha)
        if total_alpha is not None:
            self.total_alpha = total_alpha
        self.check_params()
    
    def check_params(self):
        # Basic validation of input:
        assert abs(np.sum(self.alpha) - self.total_alpha) < 1e-5, "Split alpha budget"
        if self.alpha.shape != (self.num_hypotheses,):
            raise ValueError(f"delta_init must have shape ({self.num_hypotheses},), got {self.alpha.shape}")
        if self.G.shape != (self.num_hypotheses, self.num_hypotheses):
            raise ValueError(f"G_init must have shape ({self.num_hypotheses}, {self.num_hypotheses}), got {self.G.shape}")
        if np.any(self.alpha < 0):
            raise ValueError("All initial error budgets must be nonnegative.")
        if np.any(self.G < 0) or np.any(self.G > 1):
            raise ValueError("All graph weights must lie in [0, 1].")
        if not np.allclose(np.diag(self.G), 0.0):
            raise ValueError("Graph must satisfy g_{i,i} = 0 for all i.")
        if np.any(self.G.sum(axis=1) > 1 + 1e-12):
            raise ValueError("Each row of G must satisfy sum_j g_{i,j} <= 1.")

    def initialize_nsm_test(self, Nmax, alpha):
        nonparametric_nsm_test = MirroredContinuousNsmTest(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(2)
            )
        return nonparametric_nsm_test
    
    def plot_graph(self, ax=None, title: str = "Graphical Test Graph"):
        """
        Plot self.G as a weighted directed graph.

        Node labels show the hypothesis description (mu0, mu1) from
        self.hypothesis_labels and the current alpha budget.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to draw on. Creates a new figure if None.
        title : str
            Plot title.
        """
        node_labels = {i: f"{self.hypothesis_labels[i]}\nα={self.alpha[i]:.3f}"
                       for i in range(self.num_hypotheses)}

        DG = nx.DiGraph()
        DG.add_nodes_from(range(self.num_hypotheses))
        for i in range(self.num_hypotheses):
            for j in range(self.num_hypotheses):
                if i != j and self.G[i, j] > 0:
                    DG.add_edge(i, j, weight=self.G[i, j])

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(10, self.num_hypotheses * 2.5), max(8, self.num_hypotheses * 2.0)))

        pos = nx.circular_layout(DG)
        nx.draw_networkx_nodes(DG, pos, ax=ax, node_size=5000, node_color="steelblue", alpha=0.85)
        nx.draw_networkx_labels(DG, pos, labels=node_labels, ax=ax, font_size=11, font_color="white")

        edge_weights = nx.get_edge_attributes(DG, "weight")
        nx.draw_networkx_edges(DG, pos, ax=ax, edgelist=list(edge_weights.keys()),
                               width=[v * 6 for v in edge_weights.values()],
                               edge_color=list(edge_weights.values()),
                               edge_cmap=plt.cm.Oranges, edge_vmin=0, edge_vmax=1,
                               arrows=True, arrowsize=30,
                               connectionstyle="arc3,rad=0.15")
        edge_labels = {(i, j): f"{w:.2f}" for (i, j), w in edge_weights.items()}
        nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, ax=ax, font_size=11, label_pos=0.3)

        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        return ax

    def sequential_graphical_test(self, p_values:Sequence[float], verbose=False) -> Tuple[Set[int], np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        p_values : sequence of float, length N
            The p-values p_1, ..., p_N.

        Returns
        -------
        rejected : set[int]
            Set of rejected hypothesis indices, using 0-based indexing.
        delta : np.ndarray
            Final local error budgets after all updates.
        G : np.ndarray
            Final graph after all updates.

        Notes
        -----
        This implements the paper's update:
            delta_j <- delta_j + delta_i * g_{i,j}     for lambda_j not in rejected and j != i
            delta_j <- 0                               otherwise

            g_{k,j} <- (g_{k,j} + g_{k,i} * g_{i,j}) / (1 - g_{j,i} * g_{i,j})
                    for lambda_k, lambda_j not in rejected and k != j
            g_{k,j} <- 0 otherwise

        after rejecting hypothesis i.
        """
        p = np.asarray(p_values, dtype=float)
        rejected: Set[int] = set()
        graphs_over_time = []
        alpha_at_rejected = []
        assert len(p) == self.num_hypotheses, "Mismatch in the number of hypotheses"
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("All p-values must lie in [0, 1].")

        while True:
            candidates = [i for i in range(self.num_hypotheses) if i not in rejected and p[i] <= self.alpha[i]]
            if not candidates:
                break
            
            # Plot graphs:
            graphs_over_time.append(self.plot_graph())
            
            if verbose:
                print(self.G)
            
            # Choose any i such that p_i <= delta_i"
            i = candidates[0]
            # Reject i
            rejected.add(i)
            alpha_at_rejected.append(copy.deepcopy(self.alpha[i]))
            active = [j for j in range(self.num_hypotheses) if j not in rejected]

            # Update delta_j
            new_alpha = np.zeros_like(self.alpha)
            for j in active:
                new_alpha[j] = self.alpha[j] + self.alpha[i] * self.G[i, j]
            
            # Update G:
            new_G = np.zeros_like(self.G)
            for k in active:
                for j in active:
                    if k == j:
                        new_G[k, j] = 0.0
                        continue

                    denom = 1.0 - self.G[k, i] * self.G[i, k]
                    # if np.isclose(denom, 0.0):
                    #     breakpoint()
                    #     raise ZeroDivisionError(
                    #         f"Denominator became zero while updating edge ({k}, {j}). "
                    #         "Check whether the graph satisfies the needed conditions."
                    #     )

                    if denom > 0.0: 
                        new_G[k, j] = (self.G[k, j] + self.G[k, i] * self.G[i, j]) / denom
                    else:
                        new_G[k, j] = 0.0

                    if new_G[k, j] > 1.1 or new_G[k, j] < 0.0:
                        breakpoint()
            self.alpha = copy.deepcopy(new_alpha)
            self.G = copy.deepcopy(new_G)
        return rejected, graphs_over_time, alpha_at_rejected
    
    # def debug_sequential_graphical_test(self, p_values:Sequence[float], prev_rejected:set[int]=set(), verbose=False) -> Tuple[Set[int], np.ndarray, np.ndarray]:
    #     """
    #     Parameters
    #     ----------
    #     p_values : sequence of float, length N
    #         The p-values p_1, ..., p_N.

    #     Returns
    #     -------
    #     rejected : set[int]
    #         Set of rejected hypothesis indices, using 0-based indexing.
    #     delta : np.ndarray
    #         Final local error budgets after all updates.
    #     G : np.ndarray
    #         Final graph after all updates.

    #     Notes
    #     -----
    #     This implements the paper's update:
    #         delta_j <- delta_j + delta_i * g_{i,j}     for lambda_j not in rejected and j != i
    #         delta_j <- 0                               otherwise

    #         g_{k,j} <- (g_{k,j} + g_{k,i} * g_{i,j}) / (1 - g_{j,i} * g_{i,j})
    #                 for lambda_k, lambda_j not in rejected and k != j
    #         g_{k,j} <- 0 otherwise

    #     after rejecting hypothesis i.
    #     """
    #     rejected = set()
    #     for p in prev_rejected:
    #         rejected.add(p)
    #     p = np.asarray(p_values, dtype=float)
    #     alpha_tol = 1e-8
    #     alpha_at_rejected = []
    #     graphs_over_time = []

    #     assert len(p) == self.num_hypotheses, "Mismatch in the number of hypotheses"
    #     if np.any(p < 0) or np.any(p > 1):
    #         raise ValueError("All p-values must lie in [0, 1].")

    #     while True:
    #         candidates = [i for i in range(self.num_hypotheses) if i not in rejected and p[i] <= self.alpha[i] and self.alpha[i] > alpha_tol]
    #         if not candidates:
    #             break
            
    #         # Plot graphs:
    #         graphs_over_time.append(self.plot_graph())
            
    #         if verbose:
    #             print(self.G)
            
    #         # Choose any i such that p_i <= delta_i"
    #         i = candidates[0]
    #         # Reject i
    #         rejected.add(i)
    #         alpha_at_rejected.append(copy.deepcopy(self.alpha[i]))
    #         active = [j for j in range(self.num_hypotheses) if j not in rejected]
    #         print(f"P-values: {p}", "Alpha budgets: ", self.alpha, "Active hypotheses: ", active)

    #         if verbose:
    #             # Rejecting candidate i:
    #             print("============================================================")
    #             print(f"Rejecting hypothesis H{i} with p-value {p[i]:.4e} <= alpha {self.alpha[i]:.4e}")
    #             print(f"Before rejection: ")
                
    #             # print("Graph G: \n", self.G, "\n")
    #             # print("alpha budgets: ", self.alpha, "\n")
            
    #         # Update delta_j
    #         new_alpha = np.zeros_like(self.alpha)
    #         print("i: ", i, "alpha[i]: ", self.alpha[i], "G[i]: ", self.G[i, :], "\n")
    #         for j in active:
    #             new_alpha[j] = self.alpha[j] + self.alpha[i] * self.G[i, j]
    #             if new_alpha[j] < alpha_tol:
    #                 new_alpha[j] = 0.0
    #         print("new_alpha: ", new_alpha, "\n")

    #         # Update G:
    #         new_G = np.zeros_like(self.G)
    #         for k in active:
    #             for j in active:
    #                 if k == j:
    #                     new_G[k, j] = 0.0
    #                     continue
                    
    #                 prod = self.G[k, i] * self.G[i, k]
    #                 denom = 1 - prod
    #                 # if prod > 1 - 1e-6:
    #                 #     denom+=1e-6
    #                 #     print("near cycle:", {"i": i, "k": k, "prod": prod, "denom": denom})
                        
    #                 new_G[k, j] = (self.G[k, j] + self.G[k, i] * self.G[i, j]) / denom

    #                 if denom > 1e-8: 
    #                     new_G[k, j] = (self.G[k, j] + self.G[k, i] * self.G[i, j]) / denom
    #                 else:
    #                     new_G[k, j] = 0.0
                    
    #                 if new_G[k, j] > 1.1 or new_G[k, j] < 0.0:
    #                     breakpoint()
    #             print(f"Rejected H{i} -> Updated G row {k}: ", new_G[k, :])
            
    #         print("alpha budgets (before update): ", self.alpha, "\n")
    #         print("P-values: ", p, "\n")

    #         self.alpha = copy.deepcopy(new_alpha)
    #         self.G = copy.deepcopy(new_G)

    #         print("All rejected hypotheses so far: ", rejected)
    #         print("Sum: ", np.sum(new_G, axis=1))
    #         print(f"After rejecting H{i}: ")
    #         # print("Graph G: \n", self.G, "\n")
    #         print("alpha budgets (after update): ", self.alpha, "\n")
    #         breakpoint()
    #         print("============================================================\n")
    #     return rejected, graphs_over_time, alpha_at_rejected

    def initialize_graphical_nsm_test(self, Nmax, alpha):
        nonparametric_nsm_test = MirroredContinuousNsmTest_AlphaAdaptive(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(2)
            )
        return nonparametric_nsm_test

    def initialize_bonferroni_test(self, Nmax, alpha, num_hypotheses):
        # bonferroni_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha/num_hypotheses)
        bonferroni_step_test = MirroredContinuousNsmTest(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha/num_hypotheses, c=np.arange(2)
            )
        # bonferroni_step_test = MirroredStepTest(
        #     alternative=Hypothesis.P0LessThanP1, n_max=Nmax, alpha=alpha
        # )
        return bonferroni_step_test
    
    def initialize_graphical_nsm_test_partial_progress(self, Nmax, alpha):
        nonparametric_nsm_test = MirroredContinuousNsmTest_AlphaAdaptive(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(11)/10
            )
        return nonparametric_nsm_test

    def initialize_bonferroni_test_partial_progress(self, Nmax, alpha, num_hypotheses):
        #bonferroni_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha/num_hypotheses)
        bonferroni_step_test = MirroredContinuousNsmTest(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha/num_hypotheses, c=np.arange(11)/10
            )
        return bonferroni_step_test

    def run_test_on_data(self,test, data0, data1):
        result = test.run_on_sequence(data0, data1)
        return result

    def fixed_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=False):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        hypotheses_correct = np.zeros(len(ordered_hypotheses_policy_indices))
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            valid = ~np.isnan(data0) & ~np.isnan(data1)
            data0 = data0[valid]
            data1 = data1[valid]
            
            if not bernoulli:
                bonferroni_step_test = self.initialize_bonferroni_test_partial_progress(Nmax, alpha, 1)
            else:
                bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha, 1)
                len_common = min(len(data0), len(data1))
                data0 = data0[:len_common]
                data1 = data1[:len_common]

            result = self.run_test_on_data(bonferroni_step_test, data0, data1)
            
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                hypotheses_correct[i] = 1.0
                if "Time" in result.info:
                    decision_times[hypothesis_policy_indices] = result.info["Time"]
                else:
                    decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            elif result.decision == Decision.AcceptNull:
                rejected_hypotheses.append(hypothesis_policy_indices) # if we accept null, we can also infer the opposite direction
                hypotheses_correct[i] = -1.0
                if "Time" in result.info:
                    decision_times[hypothesis_policy_indices] = result.info["Time"]
                else:
                    decision_times[hypothesis_policy_indices] = result.info["result_for_null"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = Nmax
                break
        # Note that we break after the first non-rejection to mimic a fixed-sequence testing procedure, where we only move on to the next hypothesis if we rejected the previous one. This is different from the weighted_bonferroni_multitest where we will test all hypotheses regardless of rejections, but just with different alpha levels.
        return rejected_hypotheses, decision_times, hypotheses_correct

    def bonferroni_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha, bernoulli=False):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        num_hypotheses = len(ordered_hypotheses_policy_indices)
        hypotheses_correct = np.zeros(num_hypotheses)
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            valid = ~np.isnan(data0) & ~np.isnan(data1)
            data0 = data0[valid]
            data1 = data1[valid]
            
            if not bernoulli:
                bonferroni_step_test = self.initialize_bonferroni_test_partial_progress(Nmax, alpha, num_hypotheses)
            else:
                bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha, num_hypotheses)
                len_common = min(len(data0), len(data1))
                data0 = data0[:len_common]
                data1 = data1[:len_common]

            result = self.run_test_on_data(bonferroni_step_test, data0, data1)
            
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                hypotheses_correct[i] = 1.0
                if "Time" in result.info:
                    decision_times[hypothesis_policy_indices] = result.info["Time"]
                else:
                    decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            elif result.decision == Decision.AcceptNull:
                rejected_hypotheses.append(hypothesis_policy_indices) # if we accept null, we can also infer the opposite direction
                hypotheses_correct[i] = -1.0
                if "Time" in result.info:
                    decision_times[hypothesis_policy_indices] = result.info["Time"]
                else:
                    decision_times[hypothesis_policy_indices] = result.info["result_for_null"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = "N/A"
        return rejected_hypotheses, decision_times, hypotheses_correct

    def weighted_bonferroni_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha_per_hypothesis, bernoulli=False):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        num_hypotheses = len(ordered_hypotheses_policy_indices)
        hypotheses_correct = np.zeros(num_hypotheses)
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            alpha_i = alpha_per_hypothesis[i] 
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            valid = ~np.isnan(data0) & ~np.isnan(data1)
            data0 = data0[valid]
            data1 = data1[valid]
            if not bernoulli:
                bonferroni_step_test = self.initialize_bonferroni_test_partial_progress(Nmax, alpha_i, num_hypotheses)
            else:
                bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha_i, num_hypotheses)
                len_common = min(len(data0), len(data1))
                data0 = data0[:len_common]
                data1 = data1[:len_common]

            result = self.run_test_on_data(bonferroni_step_test, data0, data1)
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                hypotheses_correct[i] = 1.0
                if "Time" in result.info:
                    decision_times[hypothesis_policy_indices] = result.info["Time"]
                else:
                    decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            elif result.decision == Decision.AcceptNull:
                rejected_hypotheses.append(hypothesis_policy_indices) # if we accept null, we can also infer the opposite direction
                hypotheses_correct[i] = -1.0
                if "Time" in result.info:
                    decision_times[hypothesis_policy_indices] = result.info["Time"]
                else:
                    decision_times[hypothesis_policy_indices] = result.info["result_for_null"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = "N/A"
        return rejected_hypotheses, decision_times, hypotheses_correct

    def parse_nsm_test_result(self, test_result):
        if test_result.decision == Decision.AcceptAlternative:
            decision_str = "P0LessThanP1"
            time_of_decision = test_result.info["result_for_alternative"].info["Time"]
        elif test_result.decision == Decision.AcceptNull:
            decision_str = "P0MoreThanP1"
            time_of_decision = test_result.info["result_for_null"].info["Time"]
        else:
            decision_str = "FailToDecide"
            time_of_decision = None
        return decision_str, time_of_decision


    def sequential_graphical_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha_per_hypothesis=None, weighted_G=None, verbose=False, bernoulli=False):
        '''
        Similar to graphical_multitest but we will run the graphical test sequentially after each new p-value is computed and update the graph and alpha budgets accordingly.
        '''
        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        num_hypotheses = len(ordered_hypotheses_policy_indices)
        policy_evals = {i: [] for i in range(policy_data.shape[1])}
        print(f"Running means: ", np.nanmean(policy_data, axis=0), "\n")

        if alpha_per_hypothesis is not None:
            self.set_params(alpha=alpha_per_hypothesis)
        if weighted_G is not None:
            self.set_params(G = weighted_G)
        init_graph = self.G.copy()
        rejected = [] # indices of rejected hypotheses
        alpha_at_rejected = []
        graphs_over_time = []

        # Initialize p-values for all hypotheses to 1 (not rejected)
        p_values = np.ones(num_hypotheses)
        tests = dict()
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            # Getting data before running graphical test:
            if alpha_per_hypothesis is not None:
                alpha_i = alpha_per_hypothesis[i]
            else:
                alpha_i = self.alpha[i] # fraction of alpha allocated to this hypothesis
            if not bernoulli:
                nsm_test = self.initialize_graphical_nsm_test_partial_progress(Nmax, alpha_i)
            else:
                nsm_test = self.initialize_graphical_nsm_test(Nmax, alpha_i)
            tests[i] = nsm_test
        
        #################
        # Initial diagnostic and alpha:
        if verbose:
            print("============================================================")
            print("Initial graph G:\n", self.G)
            print("Initial alpha budgets: ", self.alpha)
            print("============================================================\n")
        
        # Have to fix this to only gather data for "active" hypotheses (here, hypotheses with value > alpha / N, e.g., over Bonferroni)
        hypotheses_completed = np.zeros(num_hypotheses)
        hypotheses_correct = np.zeros(num_hypotheses)

        for k in range(len(policy_data)): # sequentially go through data points
            all_rejected = [i for i in range(num_hypotheses) if i in rejected]
            if len(all_rejected) == num_hypotheses:
                return rejected_hypotheses, rejected, decision_times, p_values, self.G, graphs_over_time, alpha_at_rejected, hypotheses_correct
            
            for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
                if i not in rejected: # Keep collecting data
                    p0_index, p1_index = hypothesis_policy_indices
                    data0 = policy_data[:, p0_index]
                    data1 = policy_data[:, p1_index]
                    if np.isnan(data0[k]) or np.isnan(data1[k]):
                        breakpoint()
                        continue
                    nsm_test = tests[i]
                    nsm_result = nsm_test.step(data0[k], data1[k]) # updating step function
                    if nsm_result.decision == Decision.AcceptAlternative:
                        hypotheses_correct[i] = 1.0 
                    elif nsm_result.decision == Decision.AcceptNull:
                        hypotheses_correct[i] = -1.0
                    
                    p_values[i] = nsm_test._p_value # Assuming the test result contains the p-value in info dictionary
                    # if i == 2 and rejected == [0,1]:
                    #     print("Trials: ", k, "data0: ", data0[k], "data1: ", data1[k], "NSM Result: ", nsm_result, "\n")
                    #     breakpoint()

                    if verbose:
                        if k % 50 == 0:
                            print(
                                "\n"
                                "============================================================\n"
                                f"  Iteration / Time : {k:<5d}    Hypothesis : H{i}\n"
                                f"  Policy indices   : {hypothesis_policy_indices}\n"
                                "------------------------------------------------------------\n"
                                f"  alpha            : {self.alpha[i]:.4e}\n"
                                f"  p-value          : {p_values[i]:.4e}\n"
                                f"  samples          : ({len(policy_evals[p0_index])}, "
                                f"{len(policy_evals[p1_index])})\n"
                                f"  running means    : ({np.nanmean(data0[:k]):.4f}, "
                                f"{np.nanmean(data1[:k]):.4f})\n"
                                "------------------------------------------------------------\n"
                                f"  Rejected         : {rejected}\n"
                                "------------------------------------------------------------\n"
                                f"  Current graph G:\n{self.G}\n"
                                "============================================================\n"
                            )
                            

            candidates = [i for i in range(self.num_hypotheses) if i not in rejected and p_values[i] <= self.alpha[i]]
            if verbose:
                print(f"At time {k}, p-values: {p_values}, alpha: {self.alpha}, candidates for rejection: {candidates}, collected data for all not in {rejected}")

            if len(candidates) > 0:
                # Plot graphs:
                rejected_at_k, graph_over_time, new_alpha_at_rejected = self.sequential_graphical_test(p_values, verbose=False)
                graphs_over_time.extend(graph_over_time)
                for j in range(len(new_alpha_at_rejected)):
                    alpha_at_rejected.append(new_alpha_at_rejected[j])
                
                for i in rejected_at_k:
                    if i not in rejected:
                        rejected.append(i)
                        hypothesis_policy_indices = ordered_hypotheses_policy_indices[i]
                        if hypothesis_policy_indices not in rejected_hypotheses:
                            rejected_hypotheses.append(hypothesis_policy_indices)
                            decision_times[hypothesis_policy_indices] = k # or nsm_time_of_decision, but we will just use k for now since it's more intuitive to say we made the decision at time k when we see the data point at time k.

                # Reset alpha for remaining hypotheses to initial alpha (or alpha_per_hypothesis if provided) after each update:
                for i in range(num_hypotheses):
                    if i not in rejected:
                        nsm_test = tests[i]
                        nsm_test.set_alpha(self.alpha[i])
                      
        return rejected_hypotheses, rejected, decision_times, p_values, self.G, graphs_over_time, alpha_at_rejected, hypotheses_correct            
                        
        