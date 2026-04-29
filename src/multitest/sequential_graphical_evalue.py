from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.nonparametric_nsm import MirroredContinuousNsmTest
from sequentialized_barnard_tests.nsm_graphical_evalue import MirroredContinuousNsmTest_AlphaAdaptive
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
        return rejected, graphs_over_time

    def initialize_graphical_nsm_test(self, Nmax, alpha):
        nonparametric_nsm_test = MirroredContinuousNsmTest_AlphaAdaptive(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(2)
            )
        return nonparametric_nsm_test

    def initialize_bonferroni_test(self, Nmax, alpha, num_hypotheses):
        #bonferroni_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha/num_hypotheses)
        bonferroni_step_test = MirroredContinuousNsmTest(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha/num_hypotheses, c=np.arange(2)
            )
        return bonferroni_step_test

    def run_test_on_data(self,test, data0, data1):
        result = test.run_on_sequence(data0, data1)
        return result

    def fixed_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {}
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha, 1)
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            result = self.run_test_on_data(bonferroni_step_test, data0, data1)
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = Nmax
        return rejected_hypotheses, decision_times

    def bonferroni_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {}
        num_hypotheses = len(ordered_hypotheses_policy_indices)
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha, num_hypotheses)
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            result = self.run_test_on_data(bonferroni_step_test, data0, data1)
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = Nmax
        return rejected_hypotheses, decision_times

    def weighted_bonferroni_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha_per_hypothesis):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {}
        num_hypotheses = len(ordered_hypotheses_policy_indices)
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            alpha_i = alpha_per_hypothesis[i] 
            bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha_i, 1)
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            result = self.run_test_on_data(bonferroni_step_test, data0, data1)
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = Nmax
        return rejected_hypotheses, decision_times

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


    def sequential_graphical_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha_per_hypothesis=None, weighted_G=None, verbose=False):
        '''
        Similar to graphical_multitest but we will run the graphical test sequentially after each new p-value is computed and update the graph and alpha budgets accordingly.
        '''
        rejected_hypotheses = []
        decision_times = {}
        num_hypotheses = len(ordered_hypotheses_policy_indices)
        
        if alpha_per_hypothesis is not None:
            self.set_params(alpha=alpha_per_hypothesis)
        if weighted_G is not None:
            self.set_params(G = weighted_G)
        init_graph = self.G.copy()
        rejected = [] # indices of rejected hypotheses
        graphs_over_time = []

        # Initialize p-values for all hypotheses to 1 (not rejected)
        p_values = np.ones(num_hypotheses)
        tests = dict()
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]

            # Getting data before running graphical test:
            if alpha_per_hypothesis is not None:
                alpha_i = alpha_per_hypothesis[i]
            else:
                alpha_i = self.alpha[i] # fraction of alpha allocated to this hypothesis
            nsm_test = self.initialize_graphical_nsm_test(Nmax, alpha_i)
            tests[i] = nsm_test
        
        for k in range(len(policy_data)): # sequentially go through data points
            all_rejected = [i for i in range(num_hypotheses) if i in rejected]
            if len(all_rejected) == num_hypotheses:
                return rejected_hypotheses, rejected, decision_times, p_values, self.G, graphs_over_time
            
            for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
                if i not in rejected: # Keep collecting data
                    p0_index, p1_index = hypothesis_policy_indices
                    data0 = policy_data[:, p0_index]
                    data1 = policy_data[:, p1_index]

                    nsm_test = tests[i]
                    nsm_result = nsm_test.step(data0[k], data1[k]) # updating step function
                    p_values[i] = nsm_test._e_value # Assuming the test result contains the p-value in info dictionary
                    # nsm_decision_str, nsm_time_of_decision = self.parse_nsm_test_result(nsm_result)
                    # if nsm_decision_str == "P0LessThanP1": # reject null
                    #     rejected_hypotheses.append(hypothesis_policy_indices)
                    #     decision_times[hypothesis_policy_indices] = nsm_time_of_decision if nsm_time_of_decision is not None else Nmax
            
            candidates = [i for i in range(self.num_hypotheses) if i not in rejected and p_values[i] <= self.alpha[i]]
            if verbose:
                print(f"At time {k}, p-values: {p_values}, alpha: {self.alpha}, candidates for rejection: {candidates}")

            if len(candidates) > 0:
                # Plot graphs:
                rejected_at_k, graph_over_time = self.sequential_graphical_test(p_values, verbose=False)
                graphs_over_time.extend(graph_over_time)
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

        # If some are not rejected:          
        return rejected_hypotheses, rejected, decision_times, p_values, self.G, graphs_over_time
        