from sequentialized_barnard_tests.base import Decision, Hypothesis
from nscore.nonparametric_nsm import ContinuousNsmTest, MirroredContinuousNsmTest
from multitest.alpha_adaptive_nsm import MirroredContinuousNsmTestAlphaAdaptive
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from math import comb
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
from multitest.transivity import transitive_closure_relations
import os
from multitest.inference_modes import validate_inference_mode

class SequentialGraphicalTest:
    def __init__(
        self,
        num_policies,
        null_hypotheses,
        total_alpha=0.05,
        inference_mode="ranking_no_ties",
    ):
        # See README "Inference modes" and statistical_comparison_core.MirroredTestMixin.
        # Do not align AnyRank's ranking_no_ties default with SCC/SBT comparison.
        self.inference_mode = validate_inference_mode(inference_mode)
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
        # Off by default: per-step animation frames are huge, never-closed
        # figures that leak memory and OOM for large hypothesis counts.
        self.enable_animation = False
        # Off by default for library use: martingale plots can create many
        # caller-CWD-relative files during batch runs.
        self.enable_martingale_plots = False
        self.martingale_plot_dir = "imglib"
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
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(2),
                inference_mode=self.inference_mode,
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

    def sequential_graphical_test(self, p_values:Sequence[float], prev_rejected:List[int], hypotheses_correct:dict, hypotheses:Tuple[(int, int)], num_policies:int, p_value_types: dict, Nmax_total=None, verbose=False, allow_transitive=False) -> Tuple[Set[int], np.ndarray, np.ndarray]:
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

        rejected: Set[int] = set(prev_rejected)
        alpha_at_rejected = []
        graphs_over_time = []
        updated_hypotheses_correct = copy.deepcopy(hypotheses_correct)
        assert len(p) == self.num_hypotheses, "Mismatch in the number of hypotheses"
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("All p-values must lie in [0, 1].")

        transitive_queue = set()
        num_decisions = sum([1 for v in hypotheses_correct.values() if v != 0.0])

        while True:
            candidates = [i for i in range(self.num_hypotheses) if i not in rejected and p[i] <= self.alpha[i]]
            candidates += [
                i for i in transitive_queue
                if i not in rejected and i not in candidates
            ]

            if not candidates:
                break
            
            # Choose any i such that p_i <= delta_i"
            i = candidates[0]
            transitive_queue.discard(i)

            # Record the accepted direction before computing transitive
            # implications, so the closure below includes this rejection.
            # A direction already set (by the pairwise test's own decision or
            # by an earlier transitive implication) is kept.
            if hypotheses_correct[hypotheses[i]] == 0.0:
                direction = -1.0 if p_value_types[i] == "null" else 1.0
                hypotheses_correct[hypotheses[i]] = direction
                updated_hypotheses_correct[hypotheses[i]] = direction

            if verbose:
                print(f"Candidate hypotheses to reject at this step: {candidates}. We will reject hypothesis H{i} with p-value {p[i]:.4e} <= alpha {self.alpha[i]:.4e}")
            
            if allow_transitive:
                rejected_at_curr_step = [hypotheses[i] for i in rejected]
                hyp_correct_at_curr_step = {hypotheses[i]: hypotheses_correct[hypotheses[i]] for i in rejected}
                # for candidate in candidates:
                h = hypotheses[i]
                if h not in rejected_at_curr_step:
                    rejected_at_curr_step.append(h)
                
                if h not in hyp_correct_at_curr_step:
                    hyp_correct_at_curr_step[h] = hypotheses_correct[h]
                
                closure = transitive_closure_relations(rejected_at_curr_step, hyp_correct_at_curr_step, n_policies=num_policies)
                
                for hyp in closure["implied"]:
                    if hyp[0] == hyp[1]: # no self hypotheses
                        continue
                    # An implied edge (a, b) means mu_a < mu_b. Its direction is
                    # fixed by the closure, not by the implied pair's own
                    # (undecided) p-value type.
                    if hyp in hypotheses:
                        idx = hypotheses.index(hyp)
                        h_store = hyp
                        direction=1.0
                    else:
                        h_store = (hyp[1], hyp[0])
                        idx = hypotheses.index(h_store)
                        direction=-1.0
                    if idx not in rejected:
                        transitive_queue.add(idx)
                        updated_hypotheses_correct[h_store] = direction
                        hypotheses_correct[h_store] = direction

                        if idx not in candidates:
                            candidates.append(idx)

            # Plot graphs (animation frames only; skipped by default to avoid
            # accumulating huge, never-closed figures — an OOM at scale).
            if self.enable_animation:
                graphs_over_time.append(self.plot_graph())

            if verbose:
                print(self.G)
            
            
            # Reject i (its direction was recorded above, before the closure).
            rejected.add(i)

            alpha_at_rejected.append((hypotheses[i], copy.deepcopy(self.alpha[i])))
            active = [j for j in range(self.num_hypotheses) if j not in rejected]

            if verbose:
                # Rejecting candidate i:
                print("============================================================")
                print(f"Rejecting hypothesis H{i} with p-value {p[i]:.4e} <= alpha {self.alpha[i]:.4e}")
                print(f"Before rejection: ")
                print("Graph G: \n", self.G, "\n")
                print("alpha budgets: ", self.alpha, "\n")
            
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

                    if denom > 1e-8: 
                        new_G[k, j] = (self.G[k, j] + self.G[k, i] * self.G[i, j]) / denom
                    else:
                        new_G[k, j] = 0.0

                    if new_G[k, j] > 1.1 or new_G[k, j] < 0.0:
                        raise ValueError(
                            f"Graph update produced invalid edge weight "
                            f"G[{k}, {j}]={new_G[k, j]:.6g}"
                        )
            self.alpha = copy.deepcopy(new_alpha)
            self.G = copy.deepcopy(new_G)

            if verbose:
                print(f"After rejecting H{i}: ")
                print("Graph G: \n", self.G, "\n")
                print("alpha budgets: ", self.alpha, "\n")
                print("============================================================\n")
        num_decisions = sum([1 for v in hypotheses_correct.values() if v != 0.0])
        return rejected, graphs_over_time, alpha_at_rejected, updated_hypotheses_correct

    def initialize_graphical_nsm_test(self, Nmax, alpha):
        nonparametric_nsm_test = MirroredContinuousNsmTestAlphaAdaptive(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(2),
                inference_mode=self.inference_mode,
            )
        return nonparametric_nsm_test

    def initialize_bonferroni_test(self, Nmax, alpha, num_hypotheses):
        #bonferroni_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha/num_hypotheses)
        bonferroni_step_test = MirroredContinuousNsmTest(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha/num_hypotheses, c=np.arange(2),
                inference_mode=self.inference_mode,
            )
        return bonferroni_step_test
    
    def initialize_graphical_nsm_test_partial_progress(self, Nmax, alpha):
        nonparametric_nsm_test = MirroredContinuousNsmTestAlphaAdaptive(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(11)/10,
                inference_mode=self.inference_mode,
            )
        return nonparametric_nsm_test

    def initialize_bonferroni_test_partial_progress(self, Nmax, alpha, num_hypotheses):
        #bonferroni_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha/num_hypotheses)
        bonferroni_step_test = MirroredContinuousNsmTest(
                alternative=Hypothesis.P0LessThanP1, alpha=alpha/num_hypotheses, c=np.arange(11)/10,
                inference_mode=self.inference_mode,
            )
        return bonferroni_step_test

    def run_test_on_data(self,test, data0, data1):
        result = test.run_on_sequence(data0, data1)
        return result

    def fixed_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha, Nmax_total=None, bernoulli=False):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            if not bernoulli:
                bonferroni_step_test = self.initialize_bonferroni_test_partial_progress(Nmax, alpha, 1)
            else:
                bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha, 1)
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            valid = ~np.isnan(data0) & ~np.isnan(data1)
            result = self.run_test_on_data(bonferroni_step_test, data0[valid], data1[valid])
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = Nmax
                break
        # Note that we break after the first non-rejection to mimic a fixed-sequence testing procedure, where we only move on to the next hypothesis if we rejected the previous one. This is different from the weighted_bonferroni_multitest where we will test all hypotheses regardless of rejections, but just with different alpha levels.
        return rejected_hypotheses, decision_times

    def bonferroni_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha, Nmax_total=None, bernoulli=False):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        
        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        num_hypotheses = len(ordered_hypotheses_policy_indices)

        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            if not bernoulli:
                bonferroni_step_test = self.initialize_bonferroni_test_partial_progress(Nmax, alpha, num_hypotheses)
            else:
                bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha, num_hypotheses)
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            valid = ~np.isnan(data0) & ~np.isnan(data1)
            result = self.run_test_on_data(bonferroni_step_test, data0[valid], data1[valid])
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = "N/A"
        return rejected_hypotheses, decision_times

    def weighted_bonferroni_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha_per_hypothesis, Nmax_total=None, bernoulli=False):
        '''
        Given an ordered list of hypotheses, perform Bonferroni corrected individual tests
        Return the list of rejected hypotheses along with their decision times
        '''
        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        num_hypotheses = len(ordered_hypotheses_policy_indices)
        for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            alpha_i = alpha_per_hypothesis[i] 
            if not bernoulli:
                bonferroni_step_test = self.initialize_bonferroni_test_partial_progress(Nmax, alpha_i, num_hypotheses)
            else:
                bonferroni_step_test = self.initialize_bonferroni_test(Nmax, alpha_i, num_hypotheses)
            p0_index, p1_index = hypothesis_policy_indices
            data0 = policy_data[:, p0_index]
            data1 = policy_data[:, p1_index]
            valid = ~np.isnan(data0) & ~np.isnan(data1)
            result = self.run_test_on_data(bonferroni_step_test, data0[valid], data1[valid])
            if result.decision == Decision.AcceptAlternative:
                rejected_hypotheses.append(hypothesis_policy_indices)
                decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
            else:
                decision_times[hypothesis_policy_indices] = "N/A"
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


    def sequential_graphical_multitest(self, ordered_hypotheses_policy_indices, policy_data, Nmax, alpha_per_hypothesis=None, weighted_G=None, verbose=False, bernoulli=False, allow_transitive=True, Nmax_total=None, beta=1.0, rng=None):
        '''
        Similar to graphical_multitest but we will run the graphical test sequentially after each new p-value is computed and update the graph and alpha budgets accordingly.
        '''
        if rng is None:
            rng = np.random

        rejected_hypotheses = []
        decision_times = {hypothesis: "N/A" for hypothesis in ordered_hypotheses_policy_indices}
        num_hypotheses = len(ordered_hypotheses_policy_indices)
        policy_counts = {i: 0 for i in range(policy_data.shape[1])}
        num_policies = policy_data.shape[1]
        if verbose:
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
        martingales_history = {i: [] for i in range(num_hypotheses)}
        p_value_types = {i: "N/A" for i in range(num_hypotheses)}
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
        hypotheses_correct = {hypothesis_policy_indices: 0.0 for hypothesis_policy_indices in ordered_hypotheses_policy_indices}
        KK = np.zeros(num_hypotheses)
        
        # Active sampling scheme
        iters = int(0)
        if Nmax_total is None:
            Tmax = num_policies * Nmax + 1
        else:
            Tmax = Nmax_total + 1
        hyp_completed_vs_time = np.zeros(Tmax+1)
        while np.min(hypotheses_completed) <= 0.5 and iters < Tmax:
            # Get all rejected indices
            all_rejected = [i for i in range(num_hypotheses) if i in rejected]
            
            # Mark all rejected hypotheses as completed
            for k in range(len(all_rejected)):
                if hypotheses_completed[all_rejected[k]] <= 0.5:
                    hypotheses_completed[all_rejected[k]] = 1.0
                    if verbose:
                        print(
                            f"Evaluations: {np.array2string(KK, separator=', ')}, "
                            f"active_alpha: {np.array2string(active_alpha, precision=4, separator=', ')}, "
                            f"p_values: {np.array2string(p_values, precision=4, separator=', ')}"
                        )
            
            # Hypothesis also completed if no more samples may be drawn
            for k in range(num_hypotheses):
                if KK[k] >= Nmax:
                    hypotheses_completed[k] = 1.0
            
            
            active_alpha = copy.deepcopy(self.alpha)
            for i in range(num_hypotheses):
                if hypotheses_completed[i] > 0.5:
                    active_alpha[i] *= 0.

            # Iterate through test definitions
            # Active data collection:
            # for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
            #   if hypotheses_completed[idx] <= 0.5 and self.alpha[idx] >= (np.max(active_alpha)-1e-8):
            # INSERT ON LINE 466
            # CODE INSERTION BEGIN
            idx_to_sample = -1
            beta_selection = 2.0 # Arbitrary -- set this however you like!
            assert beta_selection > 0.0

            tmp = np.exp(beta_selection * active_alpha) - 1.0
            tmp = np.maximum(tmp, 0.0)
            if np.sum(tmp) > 0.0:
                p_select_hypothesis_to_sample = tmp / np.sum(tmp)
                assert np.isclose(np.sum(p_select_hypothesis_to_sample), 1.0)
                # Select index according to this distribution
                hypothesis_idx_to_sample = rng.choice(active_alpha.shape[0], p=p_select_hypothesis_to_sample)
            
            policy_0_index, policy_1_index = ordered_hypotheses_policy_indices[hypothesis_idx_to_sample]
            if policy_counts[policy_0_index] <= policy_counts[policy_1_index]:
                policy_to_sample = policy_0_index
            else:
                policy_to_sample = policy_1_index

            # Make sure that we are not sampling a policy that has already been sampled to the maximum number of times (Nmax)
            policy_counts[policy_to_sample] += 1

            # Assertion to double check that we are not sampling a policy that has already been sampled to the maximum number of times (Nmax)
            assert policy_counts[policy_to_sample] <= len(policy_data[:,policy_to_sample]), f"Policy {policy_to_sample} has been sampled {policy_counts[policy_to_sample]} times, but only has {len(policy_data[:,policy_to_sample])} samples available."

            # CODE INSERTION END
            for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
                if hypotheses_completed[i] <= 0.5: # Only update hypotheses that are not completed
                    p0_index, p1_index = hypothesis_policy_indices
                    
                    # Incomplete hypothesis and includes the policy we are sampling, so we will update it
                    if p0_index == policy_to_sample or p1_index == policy_to_sample:
                        data0 = policy_data[:, p0_index]
                        data1 = policy_data[:, p1_index]
                        
                        
                        if p0_index == policy_to_sample and policy_counts[p0_index] > policy_counts[p1_index]:
                            continue
                        if p1_index == policy_to_sample and policy_counts[p1_index] > policy_counts[p0_index]:
                            continue

                        # Evaluations for the hypothesis i
                        evals_for_hypothesis_i = int(KK[i])
                        # Making sure that we are not sampling a policy that has already been sampled to the maximum number of times (Nmax)
                        assert (p0_index == policy_to_sample and evals_for_hypothesis_i == policy_counts[p0_index] - 1) or (p1_index == policy_to_sample and evals_for_hypothesis_i == policy_counts[p1_index] - 1), f"Mismatch in counts for hypothesis {i}: k={evals_for_hypothesis_i}, policy_counts={policy_counts}, p0_index={p0_index}, p1_index={p1_index}"

                        # no more trials available
                        if evals_for_hypothesis_i >= len(data0) - 1 or evals_for_hypothesis_i>= len(data1)-1 or np.isnan(data0[evals_for_hypothesis_i]) or np.isnan(data1[evals_for_hypothesis_i]):
                            hypotheses_completed[i] = 1.0
                            continue
                        
                        nsm_test = tests[i]
                        nsm_result = nsm_test.step(data0[evals_for_hypothesis_i], data1[evals_for_hypothesis_i]) # updating step function
                        if nsm_result.decision == Decision.AcceptAlternative:
                            hypotheses_correct[hypothesis_policy_indices] = 1.0 
                        elif nsm_result.decision == Decision.AcceptNull:
                            hypotheses_correct[hypothesis_policy_indices] = -1.0
                        
                        p_values[i] = nsm_test._p_value # Assuming the test result contains the p-value in info dictionary
                        martingales_history[i].append(1/nsm_test._p_value)
                        p_value_types[i] = nsm_test._p_type
                        KK[i] += 1
                    
            iters+=1
            hyp_completed_vs_time[iters] = np.mean(hypotheses_completed)
            candidates = [i for i in range(self.num_hypotheses) if i not in rejected and p_values[i] <= self.alpha[i]]
            if verbose:
                print(f"At policy complexity {policy_counts}, p-values: {p_values}, alpha: {self.alpha}, candidates for rejection: {candidates}")

            if len(candidates) > 0:
                # Plot martingale history only when explicitly requested.
                if self.enable_martingale_plots:
                    plot_dir = os.path.join(
                        self.martingale_plot_dir,
                        f"alpha_{self.total_alpha}",
                    )
                    os.makedirs(plot_dir, exist_ok=True)
                    for i in candidates:
                        fig, ax = plt.subplots()
                        ax.set_ylabel("Martingale Value")
                        ax.set_xlabel("Time")
                        ax.plot(martingales_history[i], label=f"H{i}, alpha={self.alpha[i]:.4f}, p-value={p_values[i]:.4f}")
                        ax.legend()
                        plt.savefig(os.path.join(plot_dir, f"martingale_history_H{i}.png"))
                        plt.close(fig)

                rejected_at_k, graph_over_time, new_alpha_at_rejected, hypotheses_correct = self.sequential_graphical_test(p_values, rejected, hypotheses_correct, ordered_hypotheses_policy_indices, policy_data.shape[1], p_value_types, verbose=verbose, allow_transitive=allow_transitive)
                graphs_over_time.extend(graph_over_time)
                for j in range(len(new_alpha_at_rejected)):
                    alpha_at_rejected.append(new_alpha_at_rejected[j])
                
                for i in rejected_at_k:
                    if i not in rejected:
                        rejected.append(i)
                        hypothesis_policy_indices = ordered_hypotheses_policy_indices[i]
                        if hypothesis_policy_indices not in rejected_hypotheses:
                            rejected_hypotheses.append(hypothesis_policy_indices)
                            decision_times[hypothesis_policy_indices] = KK[i] # or nsm_time_of_decision, but we will just use k for now since it's more intuitive to say we made the decision at time k when we see the data point at time k.
                
                
                # Assert times to decision and number of hypotheses_correct that are not 0
                num_decided = sum([1 for v in hypotheses_correct.values() if v != 0.0])
                num_decision_times = sum([1 for v in decision_times.values() if v != "N/A"])
                if num_decided != num_decision_times and verbose:
                    print(f"Number of decided hypotheses {num_decided} does not match number of decision times {num_decision_times}")
                    for k,v in hypotheses_correct.items():
                        if k not in rejected_hypotheses:
                            print(f"Hypothesis {k} with correctness {v} and decision time {decision_times[k]}")

                # Re-sync remaining tests to the recycled graphical alpha. This
                # keeps the individual NSM decision boundary aligned with the
                # graphical p-value candidate check above.
                for i in range(num_hypotheses):
                    if i not in rejected:
                        nsm_test = tests[i]
                        nsm_test.alpha = self.alpha[i]
        # Fill any unused tail once with the final observed completion fraction.
        if iters + 1 < hyp_completed_vs_time.shape[0]:
            hyp_completed_vs_time[iters + 1:] = hyp_completed_vs_time[iters]
        return rejected_hypotheses, rejected, decision_times, p_values, self.G, graphs_over_time, alpha_at_rejected, hypotheses_correct, hyp_completed_vs_time                   
                        
        # for k in range(len(policy_data)): # sequentially go through data points
        #     all_rejected = [i for i in range(num_hypotheses) if i in rejected]
        #     if len(all_rejected) == num_hypotheses:
        #         return rejected_hypotheses, rejected, decision_times, p_values, self.G, graphs_over_time, alpha_at_rejected
        #     for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
        #         if i not in rejected: # Keep collecting data
        #             p0_index, p1_index = hypothesis_policy_indices
        #             data0 = policy_data[:, p0_index]
        #             data1 = policy_data[:, p1_index]
        #             nsm_test = tests[i]
        #             nsm_result = nsm_test.step(data0[k], data1[k]) # updating step function
        #             p_values[i] = nsm_test._p_value # Assuming the test result contains the p-value in info dictionary
        #             # nsm_decision_str, nsm_time_of_decision = self.parse_nsm_test_result(nsm_result)
        #             # if nsm_decision_str == "P0LessThanP1": # reject null
        #             #     rejected_hypotheses.append(hypothesis_policy_indices)
        #             #     decision_times[hypothesis_policy_indices] = nsm_time_of_decision if nsm_time_of_decision is not None else Nmax
            
        #     candidates = [i for i in range(self.num_hypotheses) if i not in rejected and p_values[i] <= self.alpha[i]]
        #     if verbose:
        #         print(f"At time {k}, p-values: {p_values}, alpha: {self.alpha}, candidates for rejection: {candidates}")
        #     if len(candidates) > 0:
        #         # Plot graphs:
        #         rejected_at_k, graph_over_time, new_alpha_at_rejected = self.sequential_graphical_test(p_values, verbose=False)
        #         graphs_over_time.extend(graph_over_time)
        #         for j in range(len(new_alpha_at_rejected)):
        #             alpha_at_rejected.append(new_alpha_at_rejected[j]) 
        #         for i in rejected_at_k:
        #             if i not in rejected:
        #                 rejected.append(i)
        #                 hypothesis_policy_indices = ordered_hypotheses_policy_indices[i]
        #                 if hypothesis_policy_indices not in rejected_hypotheses:
        #                     rejected_hypotheses.append(hypothesis_policy_indices)
        #                     decision_times[hypothesis_policy_indices] = k # or nsm_time_of_decision, but we will just use k for now since it's more intuitive to say we made the decision at time k when we see the data point at time k.

        #         # Reset alpha for remaining hypotheses to initial alpha (or alpha_per_hypothesis if provided) after each update:
        #         for i in range(num_hypotheses):
        #             if i not in rejected:
        #                 nsm_test = tests[i]
        #                 nsm_test.alpha = self.alpha[i]
        # return rejected_hypotheses, rejected, decision_times, p_values, self.G, graphs_over_time, alpha_at_rejected                
                        
