import numpy as np
from sequentialized_barnard_tests import StepTest
from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.nonparametric_nsm import MirroredContinuousNsmTest
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from math import comb
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class GraphicalTest:
    def __init__(self, num_policies, total_alpha=0.05):
        self.total_alpha = total_alpha # total alpha is the family-wise error rate we want to control, so we will be working with 1 - total_alpha as the error budget to allocate
        self.num_policies=num_policies # number of policies
        self.num_hypotheses = comb(self.num_policies,2) # number of hypotheses

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
        self.check_params()
    
    def set_proportional_weights(self, beta=1):
        # Set edge weights proportional to alpha_j^beta, where beta is a tunable parameter. Higher initial alpha budget gets higher weight
        for i in range(self.num_hypotheses):
            for j in range(self.num_hypotheses):
                if i != j:
                    self.G[i,j] = self.alpha[j]**beta
            # Normalize row i to sum to 1 (if not all zero)
            row_sum = np.sum(self.G[i])
            if row_sum > 0:
                self.G[i] /= row_sum
        self.check_params()

    def check_params(self, G: Optional = None, alpha:Optional = None, total_alpha:Optional = None):
        if G:
            self.G = np.asarray(G)
        if alpha:
            self.alpha = np.asarray(alpha)
        if total_alpha:
            self.total_alpha = total_alpha
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

    def plot_graph(self, hypothesis_labels: Optional[List[str]] = None, ax=None, title: str = "Graphical Test Graph"):
        """
        Plot self.G as a weighted directed graph.

        Parameters
        ----------
        hypothesis_labels : list of str, optional
            Node labels. Defaults to "H0", "H1", ... with alpha budgets shown.
        ax : matplotlib Axes, optional
            Axes to draw on. Creates a new figure if None.
        title : str
            Plot title.
        """
        if hypothesis_labels is None:
            hypothesis_labels = [f"H{i}\n(α={self.alpha[i]:.3f})" for i in range(self.num_hypotheses)]

        DG = nx.DiGraph()
        DG.add_nodes_from(range(self.num_hypotheses))
        for i in range(self.num_hypotheses):
            for j in range(self.num_hypotheses):
                if i != j and self.G[i, j] > 0:
                    DG.add_edge(i, j, weight=self.G[i, j])

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, self.num_hypotheses * 1.5), max(5, self.num_hypotheses * 1.2)))

        pos = nx.circular_layout(DG)
        nx.draw_networkx_nodes(DG, pos, ax=ax, node_size=1800, node_color="steelblue", alpha=0.85)
        nx.draw_networkx_labels(DG, pos, labels={i: hypothesis_labels[i] for i in range(self.num_hypotheses)},
                                ax=ax, font_size=8, font_color="white")

        edge_weights = nx.get_edge_attributes(DG, "weight")
        nx.draw_networkx_edges(DG, pos, ax=ax, edgelist=list(edge_weights.keys()),
                               width=[v * 4 for v in edge_weights.values()],
                               edge_color=list(edge_weights.values()),
                               edge_cmap=plt.cm.Oranges, edge_vmin=0, edge_vmax=1,
                               arrows=True, arrowsize=20,
                               connectionstyle="arc3,rad=0.15")
        edge_labels = {(i, j): f"{w:.2f}" for (i, j), w in edge_weights.items()}
        nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, ax=ax, font_size=7, label_pos=0.3)

        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        return ax

    def graphical_test(self, p_values:Sequence[float]) -> Tuple[Set[int], np.ndarray, np.ndarray]:
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

        assert len(p) == self.num_hypotheses, "Mismatch in the number of hypotheses"
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("All p-values must lie in [0, 1].")

        while True:
            candidates = [i for i in range(self.num_hypotheses) if i not in rejected and p[i] <= self.alpha[i]]
            if not candidates:
                break

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

                    denom = 1.0 - self.G[j, i] * self.G[i, j]
                    if np.isclose(denom, 0.0):
                        raise ZeroDivisionError(
                            f"Denominator became zero while updating edge ({k}, {j}). "
                            "Check whether the graph satisfies the needed conditions."
                        )

                    new_G[k, j] = (self.G[k, j] + self.G[k, i] * self.G[i, j]) / denom
            self.alpha = new_alpha
            self.G = new_G
        return rejected

def initialize_bonferroni_test(Nmax, alpha, num_hypotheses):
    #bonferroni_step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha/num_hypotheses)
    bonferroni_step_test = MirroredContinuousNsmTest(
            alternative=Hypothesis.P0LessThanP1, alpha=alpha/num_hypotheses, c=np.arange(2)
        )
    return bonferroni_step_test

def initialize_step_test(Nmax, alpha):
    step_test = StepTest(Hypothesis.P0LessThanP1, Nmax, alpha)
    return step_test

def initialize_nsm_test(Nmax, alpha):
    nonparametric_nsm_test = MirroredContinuousNsmTest(
            alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(2)
        )
    return nonparametric_nsm_test

def run_test_on_data(test, data0, data1):
    result = test.run_on_sequence(data0, data1)
    return result

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
            decision_times[hypothesis_policy_indices] = result.info["result_for_alternative"].info["Time"]
        else:
            decision_times[hypothesis_policy_indices] = Nmax
    return rejected_hypotheses, decision_times

def parse_nsm_test_result(test_result):
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

def graphical_multitest(ordered_hypotheses_policy_indices, policy_data, Nmax, alpha, either_decision=True, alpha_per_hypothesis=None):
    '''
    Given an ordered list of hypotheses, perform the graphical test
    Return the list of rejected hypotheses along with their decision times
    # We will represent a hypothesis as a policy pair (p0, p1) and by default use it to indicate H0: p0 <= p1 vs H1: p0 > p1
    '''
    rejected_hypotheses = []
    decision_times = {}
    num_hypotheses = len(ordered_hypotheses_policy_indices)
    graphical_test = GraphicalTest(num_policies=policy_data.shape[1], total_alpha=alpha)
    graphical_test.set_proportional_weights(beta=1) # Set edge weights inversely proportional to alpha budgets, so that rejecting a hypothesis with a smaller alpha budget will free up more error budget for the remaining hypotheses. This is just a heuristic and can be tuned.
    init_graph = graphical_test.G.copy()

    # Print weights to file:

    # Initialize p-values for all hypotheses to 1 (not rejected)
    p_values = np.ones(num_hypotheses)

    for i, hypothesis_policy_indices in enumerate(ordered_hypotheses_policy_indices):
        p0_index, p1_index = hypothesis_policy_indices
        data0 = policy_data[:, p0_index]
        data1 = policy_data[:, p1_index]

        # Getting data before running graphical test:
        if alpha_per_hypothesis is not None:
            alpha_i = alpha_per_hypothesis[i]
        else:
            alpha_i = graphical_test.alpha[i] # fraction of alpha allocated to this hypothesis
        nsm_test = initialize_nsm_test(Nmax, alpha_i)
        
        # Compute p-value for this hypothesis using a suitable test (e.g., t-test)
        # Here we use a placeholder function compute_p_value(data0, data1) that you would need to implement
        nsm_result = run_test_on_data(nsm_test, data0, data1)
        p_values[i] = nsm_test._p_value # Assuming the test result contains the p-value in info dictionary
        nsm_decision_str, nsm_time_of_decision = parse_nsm_test_result(nsm_result)
        # Current decision times: either the time of decision
        if either_decision:
            decision_times[hypothesis_policy_indices] = nsm_time_of_decision if nsm_time_of_decision is not None else Nmax
        else:
            # only reject null
            if nsm_result.decision == Decision.AcceptAlternative: # reject null
                decision_times[hypothesis_policy_indices] = nsm_time_of_decision
            else:
                decision_times[hypothesis_policy_indices] = Nmax

    # Graphical test will be run after all hypotheses have their p-values computed, and it will determine which hypotheses are rejected and update the graph accordingly.
    # Run the graphical test with the current p-values
    rejected_hypotheses = graphical_test.graphical_test(p_values)
    return rejected_hypotheses, decision_times, p_values, init_graph
if __name__ == "__main__":
    # Test graphical test procedure on toy data:
    graph = GraphicalTest(num_policies=3, total_alpha=0.95)
    # graph.G = np.array([[0, 0.5, 0.5],
    #                     [0.0, 0, 1],
    #                     [0.0, 0.0, 0]])
    pvalues = [0.02, 0.055, 0.012]
    rejected = graph.graphical_test(pvalues)
    print("Rejected hypotheses indices: ", rejected)