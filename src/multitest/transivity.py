from itertools import combinations

def transitive_closure_relations(hypotheses, decisions, n_policies=None):
    """
    Compute transitive closure of pairwise decisions.

    Parameters
    ----------
    hypotheses : list[str]
        e.g. ["0,1", "1,2", "0,2"]

    decisions : list[int]
        1  => A < B
        -1 => A > B
        0  => no decision

    Returns
    -------
    dict
    """

    if n_policies is None:
        nodes = set()
        for h in hypotheses:
            a, b = map(int, h.split(","))
            nodes.update([a, b])
        n_policies = max(nodes) + 1

    direct = set()

    for (a, b), d in decisions.items():
        if d == 1:
            direct.add((a, b))      # a < b

        elif d == -1:
            direct.add((b, a))      # b < a

    closure = set(direct)

    changed = True
    while changed:
        changed = False
        new_edges = set()

        for a, b in closure:
            for c, d in closure:
                if b == c and (a, d) not in closure:
                    new_edges.add((a, d))

        if new_edges:
            closure |= new_edges
            changed = True

    implied = closure - direct

    contradictions = []

    for i, j in combinations(range(n_policies), 2):
        if (i, j) in closure and (j, i) in closure:
            contradictions.append((i, j))

    all_pairs = set(combinations(range(n_policies), 2))

    resolved = set()
    unresolved = set()

    for i, j in all_pairs:
        if (i, j) in closure or (j, i) in closure:
            resolved.add((i, j))
        else:
            unresolved.add((i, j))

    return {
        "direct": direct,
        "implied": implied,
        "closure": closure,
        "resolved": resolved,
        "unresolved": unresolved,
        "contradictions": contradictions,
    }


def transitive_decision_times(hypothesis_times, decisions, n_policies):
    """
    Compute earliest time each ordering becomes known.

    Parameters
    ----------
    hypothesis_times : dict
        {"0,1": 10, "1,2": 20, ...}

    decisions : dict
        {"0,1": 1, "1,2": 1, ...}

    Returns
    -------
    dict
        For each pair:
            direction
            earliest_time
            direct_time
            saved_trials
    """

    INF = float("inf")

    dist = [[INF] * n_policies for _ in range(n_policies)]

    direct_times = {}

    for h, d in decisions.items():

        if d == 0:
            continue

        a, b = h
        t = hypothesis_times[h]

        if d == 1:
            dist[a][b] = t
            direct_times[(a, b)] = t

        elif d == -1:
            dist[b][a] = t
            direct_times[(b, a)] = t

    #
    # Floyd-Warshall transitive timing
    #
    for k in range(n_policies):
        for i in range(n_policies):
            for j in range(n_policies):

                if dist[i][k] == INF or dist[k][j] == INF:
                    continue

                implied_time = max(dist[i][k], dist[k][j])

                if implied_time < dist[i][j]:
                    dist[i][j] = implied_time

    results = {}

    for i, j in combinations(range(n_policies), 2):

        direct_time = hypothesis_times.get(f"{i},{j}", None)

        if dist[i][j] < INF:

            earliest_time = dist[i][j]

            results[(i, j)] = {
                "direction": 1,
                "earliest_time": earliest_time,
                "direct_time": direct_time,
                "saved_trials":
                    0 if direct_time is None
                    else max(0, direct_time - earliest_time)
            }

        elif dist[j][i] < INF:

            earliest_time = dist[j][i]

            results[(i, j)] = {
                "direction": -1,
                "earliest_time": earliest_time,
                "direct_time": direct_time,
                "saved_trials":
                    0 if direct_time is None
                    else max(0, direct_time - earliest_time)
            }

        else:
            results[(i, j)] = {
                "direction": 0,
                "earliest_time": None,
                "direct_time": direct_time,
                "saved_trials": 0,
            }

    return results


if __name__ == "__main__": 
    hypotheses = [(0,1), (3,1), (0,2)]
    decision_dict = {
        (0,1): 1,
        (3,1): -1,
    }

    closure = transitive_closure_relations(
        hypotheses,
        decision_dict,
        n_policies=4,
    )

    print("Implied relations:")
    print(closure["implied"])
    breakpoint()

    times = {
        (0,1): 10,
        (1,3): 20,
        (1,2): 15,
        (0,2): 30,
    }

    decision_dict = {
        (0,1): 1,
        (1,3): 1,
        (0,2): -1,
    }

    times = {
        (2,5): 12,
        (1,2): 25,
        (4,5): 125,
        (1,4): 86,
        (0,5): 11,
        (0,1): 15,
        (3,5): 123,
        (1,3): 73,
        (2,3): 10,
        (3,4): 150,
      }

    decision_dict = {
        (2,5): -1.0,
        (1,2): 1.0,
        (4,5): -1.0,
        (1,4): 1.0,
        (0,5): -1.0,
        (0,1): -1.0,
        (3,5): -1.0,
        (1,3): 1.0,
        (2,3): -1.0,
      }
    timing = transitive_decision_times(
        times,
        decision_dict,
        n_policies=6,
    )

    print(timing)
    breakpoint()
    print(timing)
