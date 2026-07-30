import argparse
import json
import os
import re
import numpy as np
from pathlib import Path


METHOD_LABELS = {
    'graphical':           'Graphical (ours)',
    'bonferroni':          'Bonferroni',
    'weighted_bonferroni': 'Weighted Bonferroni',
    'fixed':               'Fixed Sequence',
    'graphical_active':    'Graphical Active',
    'evalues':             'E-values (passive)',
    'evalues_active':      'E-values (active)',
}

EXP_LABELS = {
    'roboarena4':         'RoboArena-4',
    'roboarena7':         'RoboArena-7',
    'roboarena4_wm_prior': 'RoboArena-4 (WM)',
    'roboarena7_wm_prior': 'RoboArena-7 (WM)',
}

GRAPH_LABELS = {
    'soft_masked':     'Soft-masked',
    'fully_connected': 'Fully connected',
}

# Order methods appear in table rows (if methods are rows)
METHOD_ORDER = ['graphical', 'graphical_active', 'evalues', 'evalues_active',
                'bonferroni', 'weighted_bonferroni', 'fixed']

EXP_ORDER = ['roboarena4', 'roboarena7', 'roboarena4_wm_prior', 'roboarena7_wm_prior']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_combo_dirname(dirname):
    """
    Parse a level-1 combo directory name like
    'roboarena_graphical_active_soft_masked_transitive'
    into {'method': 'graphical_active', 'graph_type': 'soft_masked'}.
    Returns None if the name doesn't match the expected pattern.
    """
    # Strip known prefix/suffix
    name = dirname
    for prefix in ('roboarena_',):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ('_transitive',):
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Known graph types (longest first to avoid partial match)
    for graph in ('soft_masked', 'fully_connected'):
        if name.endswith('_' + graph):
            method = name[: -(len(graph) + 1)]
            return {'method': method, 'graph_type': graph}
    return None


def find_result_dirs(results_dir):
    """
    Walk up to two directory levels below results_dir looking for dirs that
    contain config.json.  Returns a list of dicts:
        {'path': Path, 'method': str, 'graph_type': str, 'exp_name': str}

    Supports two layouts:
      Flat:   results_dir/{exp_dir}/config.json
      Nested: results_dir/{combo_dir}/{exp_dir}/config.json
                where combo_dir encodes method + graph_type
    """
    p = Path(results_dir)

    # Case 1: results_dir itself is a result dir
    if (p / 'config.json').exists():
        return [{'path': p, 'method': None, 'graph_type': None, 'exp_name': p.name}]

    entries = []

    for level1 in sorted(p.iterdir()):
        if not level1.is_dir():
            continue

        combo = _parse_combo_dirname(level1.name)

        # Flat layout: level1 has config.json directly
        if (level1 / 'config.json').exists():
            entries.append({
                'path':       level1,
                'method':     combo['method']    if combo else None,
                'graph_type': combo['graph_type'] if combo else None,
                'exp_name':   level1.name,
            })
            continue

        # Nested layout: level1 is a combo dir; level2 dirs are exp_name dirs
        for level2 in sorted(level1.iterdir()):
            if level2.is_dir() and (level2 / 'config.json').exists():
                entries.append({
                    'path':       level2,
                    'method':     combo['method']    if combo else None,
                    'graph_type': combo['graph_type'] if combo else None,
                    'exp_name':   level2.name,
                })

    return entries


def _parse_params_from_filename(name):
    m = re.search(r'_N(\d+)_n(\d+)_alpha([\d.]+)_beta([\d.]+)', name)
    if not m:
        return {}
    return {'N': int(m.group(1)), 'n': int(m.group(2)),
            'alpha': float(m.group(3)), 'beta': float(m.group(4))}


def _parse_sample_complexity_file(filepath):
    with open(filepath) as f:
        content = f.read()

    n_hyp_m = re.search(r'Total hypotheses: (\d+)', content)
    n_hyp = int(n_hyp_m.group(1)) if n_hyp_m else None

    patterns = {
        'graphical':           r'Our Approach \(Graphical\):\s+([\d.]+)\s+\(rejected ([\d.]+)/(\d+)\)',
        'evalues':             r'E-values \(passive\):\s+([\d.]+)\s+\(rejected ([\d.]+)/(\d+)\)',
        'evalues_active':      r'E-values \(active\):\s+([\d.]+)\s+\(rejected ([\d.]+)/(\d+)\)',
        'graphical_active':    r'Active Graphical \(p-value\):\s+([\d.]+)\s+\(rejected ([\d.]+)/(\d+)\)',
        'bonferroni':          r'Bonferroni:\s+([\d.]+)\s+\(rejected ([\d.]+)/(\d+)\)',
        'weighted_bonferroni': r'Weighted Bonferroni:\s+([\d.]+)\s+\(rejected ([\d.]+)/(\d+)\)',
        'fixed':               r'Fixed Sequence:\s+([\d.]+)\s+\(rejected ([\d.]+)/(\d+)\)',
    }

    result = {'n_hypotheses': n_hyp, 'methods': {}}
    for method, pat in patterns.items():
        m = re.search(pat, content)
        if m:
            result['methods'][method] = {
                'sample_complexity': float(m.group(1)),
                'n_rejected':        float(m.group(2)),
                'n_total':           int(m.group(3)),
            }
    return result


def _parse_rejection_counts_file(filepath):
    with open(filepath) as f:
        content = f.read()

    n_hyp_m = re.search(r'Total hypotheses: (\d+)', content)
    n_hyp = int(n_hyp_m.group(1)) if n_hyp_m else None

    result = {'n_hypotheses': n_hyp, 'methods': {}}
    for line in content.splitlines():
        m = re.match(r'(\w+): mean=([\d.]+) / (\d+)', line)
        if m:
            result['methods'][m.group(1)] = {
                'mean_rejected': float(m.group(2)),
                'n_hypotheses':  int(m.group(3)),
            }
    return result


def _parse_rankings_file(filepath):
    """Parse policy_rankings_*.txt into {method: [(policy, letters), ...]}."""
    with open(filepath) as f:
        content = f.read()

    rankings = {}
    current_method = None
    for line in content.splitlines():
        sep_match = re.match(r'={10,}', line)
        if sep_match:
            continue
        method_candidates = [m for m in METHOD_LABELS if line.strip() == m]
        if method_candidates:
            current_method = method_candidates[0]
            rankings[current_method] = []
            continue
        if current_method:
            m = re.match(r'\s+(\S+):\s+(\{.*\})', line)
            if m:
                rankings[current_method].append((m.group(1), m.group(2)))
    return rankings


def load_result_dir(result_dir, method=None, graph_type=None, exp_name=None):
    result_dir = Path(result_dir)
    data = {
        'name':       result_dir.name,
        'method':     method,
        'graph_type': graph_type,
        'exp_name':   exp_name or result_dir.name,
    }

    config_path = result_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            data['config'] = json.load(f)

    meta_path = result_dir / 'policy_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            data['policy_meta'] = json.load(f)

    sc_dir = result_dir / 'sample_complexity'
    data['sample_complexity'] = []
    if sc_dir.exists():
        for fp in sorted(sc_dir.glob('actual_sample_complexity_*.txt')):
            entry = _parse_sample_complexity_file(fp)
            entry['params'] = _parse_params_from_filename(fp.name)
            data['sample_complexity'].append(entry)

    rc_dir = result_dir / 'rejection_counts'
    data['rejection_counts'] = []
    if rc_dir.exists():
        for fp in sorted(rc_dir.glob('rejection_counts_*.txt')):
            entry = _parse_rejection_counts_file(fp)
            entry['params'] = _parse_params_from_filename(fp.name)
            data['rejection_counts'].append(entry)

    data['rankings'] = []
    for fp in sorted(result_dir.glob('policy_rankings_*.txt')):
        entry = _parse_rankings_file(fp)
        entry['params'] = _parse_params_from_filename(fp.name)
        data['rankings'].append(entry)

    return data


def load_all(results_dir):
    entries = find_result_dirs(results_dir)
    if not entries:
        raise FileNotFoundError(f'No result directories found under {results_dir}')
    all_data = []
    for e in entries:
        d = load_result_dir(
            e['path'],
            method=e['method'],
            graph_type=e['graph_type'],
            exp_name=e['exp_name'],
        )
        all_data.append(d)
    return all_data


# ---------------------------------------------------------------------------
# LaTeX table building
# ---------------------------------------------------------------------------

def build_latex_table(header_row, data_rows, caption='', label='tab:results'):
    """
    header_row : list of strings (first cell is the row-label column header)
    data_rows  : list of lists (first element is row label, rest are values)
    """
    n_data_cols = len(header_row) - 1
    col_spec = 'l' + 'r' * n_data_cols

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        ' & '.join(header_row) + r' \\',
        r'\midrule',
    ]
    for row in data_rows:
        lines.append(' & '.join(str(v) for v in row) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{' + caption + r'}',
        r'\label{' + label + r'}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Table builders (add new ones here as the schema is refined)
# ---------------------------------------------------------------------------

def _get_sc_cell(d, method, beta_filter, fmt):
    """Extract a sample-complexity cell value from a single result dict."""
    sc_entries = d.get('sample_complexity', [])
    if beta_filter is not None:
        sc_entries = [e for e in sc_entries if e.get('params', {}).get('beta') == beta_filter]
    if not sc_entries:
        return '--'
    mdata = sc_entries[0].get('methods', {}).get(method)
    if mdata is None:
        return '--'
    sc  = mdata['sample_complexity']
    rej = mdata['n_rejected']
    tot = mdata['n_total']
    return f'{sc:{fmt}} ({rej:.0f}/{tot})'


def _get_rc_cell(d, method, beta_filter):
    """Extract a rejection-count cell value from a single result dict."""
    rc_entries = d.get('rejection_counts', [])
    if beta_filter is not None:
        rc_entries = [e for e in rc_entries if e.get('params', {}).get('beta') == beta_filter]
    if not rc_entries:
        return '--'
    entry = rc_entries[0]
    mdata = entry.get('methods', {}).get(method)
    if mdata is None:
        return '--'
    return f"{mdata['mean_rejected']:.1f}/{entry['n_hypotheses']}"


def table_by_method_and_exp(all_data, metric='sample_complexity', methods=None,
                             exps=None, graph_types=None, beta_filter=None, fmt='.1f'):
    """
    Rows  = methods
    Cols  = (exp_name, graph_type) pairs  — one column per combination present in data
    Cells = sample_complexity  or  n_rejected

    all_data : list of result dicts (from load_all)
    """
    if methods is None:
        methods = METHOD_ORDER

    # Collect unique (exp_name, graph_type) pairs, in a stable order
    seen = {}
    for d in all_data:
        exp   = d.get('exp_name') or ''
        graph = d.get('graph_type') or ''
        if exps        and exp   not in exps:        continue
        if graph_types and graph not in graph_types: continue
        key = (exp, graph)
        if key not in seen:
            seen[key] = d  # first hit wins for lookup below

    # Sort columns: exp order first, then graph order
    def col_sort(key):
        exp, graph = key
        ei = EXP_ORDER.index(exp)   if exp   in EXP_ORDER   else 999
        gi = ['soft_masked', 'fully_connected'].index(graph) if graph in ['soft_masked', 'fully_connected'] else 999
        return (ei, gi)

    col_keys = sorted(seen.keys(), key=col_sort)

    # Column headers
    col_headers = []
    for exp, graph in col_keys:
        exp_lbl   = EXP_LABELS.get(exp, exp)
        graph_lbl = GRAPH_LABELS.get(graph, graph)
        col_headers.append(f'{exp_lbl} / {graph_lbl}' if graph else exp_lbl)

    header = ['Method'] + col_headers

    # Build a lookup: (exp_name, graph_type, method_from_dirname) -> result dict
    # Since each dir only ran ONE method, the method in d['method'] tells us which method's
    # actual results are present. But the file may also contain other methods if all were run.
    # We do a two-pass: first try exact (method == d['method']), then fall back to any dir
    # for that (exp, graph) that has the data.
    lookup = {}  # (exp, graph) -> list of result dicts
    for d in all_data:
        exp   = d.get('exp_name') or ''
        graph = d.get('graph_type') or ''
        lookup.setdefault((exp, graph), []).append(d)

    rows = []
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        row = [label]
        for key in col_keys:
            dirs = lookup.get(key, [])
            cell = '--'
            for d in dirs:
                if metric == 'sample_complexity':
                    c = _get_sc_cell(d, method, beta_filter, fmt)
                else:
                    c = _get_rc_cell(d, method, beta_filter)
                if c != '--':
                    cell = c
                    break
            row.append(cell)
        rows.append(row)

    return header, rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a LaTeX table from multitest experiment results.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to roboarena_outputs (or any parent of result dirs)')
    parser.add_argument('--metric', type=str, default='n_rejected',
                        choices=['sample_complexity', 'n_rejected'],
                        help='Metric to display in cells')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        choices=list(METHOD_LABELS.keys()),
                        help='Methods to include as rows (default: all in METHOD_ORDER)')
    parser.add_argument('--exps', type=str, nargs='+', default=None,
                        choices=list(EXP_LABELS.keys()),
                        help='Experiment names to include as columns (default: all found)')
    parser.add_argument('--graph_types', type=str, nargs='+', default=None,
                        choices=['soft_masked', 'fully_connected'],
                        help='Graph types to include as columns (default: all found)')
    parser.add_argument('--beta', type=float, default=None,
                        help='Filter results to this beta value')
    parser.add_argument('--fmt', type=str, default='.1f',
                        help='Number format for sample complexity values')
    parser.add_argument('--caption', type=str, default='Experimental results')
    parser.add_argument('--label', type=str, default='tab:results')
    parser.add_argument('--output', type=str, default=None,
                        help='Write LaTeX to this file instead of stdout')
    args = parser.parse_args()

    all_data = load_all(args.results_dir)

    # Print a summary of what was loaded
    print(f'Loaded {len(all_data)} result director(y/ies):')
    for d in all_data:
        sc_count = sum(len(e.get('methods', {})) for e in d.get('sample_complexity', []))
        rc_count = sum(len(e.get('methods', {})) for e in d.get('rejection_counts', []))
        print(f"  [{d['method'] or '?'} / {d['graph_type'] or '?'}] {d['exp_name']}"
              f"  sc_entries={sc_count}  rc_entries={rc_count}")

    header, rows = table_by_method_and_exp(
        all_data,
        metric=args.metric,
        methods=args.methods,
        exps=args.exps,
        graph_types=args.graph_types,
        beta_filter=args.beta,
        fmt=args.fmt,
    )

    latex = build_latex_table(header, rows, caption=args.caption, label=args.label)

    if args.output:
        Path(args.output).write_text(latex + '\n')
        print(f'LaTeX table written to {args.output}')
    else:
        print()
        print(latex)
