"""Inject Kaggle log summary stats into local HPO result JSON files.

For retention rates computed on Kaggle that are missing locally, this script
creates/updates JSON files with:
  - Regenerated hp_configs (same LHC seed=42 used on Kaggle)
  - robustness_stats from the log
  - Empty hp_results (violin/heatmap plots will skip these entries;
    yield curves, delta-yield, homophily scatter all work fully)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

REPO_ROOT = Path(__file__).resolve().parent.parent if '__file__' in dir() else Path('.')
RESULTS_DIR = REPO_ROOT / 'results' / 'hpo'

# ── HP space (must match run_hpo_robustness.py exactly) ───────────────────────
LR_MIN, LR_MAX   = 1e-4, 1e-1
WD_MIN, WD_MAX   = 0.0,  5e-2
DO_MIN, DO_MAX   = 0.0,  0.9
HIDDEN_CHOICES   = [8, 16, 32, 64, 128, 256]
LAYER_CHOICES    = [1, 2, 3, 4]

def sample_hp_configs(n=100, seed=42):
    sampler = LatinHypercube(d=5, seed=seed)
    samples = sampler.random(n=n)
    configs = []
    for u_lr, u_wd, u_do, u_hid, u_lay in samples:
        lr    = 10 ** (np.log10(LR_MIN) + u_lr * (np.log10(LR_MAX) - np.log10(LR_MIN)))
        wd    = WD_MIN + u_wd * (WD_MAX - WD_MIN)
        do    = DO_MIN + u_do * (DO_MAX - DO_MIN)
        hi    = min(int(u_hid * len(HIDDEN_CHOICES)), len(HIDDEN_CHOICES) - 1)
        la    = min(int(u_lay * len(LAYER_CHOICES)),  len(LAYER_CHOICES)  - 1)
        configs.append({'lr': float(lr), 'weight_decay': float(wd), 'dropout': float(do),
                        'hidden_channels': HIDDEN_CHOICES[hi], 'num_layers': LAYER_CHOICES[la]})
    return configs


def parse_log(log_text: str) -> dict:
    """Parse Kaggle log → {dataset: {metric: {r: {meta, stats}}}}."""
    # Match block headers
    ds_metric_re = re.compile(r'#\s*DATASET:\s*(\w+)\s*\|\s*Metric:\s*(\S+)')
    r_header_re  = re.compile(r'r=([\d.]+)\s+\((\d[\d,]*)\s+edges,\s+eff_h=([\d.]+)\)')
    r_done_re    = re.compile(r'r=([\d.]+)\s+DONE\s+std=([\d.]+)\s+yield_within=([\d.]+)%\s+yield_vs_dense=([\d.]+)%\s+gain=([\d.]+)x')
    homophily_re = re.compile(r'h=([\d.]+)')
    running_re   = re.compile(r'\[100/100\]\s+running mean acc=([\d.]+)')
    
    data = {}
    cur_ds = cur_metric = None
    cur_r = cur_eff_h = cur_n_edges = None
    cur_h = None
    
    for line in log_text.splitlines():
        m = ds_metric_re.search(line)
        if m:
            cur_ds = m.group(1).lower()
            cur_metric = m.group(2)
            if cur_ds not in data:
                data[cur_ds] = {}
            if cur_metric not in data[cur_ds]:
                data[cur_ds][cur_metric] = {}
            continue
        
        if cur_ds is None:
            continue
        
        # Homophily from dataset info line
        if 'nodes' in line and 'edges' in line and 'h=' in line:
            m = homophily_re.search(line)
            if m:
                cur_h = float(m.group(1))
        
        # r header
        m = r_header_re.search(line)
        if m:
            cur_r = m.group(1)
            cur_n_edges = int(m.group(2).replace(',', ''))
            cur_eff_h = float(m.group(3))
            continue
        
        # running mean at 100/100
        m = running_re.search(line)
        if m and cur_r:
            if cur_r not in data[cur_ds][cur_metric]:
                data[cur_ds][cur_metric][cur_r] = {}
            data[cur_ds][cur_metric][cur_r]['acc_mean'] = float(m.group(1))
        
        # DONE line
        m = r_done_re.search(line)
        if m and cur_ds and cur_metric:
            r = m.group(1)
            entry = data[cur_ds][cur_metric].setdefault(r, {})
            entry.update({
                'n_edges':           cur_n_edges,
                'eff_h':             cur_eff_h,
                'acc_std':           float(m.group(2)),
                'yield_rate_within': float(m.group(3)) / 100.0,
                'yield_rate_vs_dense': float(m.group(4)) / 100.0,
                'robustness_gain':   float(m.group(5)),
                'homophily':         cur_h,
            })
    
    return data


def write_json_files(parsed: dict, results_dir: Path, hp_configs: list):
    """Create/update JSON files from parsed log data."""
    RETENTION_RATES = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    for dataset, metrics in parsed.items():
        for metric, r_data in metrics.items():
            path = results_dir / f'{dataset}_hpo_{metric}.json'
            
            # Determine homophily and num_classes from existing file or parsed data
            homophily = None
            num_features = num_classes = None
            existing_results = {}
            
            if path.exists():
                existing = json.load(open(path))
                homophily = existing['meta'].get('homophily')
                num_features = existing['meta'].get('num_features')
                num_classes  = existing['meta'].get('num_classes')
                existing_results = existing.get('results', {})
            
            if homophily is None:
                # Try to get from parsed data
                for r, entry in r_data.items():
                    if entry.get('homophily'):
                        homophily = entry['homophily']
                        break
            
            # Build best_dense_acc: take from r=1.0 if available
            best_dense_acc = None
            if '1.0' in existing_results:
                best_dense_acc = existing_results['1.0']['meta'].get('best_dense_acc')
            
            # Build results dict: start with existing, then overwrite/add from log
            results = dict(existing_results)
            
            for r_str, entry in r_data.items():
                r_float = float(r_str)
                
                # Skip if already fully computed in existing file
                if r_str in results and results[r_str].get('hp_results'):
                    print(f'  {dataset}/{metric} r={r_str}: skipping (already has {len(results[r_str]["hp_results"])} hp_results)')
                    continue
                
                stats = {
                    'acc_mean':            entry.get('acc_mean'),
                    'acc_std':             entry.get('acc_std'),
                    'acc_iqr':             None,  # not in log
                    'acc_best':            None,  # not in log
                    'acc_worst':           None,  # not in log
                    'yield_rate_within':   entry.get('yield_rate_within'),
                    'yield_rate_vs_dense': entry.get('yield_rate_vs_dense'),
                    'robustness_gain':     entry.get('robustness_gain'),
                }
                
                meta_block = {
                    'retention_ratio': r_float,
                    'n_edges':         entry.get('n_edges'),
                    'best_dense_acc':  best_dense_acc,
                    'effective_h':     entry.get('eff_h'),
                }
                
                results[r_str] = {
                    'meta':             meta_block,
                    'hp_results':       [],  # empty — no per-config data from log
                    'robustness_stats': stats,
                }
                print(f'  {dataset}/{metric} r={r_str}: injected from log')
            
            # Write file
            payload = {
                'meta': {
                    'dataset':         dataset,
                    'homophily':       homophily,
                    'n_hp_configs':    100,
                    'n_seeds':         3,
                    'metric':          metric,
                    'retention_rates': RETENTION_RATES,
                    'epochs':          500,
                    'patience':        50,
                    'yield_threshold': 0.90,
                    'hp_space': {
                        'lr':    [LR_MIN, LR_MAX, 'log'],
                        'weight_decay': [WD_MIN, WD_MAX, 'linear'],
                        'dropout':      [DO_MIN, DO_MAX, 'linear'],
                        'hidden':       HIDDEN_CHOICES,
                        'num_layers':   LAYER_CHOICES,
                    },
                    'num_features': num_features,
                    'num_classes':  num_classes,
                    'source':       'kaggle_log_injection',
                },
                'hp_configs': hp_configs,
                'results':    results,
            }
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2)
            
            r_done = [r for r, v in results.items() if v.get('robustness_stats')]
            print(f'Wrote {path.name}  [{len(r_done)}/5 retention rates]')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if log_file and log_file.exists():
        log_text = log_file.read_text()
        print(f'Parsing log: {log_file}')
    else:
        print('No log file provided. Use: python inject_kaggle_log.py <logfile>')
        sys.exit(1)
    
    hp_configs = sample_hp_configs(100, seed=42)
    parsed = parse_log(log_text)
    
    print(f'\nParsed data for: {list(parsed.keys())}')
    for ds, metrics in parsed.items():
        for metric, r_data in metrics.items():
            rs = sorted(r_data.keys(), key=float)
            print(f'  {ds}/{metric}: r={rs}')
    
    print('\nWriting JSON files...')
    write_json_files(parsed, RESULTS_DIR, hp_configs)
    print('\nDone.')
