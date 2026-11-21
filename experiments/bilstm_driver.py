"""Grid on the root-disjoint development set, then three seeds per split for the chosen setting."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import sys, json, subprocess, glob
SEEDS = (20261119, 20261120, 20261121)
GRID = [(128, 2, 0.3), (256, 2, 0.3), (128, 2, 0.5)]
def run(split, H, L, d, seed):
    tag = f'{split}_H{H}_L{L}_d{d}_s{seed}'
    subprocess.run([sys.executable, str(HERE / 'bilstm.py'), split, str(H), str(L), str(d), str(seed), 'test'], check=True)
    return json.load(open(RESULTS + f'/res_bilstm_{tag}_test.json'))
grid = {}
for H, L, d in GRID:
    r = run('root', H, L, d, SEEDS[0]); grid[f'{H}_{L}_{d}'] = r['dev']['F1']
    print('GRID', H, L, d, 'dev F1', round(r['dev']['F1'], 4), 'acc', round(r['dev']['acc'], 4), flush=True)
best = max(grid, key=grid.get); H, L, d = int(best.split('_')[0]), int(best.split('_')[1]), float(best.split('_')[2])
print('BEST', best, flush=True)
for seed in SEEDS[1:]: run('root', H, L, d, seed)
for seed in SEEDS: run('random', H, L, d, seed)
json.dump({'grid_dev_F1_root': grid, 'best': best, 'seeds': SEEDS}, open(RESULTS + '/res_bilstm_grid.json', 'w'), indent=1)
print('DONE', flush=True)
