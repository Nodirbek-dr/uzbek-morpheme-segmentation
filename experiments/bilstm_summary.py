"""Summarize the BiLSTM runs: the setting chosen on the root-disjoint development set and the test scores
of the three seeds per split. Table 2 reports the mean over the seeds; the per-seed values are kept as well."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = str(ROOT / 'results')
sys.path.insert(0, str(HERE))
import json, statistics
G = json.load(open(RESULTS + '/res_bilstm_grid.json'))
H, L, d = G['best'].split('_'); seeds = G['seeds']
summary = {'config': {'hidden': int(H), 'layers': int(L), 'dropout': float(d)}, 'grid_dev_F1_root': G['grid_dev_F1_root'], 'seeds': seeds}
preds = {}
for split in ('random', 'root'):
    rs = [json.load(open(RESULTS + f'/res_bilstm_{split}_H{H}_L{L}_d{d}_s{s}_test.json')) for s in seeds]
    preds[split] = [f'pred_bilstm_{split}_H{H}_L{L}_d{d}_s{s}.json' for s in seeds]
    mean = {k: statistics.mean(r['test'][k] for r in rs) for k in ('P', 'R', 'F1', 'acc')}
    summary[split] = mean | {'per_seed': [{k: r['test'][k] for k in ('P', 'R', 'F1', 'acc', 'F1_ci', 'acc_ci')} | {'seed': s, 'best_epoch': r['best_epoch'], 'dev_F1': r['dev']['F1']}
                                          for s, r in zip(seeds, rs)]}
    f1 = [r['test']['F1'] for r in rs]; acc = [r['test']['acc'] for r in rs]
    summary[split + '_seeds'] = {'F1': f1, 'acc': acc, 'F1_mean': statistics.mean(f1), 'F1_sd': statistics.stdev(f1),
                                 'acc_mean': statistics.mean(acc), 'acc_sd': statistics.stdev(acc),
                                 'F1_range': max(f1) - min(f1), 'acc_range': max(acc) - min(acc), 'epochs': [r['best_epoch'] for r in rs]}
out = {'summary': summary, 'preds_root': preds['root'], 'preds_random': preds['random'],
       'pred_root': preds['root'][0], 'pred_random': preds['random'][0]}  # first seed, kept for scripts that need one file
json.dump(out, open(RESULTS + '/res_bilstm_main.json', 'w'), indent=1)
for split in ('random', 'root'):
    s = summary[split]; ss = summary[split + '_seeds']
    print(split, f"mean P={s['P']:.4f} R={s['R']:.4f} F1={s['F1']:.4f} Acc={100 * s['acc']:.2f}",
          f"| seeds F1={[round(x, 4) for x in ss['F1']]} acc={[round(100 * x, 2) for x in ss['acc']]} epochs={ss['epochs']}")
