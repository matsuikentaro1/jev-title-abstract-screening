"""Calibration, confidence intervals, paired comparisons, and supplementary analyses.

Uses the saved outputs in this package only; no network connection or API key is needed.
Requires numpy and scipy (matplotlib is optional, for the calibration plot):

    python -m pip install -r requirements-additional.txt
    python scripts/additional_analyses.py

Reads data/records.csv, data/predictions.csv.gz, data/requests.csv.gz, data/runs.json,
configuration/reference_label_provenance.json, and configuration/included_reports.json.
Writes CSV files and summary_settings.json to results/additional_analyses/ (change with --output-dir).

Analyses: (1) sensitivity analysis with the original (pre-adjudication) labels and retention of the
reports ultimately included in the source reviews; (2) calibration and discrimination of Jev
probabilities; (3) record-level bootstrap 95% CIs and paired differences between Jev and each LLM
condition; (4) cutoff transfer between datasets; (5) API charges and processing time per 1000
records; (6) human-retained records excluded by Jev Noul k1 or Choice in at least one run;
(7) exclusions of human-retained records by position in the request and by request size.

Conventions follow scripts/recalculate.py: Jev retains a record if probability >= cutoff
(0.50 fixed); LLM decisions are I/E; sensitivity/specificity are computed per run and summarised
as mean (min-max) over runs.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse, csv, gzip, json, math, statistics as st
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description='Additional analyses of the saved model outputs.')
_ap.add_argument('--output-dir', type=Path, default=REPO / 'results' / 'additional_analyses')
OUT = _ap.parse_args().output_dir
OUT.mkdir(parents=True, exist_ok=True)
INCLUDED = REPO / 'configuration' / 'included_reports.json'
SEED = 20261029
NBOOT = 2000
DATASETS = ['takeshima', 'maruki']
JEV_CONDS = ['noul-k1', 'noul-k10', 'noul-k25', 'noul-k50', 'choice-v4']
LLM_CONDS = ['gpt-5-mini-minimal', 'gpt-5-mini-default', 'gpt-6-astra-default']
ALL_CONDS = JEV_CONDS + LLM_CONDS


def read_csv(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def wilson(x, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'), float('nan'))
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))


def r1(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


# ---------------------------------------------------------------- load
records = read_csv(REPO / 'data/records.csv')
ids = [r['record_id'] for r in records]
assert len(ids) == 4527
ds = {r['record_id']: r['dataset'] for r in records}
label_adj = {r['record_id']: int(r['reference_retained']) for r in records}
runs = json.loads((REPO / 'data/runs.json').read_text(encoding='utf-8'))
meta = {r['run_id']: r for r in runs}
groups = defaultdict(list)
for r in runs:
    groups[r['condition']].append(r['run_id'])
for g in groups:
    groups[g].sort(key=lambda k: meta[k]['rep'])
assert set(groups) == set(ALL_CONDS)

predictions = read_csv(REPO / 'data/predictions.csv.gz')
requests = read_csv(REPO / 'data/requests.csv.gz')
req_size = {(r['run_id'], r['request_id']): int(r['records_in_request']) for r in requests}
pred = defaultdict(dict)        # run -> record -> prob (Jev) or 'I'/'E'
position = defaultdict(dict)    # run -> record -> (position, request size)
for r in predictions:
    k, i = r['run_id'], r['record_id']
    pred[k][i] = float(r['probability_retained']) if r['probability_retained'] else r['decision']
    position[k][i] = (int(r['position_in_request']), req_size[(k, r['request_id'])])

idx = {i: n for n, i in enumerate(ids)}
dsarr = np.array([ds[i] for i in ids])
# retained matrix per condition: runs x records (bool) at 50% / categorical; prob matrix for Jev
RET = {}
PROB = {}
for g in ALL_CONDS:
    rr = groups[g]
    if g in JEV_CONDS:
        P = np.array([[pred[k][i] for i in ids] for k in rr], dtype=float)
        PROB[g] = P
        RET[g] = P >= 0.5
    else:
        RET[g] = np.array([[pred[k][i] == 'I' for i in ids] for k in rr], dtype=bool)

# ---------------------------------------------------------------- labels
prov = json.loads((REPO / 'configuration/reference_label_provenance.json').read_text(encoding='utf-8'))
changes = prov['corrections_and_confirmations']
published = [c['record_id'] for c in changes if c['origin'].startswith('published adjudication')]
related = [c['record_id'] for c in changes if c['origin'].startswith('current author-confirmed')]
assert len(published) == 13 and related == ['maruki:1240']
# Original labels: the 13 published adjudications and maruki:1240 are reverted to retained.
# takeshima:651 and maruki:794 had discrepant or missing source labels and keep their resolved labels.
label_orig = dict(label_adj)              # scenario A: revert all 14 positive->negative changes
for i in published + related:
    assert label_adj[i] == 0
    label_orig[i] = 1
label_orig13 = dict(label_adj)            # scenario B: revert only the 13 published adjudications
for i in published:
    label_orig13[i] = 1
LABELSETS = {'adjudicated': label_adj, 'original_14': label_orig, 'original_13': label_orig13}
Y = {n: np.array([L[i] for i in ids], dtype=int) for n, L in LABELSETS.items()}

mapping_rows = []
for c in changes:
    i = c['record_id']
    if i in published or i in related:
        direction = 'retained -> excluded (original 1, adjudicated 0)'
    elif i == 'takeshima:651':
        direction = 'discrepant source labels (exclude in one source file, include in the other) -> retained'
    else:
        direction = 'missing source label -> excluded'
    mapping_rows.append({'record_id': i, 'dataset': ds[i], 'adjudicated_label': label_adj[i],
                         'original_label_scenarioA_14': label_orig[i], 'original_label_scenarioB_13': label_orig13[i],
                         'origin': c['origin'], 'direction': direction})
write_csv('01a_label_change_mapping.csv', mapping_rows)

# ---------------------------------------------------------------- included reports
inc_papers = json.loads(INCLUDED.read_text(encoding='utf-8'))
inc_record = {}
for p in inc_papers:
    for i in p['record_ids']:
        inc_record[i] = p['paper']


def perrun(g, y, d):
    m = dsarr == d
    R = RET[g][:, m]
    yy = y[m]
    tp = (R & (yy == 1)).sum(1)
    fp = (R & (yy == 0)).sum(1)
    P, N = (yy == 1).sum(), (yy == 0).sum()
    return dict(tp=tp, fn=P - tp, fp=fp, tn=N - fp, P=P, N=N, sens=100 * tp / P, spec=100 * (N - fp) / N,
                retained=R.sum(1), wr=100 * (m.sum() - R.sum(1)) / m.sum())


def mmm(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), v.min(), v.max()


def fmt(v):
    a, b, c = mmm(v)
    return f'{a:.1f}' if round(b, 1) == round(c, 1) else f'{a:.1f} ({b:.1f}-{c:.1f})'


summary = {}
md = []

# ================================================================ 1. original labels
rows1 = []
for d in DATASETS:
    for g in ALL_CONDS:
        row = {'dataset': d, 'condition': g}
        for n in LABELSETS:
            x = perrun(g, Y[n], d)
            row[f'{n}_P'] = int(x['P'])
            for f in ['sens', 'spec']:
                a, b, c = mmm(x[f])
                row[f'{n}_{f}_mean'], row[f'{n}_{f}_min'], row[f'{n}_{f}_max'] = r1(a), r1(b), r1(c)
            row[f'{n}_fn_mean'] = r1(np.mean(x['fn']))
            row[f'{n}_sens_fmt'], row[f'{n}_spec_fmt'] = fmt(x['sens']), fmt(x['spec'])
        rows1.append(row)
write_csv('01b_original_label_sens_spec.csv', rows1)
# how each condition treated the reverted records
rows1c = []
for i in published + related:
    row = {'record_id': i, 'dataset': ds[i], 'scenario': 'published' if i in published else 'related_report'}
    for g in ALL_CONDS:
        row[f'{g}_runs_retained'] = f'{int(RET[g][:, idx[i]].sum())}/{RET[g].shape[0]}'
        if g in JEV_CONDS:
            row[f'{g}_mean_prob'] = r1(PROB[g][:, idx[i]].mean())
    rows1c.append(row)
write_csv('01c_reverted_records_by_condition.csv', rows1c)
# ultimately included reports: per condition, number of included papers missed per run
rows1d = []
for g in ALL_CONDS:
    for d in DATASETS:
        papers = [p for p in inc_papers if p['dataset'] == d]
        miss = []
        for r in range(RET[g].shape[0]):
            miss.append(sum(all(not RET[g][r, idx[i]] for i in p['record_ids']) for p in papers))
        rows1d.append({'condition': g, 'dataset': d, 'included_papers': len(papers),
                       'papers_missed_per_run': ','.join(map(str, miss)),
                       'included_records_all_positive_both_labelsets': all(label_adj[i] == 1 and label_orig[i] == 1 for p in papers for i in p['record_ids'])})
write_csv('01d_included_reports_check.csv', rows1d)
changed_in_included = sorted(set(inc_record) & {c['record_id'] for c in changes})
summary['1_original_labels'] = {'changed_records_in_included_reports': changed_in_included,
                                'note': 'takeshima:651 is an included-report record, retained under every label set.'}

# ================================================================ 2. calibration
EPS = 1e-6


def logistic_fit(x, y, offset=None, slope_free=True):
    """Newton-Raphson logistic regression. Returns (intercept, slope)."""
    X = np.column_stack([np.ones_like(x), x]) if slope_free else np.ones((len(x), 1))
    off = np.zeros_like(x) if offset is None else offset
    b = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ b + off
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        g = X.T @ (y - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H + 1e-10 * np.eye(len(b)), g)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def ece(p, y, bins=10):
    e = np.minimum((p * bins).astype(int), bins - 1)    # equal-width bins [0,.1),...,[.9,1.0]
    tot = 0.0
    for b in range(bins):
        m = e == b
        if m.any():
            tot += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return tot


rows2, curve = [], []
for lab in ['adjudicated', 'original_14']:
    for d in DATASETS:
        m = dsarr == d
        y = Y[lab][m]
        for g in JEV_CONDS:
            P = PROB[g][:, m]
            br, ec, sl, ic, citl, mp, auc = [], [], [], [], [], [], []
            for r in range(P.shape[0]):
                p = P[r]
                br.append(np.mean((p - y) ** 2))
                ec.append(ece(p, y))
                lp = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
                a, b = logistic_fit(lp, y)
                sl.append(b); ic.append(a)
                citl.append(logistic_fit(lp, y, offset=lp, slope_free=False)[0])
                mp.append(p.mean())
                rk = stats.rankdata(p)
                npos = y.sum(); nneg = len(y) - npos
                auc.append((rk[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
            prev = y.mean()
            row = {'label_set': lab, 'dataset': d, 'condition': g, 'n_records': int(m.sum()), 'prevalence_pct': r1(100 * prev),
                   'mean_predicted_pct': r1(100 * np.mean(mp)),
                   'brier_null_prevalence': r1(prev * (1 - prev))}
            for nm, v in [('brier', br), ('ece', ec), ('cal_slope', sl), ('cal_intercept', ic), ('citl_intercept_offset', citl), ('auc', auc)]:
                a, b, c = mmm(v)
                row[f'{nm}_mean'], row[f'{nm}_min'], row[f'{nm}_max'] = r1(a), r1(b), r1(c)
            row['brier_skill_score_mean'] = r1(1 - np.mean(br) / (prev * (1 - prev)))
            rows2.append(row)
            if lab == 'adjudicated':
                pp = P.ravel(); yy = np.tile(y, P.shape[0])
                e = np.minimum((pp * 10).astype(int), 9)
                for b in range(10):
                    mm = e == b
                    n = int(mm.sum())
                    if n == 0:
                        continue
                    pos = int(yy[mm].sum())
                    lo, hi = wilson(pos, n)
                    curve.append({'dataset': d, 'condition': g, 'bin': f'{b/10:.1f}-{(b+1)/10:.1f}', 'n_record_runs': n,
                                  'n_per_run_mean': r1(n / P.shape[0]), 'mean_predicted_pct': r1(100 * pp[mm].mean()),
                                  'observed_retained_pct': r1(100 * pos / n), 'wilson_lo_naive': r1(lo), 'wilson_hi_naive': r1(hi)})
write_csv('02a_calibration_metrics.csv', rows2)
write_csv('02b_calibration_curve_bins_pooled_runs.csv', curve)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    style = {'noul-k1': ('#0072B2', 'o', '-', 'Noul k1 (primary)'), 'choice-v4': ('#D55E00', 's', '-', 'Choice'),
             'noul-k10': ('#7f7f7f', None, ':', 'Noul k10'), 'noul-k25': ('#a0a0a0', None, ':', 'Noul k25'), 'noul-k50': ('#c0c0c0', None, ':', 'Noul k50')}
    for ax, d, title in [(axes[0], 'takeshima', 'A  Light therapy'), (axes[1], 'maruki', 'B  Adjunctive pharmacotherapy')]:
        ax.plot([0, 100], [0, 100], color='black', lw=0.8, ls='--', label='Perfect calibration')
        for g in ['noul-k10', 'noul-k25', 'noul-k50', 'noul-k1', 'choice-v4']:
            c = [r for r in curve if r['dataset'] == d and r['condition'] == g]
            col, mk, ls, lb = style[g]
            x = [r['mean_predicted_pct'] for r in c]; yv = [r['observed_retained_pct'] for r in c]
            if mk:
                ax.errorbar(x, yv, yerr=[[a - r['wilson_lo_naive'] for a, r in zip(yv, c)], [r['wilson_hi_naive'] - a for a, r in zip(yv, c)]],
                            color=col, marker=mk, ls=ls, lw=1.4, ms=4, capsize=2, label=lb)
            else:
                ax.plot(x, yv, color=col, ls=ls, lw=1, label=lb)
        ax.set_xlim(0, 100); ax.set_ylim(0, 105); ax.set_xlabel('Mean predicted probability of retention (%)')
        ax.set_ylabel('Observed human retention (%)'); ax.set_title(title, loc='left')
    axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    fig.text(0.01, 0.005, '10 equal-width bins; record-run pairs pooled across runs; error bars: Wilson 95% CI (ignores clustering by record).', fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / '02c_calibration_curves.png', dpi=200)
    plt.close(fig)
    summary['2_plot'] = '02c_calibration_curves.png'
except ImportError:
    summary['2_plot'] = 'matplotlib unavailable'

# ================================================================ 3. uncertainty
rng = np.random.default_rng(SEED)
frac = {g: RET[g].mean(0) for g in ALL_CONDS}            # per-record retention fraction across runs
major = {g: frac[g] >= 0.5 for g in ALL_CONDS}           # majority rule: retained in >=50% of runs
rows3, rows3b = [], []
boot_idx = {}
for lab in ['adjudicated', 'original_14']:
    for d in DATASETS:
        y = Y[lab]
        posi = np.where((dsarr == d) & (y == 1))[0]
        negi = np.where((dsarr == d) & (y == 0))[0]
        # stratified (by reference class) record-level bootstrap; same resamples used for all conditions (paired)
        bp = posi[rng.integers(0, len(posi), (NBOOT, len(posi)))]
        bn = negi[rng.integers(0, len(negi), (NBOOT, len(negi)))]
        boot_idx[(lab, d)] = (bp, bn)
        bs = {}
        for g in ALL_CONDS:
            sens = 100 * frac[g][posi].mean(); spec = 100 * (1 - frac[g][negi].mean())
            bsens = 100 * frac[g][bp].mean(1); bspec = 100 * (1 - frac[g][bn].mean(1))
            bs[g] = (bsens, bspec)
            tpm = int(major[g][posi].sum()); tnm = int((~major[g][negi]).sum())
            ws, wp = wilson(tpm, len(posi)), wilson(tnm, len(negi))
            r1run = RET[g][0]
            ws1, wp1 = wilson(int(r1run[posi].sum()), len(posi)), wilson(int((~r1run[negi]).sum()), len(negi))
            rows3.append({'label_set': lab, 'dataset': d, 'condition': g, 'n_pos': len(posi), 'n_neg': len(negi),
                          'sens_runavg': r1(sens), 'sens_boot_lo': r1(np.percentile(bsens, 2.5)), 'sens_boot_hi': r1(np.percentile(bsens, 97.5)),
                          'spec_runavg': r1(spec), 'spec_boot_lo': r1(np.percentile(bspec, 2.5)), 'spec_boot_hi': r1(np.percentile(bspec, 97.5)),
                          'majority_tp': tpm, 'sens_majority': r1(100 * tpm / len(posi)), 'sens_majority_wilson_lo': r1(ws[0]), 'sens_majority_wilson_hi': r1(ws[1]),
                          'majority_tn': tnm, 'spec_majority': r1(100 * tnm / len(negi)), 'spec_majority_wilson_lo': r1(wp[0]), 'spec_majority_wilson_hi': r1(wp[1]),
                          'sens_run1': r1(100 * r1run[posi].mean()), 'sens_run1_wilson': f'{ws1[0]:.1f}-{ws1[1]:.1f}',
                          'spec_run1': r1(100 * (1 - r1run[negi].mean())), 'spec_run1_wilson': f'{wp1[0]:.1f}-{wp1[1]:.1f}'})
        for ref in ['choice-v4', 'noul-k1']:
            for g in LLM_CONDS:
                row = {'label_set': lab, 'dataset': d, 'jev': ref, 'llm': g}
                for nm, sel, k in [('sens', posi, 0), ('spec', negi, 1)]:
                    if nm == 'sens':
                        a, b = major[ref][sel], major[g][sel]
                    else:
                        a, b = ~major[ref][sel], ~major[g][sel]     # correct = excluded
                    only_jev = int((a & ~b).sum()); only_llm = int((~a & b).sum())
                    diff = bs[ref][k] - bs[g][k]
                    point = (frac[ref][sel].mean() - frac[g][sel].mean()) * (100 if nm == 'sens' else -100)
                    row.update({f'{nm}_diff_runavg_pp': r1(point), f'{nm}_diff_boot_lo': r1(np.percentile(diff, 2.5)),
                                f'{nm}_diff_boot_hi': r1(np.percentile(diff, 97.5)),
                                f'{nm}_correct_jev_only': only_jev, f'{nm}_correct_llm_only': only_llm,
                                f'{nm}_mcnemar_exact_p': float(f'{mcnemar_exact(only_jev, only_llm):.3g}')})
                rows3b.append(row)
write_csv('03a_ci_sens_spec.csv', rows3)
write_csv('03b_paired_comparisons.csv', rows3b)

# ================================================================ 4. cutoff transfer


def best(g, d, y):
    m = (dsarr == d) & (y == 1)
    return int(PROB[g][:, m].min() * 100 + 1e-9) / 100      # same rule as recalculate.py


def at_cut(g, d, y, c):
    m = dsarr == d
    R = PROB[g][:, m] >= c
    yy = y[m]
    P, N = (yy == 1).sum(), (yy == 0).sum()
    tp = (R & (yy == 1)).sum(1); tn = N - (R & (yy == 0)).sum(1)
    return 100 * tp / P, 100 * tn / N, 100 * (m.sum() - R.sum(1)) / m.sum(), P - tp


rows4 = []
for lab in ['adjudicated', 'original_14']:
    for g in JEV_CONDS:
        for d in DATASETS:
            o = 'maruki' if d == 'takeshima' else 'takeshima'
            for kind, c in [('own-data cutoff', best(g, d, Y[lab])), (f'transferred from {o}', best(g, o, Y[lab])), ('fixed 50%', 0.5)]:
                s, sp, wr, fn = at_cut(g, d, Y[lab], c)
                rows4.append({'label_set': lab, 'condition': g, 'applied_to': d, 'cutoff_type': kind, 'cutoff': c,
                              'sens': fmt(s), 'spec': fmt(sp), 'work_reduction_pct': fmt(wr), 'fn_mean': r1(np.mean(fn)),
                              'runs_full_sensitivity': int((fn == 0).sum())})
write_csv('04_cutoff_transfer.csv', rows4)

# ================================================================ 5. cost / time per 1000
rows5 = []
cost_run = defaultdict(float)
for q in requests:
    cost_run[q['run_id']] += (int(q['input_tokens']) * float(q['input_usd_per_million']) + int(q['output_tokens']) * float(q['output_usd_per_million'])) / 1e6
for g in ALL_CONDS:
    c = [cost_run[k] for k in groups[g]]; w = [meta[k]['wall_seconds'] for k in groups[g]]
    a, b, cc = mmm(c); wa, wb, wc = mmm(w)
    rows5.append({'condition': g, 'runs': len(c), 'cost_per_run_usd_mean': r1(a), 'cost_per_1000_usd_mean': r1(a / 4.527),
                  'cost_per_1000_usd_min': r1(b / 4.527), 'cost_per_1000_usd_max': r1(cc / 4.527),
                  'wall_s_per_run_mean': r1(wa), 'seconds_per_1000_mean': r1(wa / 4.527), 'seconds_per_1000_min': r1(wb / 4.527),
                  'seconds_per_1000_max': r1(wc / 4.527), 'cost_ratio_vs_choice': None})
ch = [r for r in rows5 if r['condition'] == 'choice-v4'][0]['cost_per_1000_usd_mean']
for r in rows5:
    r['cost_ratio_vs_choice'] = r1(r['cost_per_1000_usd_mean'] / ch)
write_csv('05_cost_time_per_1000.csv', rows5)

# ================================================================ 6. false-negative list
rows6 = []
for i in ids:
    if label_adj[i] != 1:
        continue
    j = idx[i]
    ex = {g: int((~RET[g][:, j]).sum()) for g in ['noul-k1', 'choice-v4']}
    if max(ex.values()) == 0:
        continue
    row = {'record_id': i, 'dataset': ds[i]}
    for g in ['noul-k1', 'choice-v4']:
        row[f'{g}_runs_excluded_of10'] = ex[g]
        row[f'{g}_mean_prob'] = r1(PROB[g][:, j].mean())
        row[f'{g}_prob_range'] = f'{PROB[g][:, j].min():.2f}-{PROB[g][:, j].max():.2f}'
    for g in ['noul-k10', 'noul-k25', 'noul-k50']:
        row[f'{g}_runs_excluded_of10'] = int((~RET[g][:, j]).sum())
    for g in LLM_CONDS:
        row[f'{g}_runs_excluded_of3'] = int((~RET[g][:, j]).sum())
    row['ultimately_included_report'] = inc_record.get(i, '')
    row['abstract_missing'] = [r['abstract_missing'] for r in records if r['record_id'] == i][0]
    rows6.append(row)
rows6.sort(key=lambda r: (r['dataset'], -r['noul-k1_runs_excluded_of10'], -r['choice-v4_runs_excluded_of10']))
write_csv('06_false_negatives_jev_k1_choice.csv', rows6)

# ================================================================ 7. batching
rows7a, rows7b, rows7c = [], [], []
for g in ['noul-k10', 'noul-k25', 'noul-k50']:
    rr = groups[g]
    for d in DATASETS:
        posids = [i for i in ids if ds[i] == d and label_adj[i] == 1]
        allids = [i for i in ids if ds[i] == d]
        # (a) FN rate among positive record-runs by position and by request size
        obs = []
        for r, k in enumerate(rr):
            for i in posids:
                pos, size = position[k][i]
                obs.append((pos, size, pos / size, not RET[g][r, idx[i]], PROB[g][r, idx[i]]))
        obs = np.array(obs, dtype=float)
        k_max = int(g.split('k')[-1])
        pbins = [(1, 1), (2, 5), (6, 10), (11, 25), (26, 50)]
        for lo, hi in pbins:
            if lo > k_max:
                continue
            m = (obs[:, 0] >= lo) & (obs[:, 0] <= hi)
            if m.sum():
                rows7a.append({'condition': g, 'dataset': d, 'stratifier': 'position_in_request', 'level': f'{lo}-{hi}',
                               'positive_record_runs': int(m.sum()), 'fn': int(obs[m, 3].sum()), 'fn_rate_pct': r1(100 * obs[m, 3].mean()),
                               'mean_prob': r1(obs[m, 4].mean())})
        for nm, m in [('full-size request', obs[:, 1] == k_max), ('smaller (tail) request', obs[:, 1] < k_max)]:
            if m.sum():
                rows7a.append({'condition': g, 'dataset': d, 'stratifier': 'request_size', 'level': nm,
                               'positive_record_runs': int(m.sum()), 'fn': int(obs[m, 3].sum()), 'fn_rate_pct': r1(100 * obs[m, 3].mean()),
                               'mean_prob': r1(obs[m, 4].mean())})
        # logistic: FN ~ relative position (positives only), descriptive
        if obs[:, 3].sum() > 0 and obs[:, 3].sum() < len(obs):
            a, b = logistic_fit(obs[:, 2] - 0.5, obs[:, 3])
        else:
            a, b = float('nan'), float('nan')
        # (b) within-record association of probability with relative position, all records, record-demeaned
        P = PROB[g][:, [idx[i] for i in allids]]
        rel = np.array([[position[k][i][0] / position[k][i][1] for i in allids] for k in rr])
        Pd = P - P.mean(0); Rd = rel - rel.mean(0)
        slope_all = float((Pd * Rd).sum() / (Rd ** 2).sum())
        pm = np.array([label_adj[i] == 1 for i in allids])
        slope_pos = float((Pd[:, pm] * Rd[:, pm]).sum() / (Rd[:, pm] ** 2).sum())
        rho, pval = stats.spearmanr(Rd[:, pm].ravel(), Pd[:, pm].ravel())
        # (c) batch-specific FN: positives excluded in >=1 batched run but retained in all k1 runs
        k1 = RET['noul-k1']
        spec_fn = [i for i in posids if (~RET[g][:, idx[i]]).any() and k1[:, idx[i]].all()]
        shared_fn = [i for i in posids if (~RET[g][:, idx[i]]).any() and (~k1[:, idx[i]]).any()]
        rows7b.append({'condition': g, 'dataset': d, 'positive_record_runs': len(obs), 'fn_record_runs': int(obs[:, 3].sum()),
                       'logit_FN_per_unit_relpos_coef': r1(b), 'within_record_slope_prob_vs_relpos_all_records': r1(slope_all),
                       'within_record_slope_prob_vs_relpos_positives': r1(slope_pos), 'spearman_positives_demeaned': r1(rho), 'spearman_p': r1(pval),
                       'positives_ever_FN_batched': len(spec_fn) + len(shared_fn), 'of_which_also_FN_in_k1': len(shared_fn),
                       'batch_specific_FN_records': ';'.join(spec_fn),
                       'mean_prob_k1_positives': r1(PROB['noul-k1'][:, [idx[i] for i in posids]].mean()),
                       'mean_prob_batched_positives': r1(PROB[g][:, [idx[i] for i in posids]].mean())})
        for i in posids:
            j = idx[i]
            exc = ~RET[g][:, j]
            if exc.any():
                rows7c.append({'condition': g, 'dataset': d, 'record_id': i, 'runs_excluded_of10': int(exc.sum()),
                               'k1_runs_excluded_of10': int((~k1[:, j]).sum()),
                               'mean_prob_batched': r1(PROB[g][:, j].mean()), 'mean_prob_k1': r1(PROB['noul-k1'][:, j].mean()),
                               'positions_when_excluded': ','.join(f'{position[k][i][0]}/{position[k][i][1]}' for r, k in enumerate(rr) if exc[r]),
                               'positions_when_retained': ','.join(f'{position[k][i][0]}/{position[k][i][1]}' for r, k in enumerate(rr) if not exc[r])})
write_csv('07a_batching_fn_by_position_size.csv', rows7a)
write_csv('07b_batching_summary.csv', rows7b)
write_csv('07c_batching_fn_records.csv', rows7c)

summary['settings'] = {'bootstrap_reps': NBOOT, 'seed': SEED, 'bootstrap': 'record-level, stratified by reference class within dataset; statistic = run-averaged sensitivity/specificity (mean over records of the per-record retention fraction across runs); percentile 95% CI; identical resamples for all conditions (paired differences)',
                       'majority_rule': 'record retained if retained in >=50% of runs (Jev: >=5/10; LLM: >=2/3)',
                       'ece': '10 equal-width bins [0,0.1),...,[0.9,1.0]; weighted mean |observed - mean predicted|; per run, then mean/range',
                       'calibration_slope': 'logistic regression of reference label on logit(p), p clipped to [1e-6,1-1e-6], per run; CITL = intercept with logit(p) as offset'}
(OUT / 'summary_settings.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print('done ->', OUT)
