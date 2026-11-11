"""Generate LaTeX tables, figures and number macros from results/*.json."""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ldpq import Pop, consts
from predlib import eta_hat, pred, pred_step, eta_sides

R = 'results'
P = 'generated'
import os as _os
for _d in ('tables', 'figs'):
    _os.makedirs(f'{P}/{_d}', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'axes.linewidth': 0.6})
COL = {'nbs': '#1f4e79', 'bayess': '#c0392b', 'sgd': '#27ae60', 'hh': '#7d3c98'}
PNAME = {'nbs': r'\NBS{}', 'bayess': r'\BAY{}', 'sgd': r'\SGD{}', 'hh': r'\HH{}'}
DNAME = {'creditcard': 'CreditCard', 'flights': 'Flights', 'power': 'Power', 'pareto': 'Pareto(1.5)'}


def load(name):
    p = f'{R}/{name}.json'
    return json.load(open(p)) if os.path.exists(p) else None


from decimal import Decimal, ROUND_HALF_UP


def hu(x, d):
    """round half up (decimal), e.g. 0.975 -> 0.98"""
    x = float(x)
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP)
    if v == 0:
        v = abs(v)
    return f'{v:.{d}f}'


def exs(x):
    """exact short decimal for rates on a 1/40 or 1/200 grid (0.975 -> 0.975, 0.95 -> 0.95)"""
    v = round(float(x), 3)
    return f'{v:.2f}' if abs(v - round(v, 2)) < 1e-9 else f'{v:.3f}'


def pm(m, s, d=2, pct=False):
    if pct:
        m, s = 100 * m, 100 * s
    return f'\\pmv{{{hu(m, d)}}}{{{hu(s, d)}}}'


def hitcell(m, s):
    col = 'good' if m >= 0.9 else ('mid' if m >= 0.5 else 'bad')
    return f'\\cellcolor{{{col}}}' + pm(m, s, 2)


macros = {}
PLATEAU = {'0.99|0.25|ipa'}
TOLGAP = []
RQ4TH = {}
RQ4PRED = {}
PREDERR = []
PREDINF = []
NPRED = []
RQ4ERR = []
RQ4ERRQ = []
T3ERR = []
HC = {}
DIRDEV = []


# ------------------------------------------------------------------ datasets table
def tab_data():
    rows = []
    unit = {'creditcard': (0.05, 'EUR'), 'flights': (1, 'min'), 'power': (1, 'W'), 'pareto': (1, '--')}
    off = {'creditcard': 0, 'flights': -87, 'power': -1, 'pareto': -1}
    nraw = {'creditcard': 284807, 'flights': 200000, 'power': 2049280, 'pareto': 2000000}
    tail = {'creditcard': 'heavy', 'flights': 'moderate', 'power': 'bounded', 'pareto': 'heavy'}
    for d in ['creditcard', 'flights', 'power', 'pareto']:
        pop = Pop(d)
        xq = pop.quantile(0.99)
        u, un = unit[d]
        val = (xq + off[d]) * u if d != 'creditcard' else (xq - 1) * u  # stored value = round(EUR/0.05) + 1
        keff = min(10 * xq, pop.B) / xq
        ks = f'{hu(keff, 2)}' if keff < 9.999 else '10'
        rows.append(f"{DNAME[d]} & {nraw[d]:,} & $2^{{{int(math.log2(pop.B))}}}$ & {val:,.0f} {un if un!='--' else ''} & {xq:,} & {hu(pop.B / xq, 1)} & {ks} & {tail[d]}\\\\")
        macros[f'Head{d}'] = f'{hu(pop.B / xq, 1)}'
        macros[f'Keff{d}'] = ks
    s = r"""\begin{table}[t]
\centering
\caption{Datasets: true p99 (raw unit and value in $[B]$), headroom, and effective inflation $\kappa_{\mathrm{eff}}$ of the default target.}
\label{tab:data}
\footnotesize
\setlength{\tabcolsep}{3.6pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lrcrrccc}
\toprule
\hd dataset & records & $B$ & raw & $x_{0.99}$ & $B/x_{0.99}$ & $\kappa_{\mathrm{eff}}$ & tail\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_data.tex', 'w').write(s)


# ------------------------------------------------------------------ RQ1a figure
def fig_rq1a():
    r = load('rq1a')
    if r is None:
        return
    rc = load('rq1c')
    r40 = load('rq1c40') or {}
    if rc and r40:
        rc = {k: v for k, v in rc.items() if v['proto'] == 'nbs'}
        rc.update({k: v for k, v in r40.items() if v['n'] < 10**7})
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.7))
    kcol = {2: '#1f4e79', 5: '#e67e22', 10: '#c0392b', 50: '#27ae60'}
    for ax, proto in zip(axs[:2], ['nbs', 'bayess']):
        ks = sorted({v['kappa'] for v in r.values() if v['proto'] == proto})
        for k in ks:
            pts = sorted([(v['beta'], v['hit_mean'], v['hit_std'], v['theory']) for v in r.values() if v['proto'] == proto and v['kappa'] == k])
            b = np.array([p[0] for p in pts]) * 100
            m = np.array([p[1] for p in pts]); sd = np.array([p[2] for p in pts])
            ax.errorbar(b, m, yerr=sd, color=kcol[k], lw=1.0, ms=2.2, marker='o', capsize=1.3, elinewidth=0.6, label=f'$\\kappa={k}$')
            ax.axvline(pts[0][3] * 100, color=kcol[k], lw=0.6, ls='--')
        ax.set_title({'nbs': '(a) NBS: hit rate', 'bayess': '(b) BayeSS: hit rate'}[proto], fontsize=7.5, pad=2)
        ax.set_xlabel(r'poisoning rate $\beta$ (%)'); ax.set_xlim(-0.05, 2.05)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].legend(loc='lower right', frameon=False, ncol=1, handlelength=1.1, fontsize=6)
    ax = axs[2]
    for k in sorted({v['kappa'] for v in r.values() if v['proto'] == 'sgd'}):
        pts = sorted([(v['beta'], v['infl_mean'], v['infl_std']) for v in r.values() if v['proto'] == 'sgd' and v['kappa'] == k])
        ax.errorbar([100 * p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=kcol[k], lw=1.0, ms=2.2, marker='o',
                    capsize=1.3, elinewidth=0.6, label=f'$\\kappa={k}$')
        ax.axhline(k, color=kcol[k], lw=0.6, ls=':')
    ax.set_yscale('log'); ax.set_title('(c) SGD-SN-log: inflation $\\hat m/x_q$', fontsize=7.5, pad=2)
    ax.set_xlabel(r'poisoning rate $\beta$ (%)'); ax.grid(alpha=0.25, lw=0.4); ax.set_xlim(-0.05, 3.05)
    ax = axs[3]
    if rc:
        th = None
        sg = load('sgd1e7') or {}
        msp = CC().target(0.99, 10)
        for proto in ['nbs', 'bayess', 'sgd']:
            pts = sorted([(v['n'], v['mean'], v['std']) for v in rc.values() if v['proto'] == proto and v['mean'] < 0.0599])
            if proto == 'sgd' and 'sgd|1000000' in r40:
                macros['RqOneCsgdOne'] = f"{hu(100 * r40['sgd|1000000']['mean'], 2)}\\%"
            if proto == 'sgd' and 'sgd|10000000' in r40:
                v7 = r40['sgd|10000000']
                pts.append((10**7, v7['mean'], v7['std']))
                macros['RqOneCsgdBig'] = f"{hu(100 * v7['mean'], 2)}\\%"
            elif proto == 'sgd' and 'bisect' in sg:
                pts.append((10**7, sg['bisect']['mean'], sg['bisect']['std']))
                macros['RqOneCsgdBig'] = f"{hu(100 * sg['bisect']['mean'], 2)}\\%"
            if not pts:
                continue
            th = [v['theory'] for v in rc.values()][0]
            ax.errorbar([p[0] for p in pts], [100 * p[1] for p in pts], yerr=[100 * p[2] for p in pts], color=COL[proto], marker='o', ms=2.2,
                        lw=1.0, capsize=1.3, elinewidth=0.6, label={'nbs': 'NBS', 'bayess': 'BayeSS', 'sgd': 'SGD-SN-log'}[proto])
            ns_all = sorted(set([p_[0] for p_ in pts] + ([10**5] if proto == 'sgd' else [])))
            pp = [(n_, pred_step(CC(), 'creditcard', proto, n_, 1.0, 0.99, msp)) for n_ in ns_all]
            pp = [(a, b_) for a, b_ in pp if b_ is not None]
            ax.plot([a for a, _ in pp], [100 * b_ for _, b_ in pp], color=COL[proto], ls=':', lw=1.0)
            emp = {p_[0]: p_[1] for p_ in pts}
            for n_, b_ in pp:
                if n_ in emp:
                    NPRED.append((proto, n_, 100 * emp[n_], 100 * b_))
        ax.axhline(100 * th, color='k', lw=0.8, ls='--', label=r'$\beta^\star_0$ ($\eta\to0$)')
        pc = CC(); l9 = int(math.ceil(0.9 * msp)); F9 = float(pc.F(l9 - 1))
        npv = (F9 - 0.99) / F9
        ax.axhline(100 * npv, color='#7f8c8d', lw=0.8, ls=(0, (4, 1, 1, 1, 1, 1)), label='non-private')
        macros['NonPrivTen'] = f'{hu(100 * npv, 2)}\\%'
        ax.plot([], [], color='k', ls=':', lw=1.0, label='pred. (Rem. 1)')
        cj = load('cert') or {}
        cpts = sorted((v['n'], v['beta']) for k_, v in cj.items() if k_.startswith('cert|'))
        if cpts:
            ax.plot([a for a, _ in cpts], [100 * b_ for _, b_ in cpts], color=COL['nbs'], ls='-.', lw=1.0, label=r'cert. NBS ($\eta_{\mathrm{NBS}}$)')
        ax.set_xscale('log'); ax.set_xlabel('number of users $n$')
        from matplotlib.ticker import NullFormatter
        ax.set_yscale('log'); ax.set_yticks([0.5, 1, 2, 5]); ax.set_yticklabels(['0.5', '1', '2', '5']); ax.set_ylim(0.45, 30)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(frameon=False, fontsize=4.6, ncol=2, handlelength=1.0, columnspacing=0.5, handletextpad=0.3, loc='upper right')
        nb = sorted([(v['n'], v['mean']) for v in rc.values() if v['proto'] == 'nbs'])
        macros['RqOneCbig'] = f'{hu(100 * nb[-1][1], 2)}\\%'
        macros['RqOneCsmall'] = f'{hu(100 * nb[0][1], 2)}\\%'
        macros['RqOneCth'] = f'{hu(100 * th, 2)}\\%'
    ax.set_title('(d) $\\hat\\beta$ (%) vs. $n$', fontsize=7.5, pad=2); ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel('hit rate')
    plt.tight_layout(pad=0.2, w_pad=0.6, rect=(0.012, 0, 1, 1))
    plt.savefig(f'{P}/figs/fig_rq1.pdf')
    for proto in ['nbs', 'bayess', 'sgd']:
        pts = sorted([(v['beta'], v['hit_mean'], v['hit_std'], v['infl_mean'], v['infl_std']) for v in r.values() if v['proto'] == proto and v['kappa'] == 10])
        ok = [p[0] for p in pts if p[1] >= 0.9]
        macros[f'RqOneMin{proto}'] = f'{hu(100 * min(ok), 1)}\\%' if ok else 'n/a'
        for p in pts:
            if abs(p[0] - 0.008) < 1e-9:
                macros[f'RqOneHitEight{proto}'] = f'{hu(100 * p[1], 0)}\\%'
            if abs(p[0] - 0.010) < 1e-9:
                macros[f'RqOneHitTen{proto}'] = f'{hu(100 * p[1], 0)}\\%'
                macros[f'RqOneInflTen{proto}'] = f'{hu(p[3], 1)}'
    for k in [2, 5, 10, 50]:
        th = [v['theory'] for v in r.values() if v['kappa'] == k]
        if th:
            macros[f'ThKappa{["two","five","ten","fifty"][[2,5,10,50].index(k)]}'] = f'{hu(100 * th[0], 2)}\\%'
    worst = 0.0
    for proto in ['nbs', 'bayess']:
        for k in [2, 5, 10, 50]:
            ok = sorted(v['beta'] for v in r.values() if v['proto'] == proto and v['kappa'] == k and v['hit_mean'] >= 0.95)
            if ok:
                worst = max(worst, ok[0])
    macros['RqOneNinetyFive'] = f'{hu(100 * worst, 1)}\\%'
    if rc:
        macros['RqOneCbayMil'] = f"{hu(100 * rc['bayess|1000000']['mean'], 2)}\\%"
        macros['RqOneCsgdThree'] = f"{hu(100 * rc['sgd|3000000']['mean'], 2)}\\%"
    pts = [v for v in r.values() if v['proto'] == 'sgd' and v['kappa'] == 50 and abs(v['beta'] - 0.01) < 1e-9]
    if pts:
        macros['RqOneSgdFifty'] = f"{hu(pts[0]['infl_mean'], 0)}"


# ------------------------------------------------------------------ RQ1b table
def tab_rq1b():
    r = load('rq1b')
    if r is None:
        return
    qs = [0.9, 0.95, 0.99, 0.999]
    epss = [0.25, 0.5, 1.0, 2.0, 4.0]
    lines = []
    for q in qs:
        first = True
        for kind, lab in [('cliff', r'\cliff{} (OPA)'), ('ipa', 'IPA')]:
            cells = []
            for e in epss:
                v = r[f'{q}|{e}|{kind}']
                cap = 4 * max(v['theory'], v['central']) + 0.01
                emp = f"$>${hu(100 * v['mean'], 2)}" if abs(v['mean'] - cap) < 2e-4 else pm(100 * v['mean'], 100 * v['std'], 2)
                if f'{q}|{e}|{kind}' in PLATEAU:
                    emp = '---$^' + chr(92) + 'dagger$'

                thv = th_tol(CC(), q, v['mstar'], e, kind)
                TOLGAP.append((q, 100 * (v['theory'] - thv)))
                pv = pred_step(CC(), 'creditcard', 'nbs', 10**7, e, q, v['mstar'], kind)
                ps = r'$\infty$' if (pv is None or math.isinf(pv) or pv >= 1) else f'{hu(100 * pv, 2)}'
                if not emp.startswith('$>$') and pv is not None and not math.isinf(pv) and kind == 'cliff':
                    PREDERR.append((q, abs(100 * (v['mean'] - pv))))
                    T3ERR.append((q, abs(100 * (v['mean'] - pv))))
                if kind == 'ipa' and emp.startswith('$>$'):
                    PREDINF.append(pv is None or math.isinf(pv) or pv >= 1)
                cells.append(f"{emp} & {hu(100 * thv, 2)} & {ps}")
            qlab = f'\\multirow{{2}}{{*}}{{{q:g}}}' if first else ''
            first = False
            lines.append(f"{qlab} & {lab} & " + ' & '.join(cells) + r'\\')
        lines.append(r'\midrule')
    lines = lines[:-1]
    head = ' & '.join([f'\\multicolumn{{3}}{{c}}{{$\\eps={e:g}$}}' for e in epss])
    sub = ' & '.join([r'emp.$\downarrow$ & thm. & pred.'] * len(epss))
    cm = ' '.join([f'\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}' for i in range(len(epss))])
    s = r"""\begin{table*}[t]
\centering
\caption{RQ1: empirical budget $\hat\beta$ (\%; \NBS{}, $n=10^7$, CreditCard) versus ``thm.'' and ``pred.'' (\cref{sec:setup}). $>$: above the search range; $\infty$, $\dagger$: see text.}
\label{tab:rq1b}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{cl""" + 'rrr' * len(epss) + r"""}
\toprule
\hd & & """ + head + r"""\\
""" + cm + r"""
\hd $q$ & attack & """ + sub + r"""\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
"""
    open(f'{P}/tables/tab_rq1b.tex', 'w').write(s)
    v1, v2 = r['0.99|1.0|cliff'], r['0.99|1.0|ipa']
    macros['RqOneBcliff'] = f"{hu(100 * v1['mean'], 2)}\\%"
    macros['RqOneBipa'] = f"{hu(100 * v2['mean'], 2)}\\%"
    rat = [r[f'{q}|{e}|cliff']['mean'] / r[f'{q}|{e}|ipa']['mean'] for q in [0.9, 0.95] for e in epss]
    th = [1 - math.exp(-e) for q in [0.9, 0.95] for e in epss]
    macros['RqOneRatioErr'] = f"{hu(100 * max(abs(a - b) for a, b in zip(rat, th)), 0)}"


# ------------------------------------------------------------------ RQ1c figure (convergence in n)
def fig_rq1c():
    r = load('rq1c')
    if r is None:
        return
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    th = None
    for proto in ['nbs', 'bayess', 'sgd']:
        pts = sorted([(v['n'], v['mean'], v['std']) for v in r.values() if v['proto'] == proto])
        if not pts:
            continue
        th = [v['theory'] for v in r.values()][0]
        ax.errorbar([p[0] for p in pts], [100 * p[1] for p in pts], yerr=[100 * p[2] for p in pts], color=COL[proto], marker='o', ms=2.5,
                    lw=1.0, capsize=1.5, elinewidth=0.6, label={'nbs': 'NBS', 'bayess': 'BayeSS', 'sgd': 'SGD-SN-log'}[proto])
    ax.axhline(100 * th, color='k', lw=0.8, ls='--', label=r'Thm. 1 ($\eta=0$)')
    ax.set_xscale('log'); ax.set_xlabel('number of users $n$'); ax.set_ylabel(r'$\hat\beta$ (%)')
    ax.grid(alpha=0.25, lw=0.4); ax.legend(frameon=False, fontsize=6.3, ncol=2)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{P}/figs/fig_rq1c.pdf')
    nb = sorted([(v['n'], v['mean']) for v in r.values() if v['proto'] == 'nbs'])
    macros['RqOneCbig'] = f'{hu(100 * nb[-1][1], 2)}\\%'
    macros['RqOneCsmall'] = f'{hu(100 * nb[0][1], 2)}\\%'
    macros['RqOneCth'] = f'{hu(100 * th, 2)}\\%'


# ------------------------------------------------------------------ RQ2 table
def tab_rq2():
    r = load('rq2')
    if r is None:
        return
    global HC
    HC = load('hhcc')
    lines = []
    for d in ['creditcard', 'flights', 'power', 'pareto']:
        first = True
        for proto in ['nbs', 'bayess', 'sgd', 'hh']:
            kinds = ['cliff', 'tcliff'] if proto == 'hh' else ['cliff']
            for kind in kinds:
                c = r[f'{d}|{proto}|none|0.0']
                cells = [pm(c['rank_mean'], c['rank_std'], 3), pm(c['infl_mean'], c['infl_std'], 1)] if kind == 'cliff' else ['\\multicolumn{1}{c}{--}', '\\multicolumn{1}{c}{--}']
                src = (lambda b: HC[f'{d}|{b}']) if kind == 'tcliff' else (lambda b: r[f'{d}|{proto}|{kind}|{b}'])
                for b in [0.005, 0.01, 0.015, 0.02]:
                    v = src(b)
                    cells.append(hitcell(v['hit_mean'], v['hit_std']))
                v = src(0.01)
                cells.append(pm(v['infl_mean'], v['infl_std'], 1))
                lab = PNAME[proto] + ('' if kind == 'cliff' else r' \scriptsize(\textsc{T-Cliff})')
                dl = f'\\multirow{{5}}{{*}}{{\\rotatebox{{90}}{{\\scriptsize {DNAME[d]}}}}}' if first else ''
                first = False
                lines.append(f'{dl} & {lab} & ' + ' & '.join(cells) + r'\\')
        lines.append(r'\midrule')
    lines = lines[:-1]
    s = r"""\begin{table}[t]
\centering
\caption{RQ2: clean rank and inflation, hit rate under \cliff{}, and inflation at $1\%$ ($q=0.99$, $\eps=1$, $\mstar=\min\{10x_q,B\}$). Green: $\ge0.9$; yellow: $\ge0.5$.}
\label{tab:rq2}
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{clcccccccc}
\toprule
\hd & & \multicolumn{2}{c}{clean} & \multicolumn{4}{c}{hit rate$\uparrow$} & infl.\\
\cmidrule(lr){3-4}\cmidrule(lr){5-8}
\hd & protocol & rank & infl. & $\beta{=}0.5\%$ & $1\%$ & $1.5\%$ & $2\%$ & @1\%\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_rq2.tex', 'w').write(s)
    macros['RqTwoBayHitCC'] = f"{hu(100 * r['creditcard|bayess|cliff|0.01']['hit_mean'], 0)}\\%"
    macros['RqTwoNbsHitCC'] = f"{hu(100 * r['creditcard|nbs|cliff|0.01']['hit_mean'], 0)}\\%"
    macros['RqTwoSgdInflCC'] = f"{hu(r['creditcard|sgd|cliff|0.01']['infl_mean'], 1)}"
    macros['RqTwoFloodCC'] = f"{hu(r['creditcard|hh|flood|0.005']['infl_mean'], 0)}"
    hc = load('hh_ci')
    if hc:
        macros['RqTwoNoCiRank'] = f"{hu(hc['False|0.0']['rank_mean'], 2)}"
        macros['RqTwoNoCiRankStd'] = f"{hu(hc['False|0.0']['rank_std'], 2)}"
        macros['RqTwoNoCiInflOne'] = f"{hu(hc['False|0.01']['infl_mean'], 1)}"
        macros['RqTwoCiInflOne'] = f"{hu(hc['True|0.01']['infl_mean'], 2)}"
        macros['RqTwoCiHitTwo'] = f"{hu(hc['True|0.02']['hit_mean'], 2)}"
        macros['RqTwoCiHitThree'] = f"{hu(hc['True|0.03']['hit_mean'], 2)}"
    macros['RqTwoFloodFL'] = f"{hu(r['flights|hh|flood|0.005']['infl_mean'], 0)}"
    macros['RqTwoFloodPA'] = f"{hu(r['pareto|hh|flood|0.005']['infl_mean'], 0)}"
    hh1 = [HC[f'{d}|0.01']['hit_mean'] for d in ['creditcard', 'flights', 'power', 'pareto']]
    macros['RqTwoTcOneLo'] = f"{hu(min(hh1), 2)}"; macros['RqTwoTcOneHi'] = f"{hu(max(hh1), 2)}"
    macros['RqTwoTcFifteen'] = f"{hu(min(HC[f'{d}|0.015']['hit_mean'] for d in ['creditcard', 'flights', 'power', 'pareto']), 2)}"
    eh = eta_hat('hh', 10**7, 1.0, 0.99)
    macros['RqTwoHHEta'] = f"{hu(eh, 3)}"
    macros['RqTwoHHPred'] = f"{hu(100 * pred(CC(), 0.99, CC().target(0.99, 10), 1.0, eh, 'tcliff'), 2)}\\%"
    sg = load('sgd1e7') or {}
    if sg and 'hit|pareto|0.02' in sg:
        h1 = [sg[f'hit|{d}|0.01']['hit_mean'] for d in ['creditcard', 'flights', 'power', 'pareto']]
        macros['RqTwoSgdBigLo'] = f"{hu(100 * min(h1), 0)}"; macros['RqTwoSgdBigHi'] = f"{hu(100 * max(h1), 0)}\\%"
        h5 = [sg[f'hit|{d}|0.005']['hit_mean'] for d in ['creditcard', 'flights', 'power', 'pareto']]
        macros['RqTwoSgdBigHalf'] = f"{hu(100 * max(h5), 0)}\\%"
        i1 = [sg[f'hit|{d}|0.01']['infl_mean'] for d in ['creditcard', 'flights', 'pareto']]
        macros['RqTwoSgdBigInflLo'] = f"{hu(min(i1), 1)}"; macros['RqTwoSgdBigInflHi'] = f"{hu(max(i1), 1)}"
    # abstract / contribution numbers (CreditCard, kappa=10)
    r1 = load('rq1a')
    def hit1(p, b):
        return [v for v in r1.values() if v['proto'] == p and v['kappa'] == 10 and abs(v['beta'] - b) < 1e-9][0]['hit_mean']
    h = sorted([hit1('nbs', 0.01), hit1('bayess', 0.01)])
    macros['AbsBeta'] = '1\\%'
    macros['AbsHitRange'] = f"{hu(100 * h[0], 0)}--{hu(100 * h[1], 0)}\\%"
    alls = [b for b in [0.005, 0.01, 0.02, 0.03] if all(r[f'creditcard|{p}|cliff|{b}']['hit_mean'] >= 0.9
            for p in ['nbs', 'bayess', 'sgd'])]
    macros['AbsBetaAll'] = f"{100 * min(alls):g}\\%"
    mins = []
    for p in ['nbs', 'bayess', 'sgd']:
        ok = sorted(v['beta'] for v in r1.values() if v['proto'] == p and v['kappa'] == 10 and v['hit_mean'] >= 0.9)
        mins.append(ok[0])
    okhh = [b for b in [0.005, 0.01, 0.02, 0.03] if r[f'creditcard|hh|cliff|{b}']['hit_mean'] >= 0.9]
    mins.append(okhh[0])
    macros['AbsBetaRange'] = f"{100 * min(mins):g}--{100 * max(mins):g}\\%"


# ------------------------------------------------------------------ RQ3 table + figure
def rq3_out():
    r = load('rq3')
    if r is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
    rf = load('rq3fgac') or {}
    r = dict(r)
    r.update({k: v for k, v in rf.items() if k != 'beta_star'})
    acol = {'cliff': '#c0392b', 'scliff': '#e67e22', 'maxval': '#1f4e79', 'ipa': '#7f8c8d', 'fga': '#8e44ad', 'fgac': '#27ae60'}
    alab = {'cliff': 'CLIFF', 'scliff': 'S-CLIFF', 'maxval': 'MGA', 'ipa': 'IPA', 'fga': 'FGA-R', 'fgac': 'FGA-C'}
    for ax, proto in zip(axs, ['nbs', 'bayess']):
        for att in ['cliff', 'scliff', 'maxval', 'ipa', 'fga', 'fgac']:
            if not any(k.startswith(f'{proto}|{att}|') for k in r):
                continue
            pts = sorted([(v['ratio'], v['logerr_mean'], v['logerr_std']) for k, v in r.items() if k.startswith(f'{proto}|{att}|')])
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=acol[att], marker='o', ms=2.2, lw=0.9,
                        capsize=1.2, elinewidth=0.5, label=alab[att])
        ax.set_xscale('log'); ax.set_xticks([0.8, 1, 1.5, 2, 3]); ax.set_xticklabels(['.8', '1', '1.5', '2', '3'])
        ax.set_xlabel(r'budget $\beta/\beta^\star$'); ax.set_title({'nbs': 'NBS', 'bayess': 'BayeSS'}[proto], fontsize=8, pad=2)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel(r'$|\log_2(\hat m/m^\star)|$')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=5.6, loc='upper center', ncol=6, handlelength=1.0, columnspacing=0.6, handletextpad=0.3)
    plt.tight_layout(pad=0.2, w_pad=0.4, rect=(0, 0, 1, 0.9))
    plt.savefig(f'{P}/figs/fig_rq3.pdf')
    # knowledge table
    lines = []
    for s_aux in [1000, 10000, 100000, 1000000]:
        cells = []
        for proto in ['nbs', 'bayess']:
            for att in ['maxval', 'cliff']:
                v = r.get(f'know|{s_aux}|{proto}|{att}')
                cells.append(hitcell(v['hit_mean'], v['hit_std']))
                cells.append(pm(v['logerr_mean'], v['logerr_std'], 2))
        v = r[f'know|{s_aux}|nbs|maxval']
        lines.append(f'$10^{{{int(math.log10(s_aux))}}}$ & ' + pm(100 * v['beta_tuned_mean'], 100 * v['beta_tuned_std'], 2) + ' & ' + ' & '.join(cells) + r'\\')
    s = r"""\begin{table}[t]
\centering
\caption{RQ3: attacker knowledge. \cliff{} uses $1.2\tilde\beta$, \textsc{MGA} $\tilde\beta$, with $\tilde\beta$ (\%) estimated from $N_{\mathrm{aux}}$ samples.}
\label{tab:rq3}
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{cc cc cc cc cc}
\toprule
\hd & & \multicolumn{4}{c}{\NBS{}} & \multicolumn{4}{c}{\BAY{}}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
\hd & & \multicolumn{2}{c}{\textsc{MGA}} & \multicolumn{2}{c}{\cliff{}} & \multicolumn{2}{c}{\textsc{MGA}} & \multicolumn{2}{c}{\cliff{}}\\
\hd $N_{\mathrm{aux}}$ & $\tilde\beta$ (\%) & hit$\uparrow$ & log-err$\downarrow$ & hit$\uparrow$ & log-err$\downarrow$ & hit$\uparrow$ & log-err$\downarrow$ & hit$\uparrow$ & log-err$\downarrow$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_rq3.tex', 'w').write(s)
    macros['RqThreeBstar'] = f"{hu(100 * r['beta_star'], 2)}\\%"
    fc = [(k, v) for k, v in rf.items() if '|fgac|' in k]
    if fc:
        d = [abs(v['hit_mean'] - r[k.replace('|fgac|', '|cliff|')]['hit_mean']) for k, v in fc]
        macros['FgacDiffMax'] = f"{hu(max(d), 2)}"
        macros['FgacTwoLo'] = f"{hu(min(v['hit_mean'] for k, v in fc if v['ratio'] >= 2 and v['proto'] != 'sgd'), 2)}"
        macros['FgacSgdThree'] = f"{hu(rf['sgd|fgac|3.0']['hit_mean'], 2)}"
    fg = [v for k, v in r.items() if '|fga|' in k]
    if fg:
        macros['FgaHitMax'] = f"{hu(max(v['hit_mean'] for v in fg), 2)}"
        macros['FgaInflLo'] = f"{hu(min(v['infl_mean'] for v in fg if v['proto'] != 'sgd'), 1)}"
        macros['FgaInflHi'] = f"{hu(max(v['infl_mean'] for v in fg if v['proto'] != 'sgd'), 1)}"
        macros['FgaSgdInfl'] = f"{hu(max(v['infl_mean'] for v in fg if v['proto'] == 'sgd'), 1)}"
        macros['FgaSgdInflLo'] = f"{hu(min(v['infl_mean'] for v in fg if v['proto'] == 'sgd'), 1)}"
        macros['FgaErrLo'] = f"{hu(min(v['logerr_mean'] for v in fg if v['proto'] != 'sgd'), 1)}"
    for proto in ['nbs', 'bayess']:
        v = r[f'{proto}|maxval|1.5']; w = r[f'{proto}|cliff|1.5']
        macros[f'RqThreeMaxErr{proto}'] = f"{hu(v['logerr_mean'], 1)}"
        macros[f'RqThreeCliffErr{proto}'] = f"{hu(w['logerr_mean'], 2)}"
        macros[f'RqThreeMaxInfl{proto}'] = f"{hu(v['infl_mean'], 0)}"
        v = r[f'{proto}|random|3.0']
        macros[f'RqThreeRandInfl{proto}'] = f"{hu(v['infl_mean'], 2)}"


# ------------------------------------------------------------------ RQ4 tables + figure
def rq4_out():
    r = load('rq4'); rc = load('rq4c')
    if r is None:
        return
    qs = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    lines = []
    for d in ['creditcard', 'flights', 'power', 'pareto']:
        cells = [pm(100 * r[f'q|{d}|{q}']['mean'], 100 * r[f'q|{d}|{q}']['std'], 2) for q in qs]
        lines.append(f'{DNAME[d]} & ' + ' & '.join(cells) + r'\\')
        pop = Pop(d)
        ths = []
        for q in qs:
            ms = pop.target(q, 10) if pop.target(q, 10) > pop.quantile(q) + 1 else pop.B
            ths.append(th_tol(pop, q, ms, 1.0, 'cliff'))
            TOLGAP.append((q, 100 * (r[f'q|{d}|{q}']['theory'] - ths[-1])))
            RQ4TH[(d, q)] = ths[-1]
            pv = pred_step(pop, d, 'nbs', 10**7, 1.0, q, ms)
            RQ4PRED[(d, q)] = pv
            PREDERR.append((q, abs(100 * (r[f'q|{d}|{q}']['mean'] - pv))))
            RQ4ERR.append(abs(100 * (r[f'q|{d}|{q}']['mean'] - pv)))
            RQ4ERRQ.append((q, abs(100 * (r[f'q|{d}|{q}']['mean'] - pv))))
        lines.append(r'\rowcolor{zebra}\quad thm.\,/\,pred. & ' + ' & '.join([f'{hu(100 * t, 2)}\\,/\\,{hu(100 * RQ4PRED[(d, q)], 2)}' for t, q in zip(ths, qs)]) + r'\\')
    lines.append(r'\midrule $(1{-}q)(1{-}e^{-\eps})$ & ' + ' & '.join([f'{hu(100 * (1 - q) * (1 - math.exp(-1)), 2)}' for q in qs]) + r'\\')
    s = r"""\begin{table}[t]
\centering
\caption{RQ4: budget (\%) to hit $\min\{10x_q,B\}$ by quantile level $q$ (\NBS{}, $n=10^7$, $\eps=1$).}
\label{tab:rq4q}
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lcccccc}
\toprule
\hd & \multicolumn{6}{c}{$\hat\beta$ (\%)$\downarrow$ at quantile level $q$}\\
\cmidrule(lr){2-7}
\hd dataset & $0.5$ & $0.75$ & $0.9$ & $0.95$ & $0.99$ & $0.999$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_rq4q.tex', 'w').write(s)
    macros['RqFourMedian'] = f"{hu(100 * r['q|creditcard|0.5']['mean'], 1)}\\%"
    macros['RqFourPninetynine'] = f"{hu(100 * r['q|creditcard|0.99']['mean'], 2)}\\%"
    # tails figure
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    dists = [('pareto', 1.1, 'Pareto(1.1)', '#1f4e79', '-'), ('pareto', 1.5, 'Pareto(1.5)', '#2e86c1', '-'), ('pareto', 2.5, 'Pareto(2.5)', '#85c1e9', '-'),
             ('lognormal', 2.0, 'LogN(2)', '#c0392b', '-'), ('lognormal', 1.0, 'LogN(1)', '#e6b0aa', '-'), ('exponential', None, 'Exp', '#27ae60', '-'),
             ('uniform', None, 'Uniform', '#7f8c8d', '-'), ('creditcard', None, 'CreditCard', 'k', '--'), ('flights', None, 'Flights', '#7d3c98', '--'),
             ('power', None, 'Power', '#e67e22', '--')]
    for d, par, lab, colr, ls in dists:
        pts = sorted([(v['beta'], 2 ** v['log2infl_mean'], v['log2infl_std']) for k, v in r.items() if k.startswith('tail|') and v['data'] == d and v['param'] == par])
        gm = [p[1] for p in pts]
        lo = [g - g / 2 ** p[2] for g, p in zip(gm, pts)]; up = [g * 2 ** p[2] - g for g, p in zip(gm, pts)]
        ax.errorbar([100 * p[0] for p in pts], gm, yerr=[lo, up], color=colr, ls=ls, lw=0.9, marker='o', ms=1.8,
                    capsize=1, elinewidth=0.5, label=lab)
    ax.axvline(100 * 0.01 * (1 - math.exp(-1)), color='k', lw=0.5, ls=':')
    ax.set_yscale('log'); ax.set_xlabel(r'poisoning rate $\beta$ (%)'); ax.set_ylabel(r'inflation $\hat m/x_{0.99}$')
    ax.grid(alpha=0.25, lw=0.4); ax.legend(frameon=False, fontsize=5.4, ncol=2, loc='upper left', handlelength=1.3, columnspacing=0.6)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{P}/figs/fig_rq4b.pdf')
    for d, par in [('pareto', 1.1), ('uniform', None), ('lognormal', 2.0)]:
        v = [vv for k, vv in r.items() if k.startswith('tail|') and vv['data'] == d and vv['param'] == par and abs(vv['beta'] - 0.005) < 1e-9][0]
        macros[f'RqFourInfl{d}'] = f"{hu(v['infl_mean'], 0)}" if v['infl_mean'] >= 10 else f"{hu(v['infl_mean'], 2)}"
    if rc:
        lines = []
        for q in [0.95, 0.99]:
            for dd in [0.002, 0.005, 0.01, 0.02, 0.04]:
                up = rc.get(f'{q}|{dd}|up'); dn = rc.get(f'{q}|{dd}|down')
                cu = (pm(100 * up['mean'], 100 * up['std'], 2) + f" & {hu(100 * up['theory'], 2)}") if up else r'\multicolumn{2}{c}{---}'
                cd = (pm(100 * dn['mean'], 100 * dn['std'], 2) + f" & {hu(100 * dn['theory'], 2)}") if dn else r'\multicolumn{2}{c}{---}'
                rat = f"{hu(dn['mean'] / up['mean'], 2)}" if (up and dn and up['mean'] > 0) else '---'
                rat += (f" & {hu(dn['theory'] / up['theory'], 2)}" if (up and dn and up['theory'] > 0) else ' & ---')
                if up and dn and up['theory'] > 0:
                    DIRDEV.append(abs(dn['mean'] / up['mean'] - dn['theory'] / up['theory']))
                assert up or q + dd > 1 + 1e-12, (q, dd)
                lines.append(f'{q:g} & {dd:g} & {cu} & {cd} & {rat}\\\\')
            lines.append(r'\midrule')
        lines = lines[:-1]
        s = r"""\begin{table}[t]
\centering
\caption{RQ4: budget (\%) to move the $q$-th quantile up to $x_{q+d}$ or down to $x_{q-d}$, and the down/up ratio (\cref{prop:dir}).}
\label{tab:rq4dir}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{cc rr rr cc}
\toprule
\hd & & \multicolumn{2}{c}{inflate (up)} & \multicolumn{2}{c}{deflate (down)} & \multicolumn{2}{c}{ratio}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\hd $q$ & $d$ & emp.$\downarrow$ & thm. & emp.$\downarrow$ & thm. & emp. & thm.\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
        open(f'{P}/tables/tab_rq4dir.tex', 'w').write(s)
        dv = sorted(DIRDEV)
        macros['DirDevTyp'] = f"{hu(dv[-3], 2)}"
        macros['DirDevMax'] = f"{hu(dv[-1], 2)}"


# ------------------------------------------------------------------ RQ5 tables
def rq5_out():
    r = load('rq5')
    if r is None:
        return
    J = load('jump2')
    KIND = ['none', 'cliff', 'scliff', 'ipa', 'maxval', 'low']
    lines = []
    for det, lab in [('range', 'range audit'), ('jump', 'jump audit'), ('bound', r'boundary audit ($t\in\{0,B\}$)')]:
        lines.append(r'\multicolumn{8}{l}{\emph{' + lab + r'}}\\')
        for n, frac in [(10**6, 0.05), (10**7, 0.05), (10**7, 0.2), (10**8, 0.2)]:
            cells = []
            for kind in KIND:
                v = J[f'audit|{n}|{frac}|{kind}']
                cells.append(f"{hu(v[det + '_mean'], 3)}")
            lines.append(f'$10^{{{int(math.log10(n))}}}$ & {int(frac * 100)}\\% & ' + ' & '.join(cells) + r'\\')
    s1 = r"""\begin{table}[t]
\centering
\caption{RQ5: fraction of 200 runs flagged by three audits (\NBS{}, CreditCard, $\beta=1\%$). Clean: false-positive rate.}
\label{tab:rq5a}
\scriptsize
\setlength{\tabcolsep}{2.0pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{cc cccccc}
\toprule
\hd & & FPR$\downarrow$ & \multicolumn{5}{c}{detection rate$\uparrow$}\\
\cmidrule(lr){3-3}\cmidrule(lr){4-8}
\hd $n$ & audit & clean & \cliff{} & \scliff{} & IPA & \textsc{MGA} & \textsc{Low}\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_rq5a.tex', 'w').write(s1)
    a = lambda n, f, k, d: J[f'audit|{n}|{f}|{k}'][d + '_mean']
    macros['RqFiveDetBig'] = f"{hu(100 * a(10**8, 0.2, 'cliff', 'range'), 0)}\\%"
    macros['RqFiveDetSmall'] = f"{hu(100 * a(10**6, 0.05, 'cliff', 'range'), 1)}\\%"
    macros['RqFiveDetMid'] = f"{hu(100 * a(10**7, 0.2, 'cliff', 'range'), 0)}\\%"
    macros['RqFiveDetStealth'] = f"{hu(100 * a(10**8, 0.2, 'scliff', 'range'), 0)}\\%"
    macros['RqFiveRangeFPRMax'] = f"{hu(100 * max(a(n, f, 'none', 'range') for n, f in [(10**6, 0.05), (10**7, 0.05), (10**7, 0.2), (10**8, 0.2)]), 1)}\\%"
    macros['JumpFPRMax'] = f"{hu(max(a(n, f, 'none', 'jump') for n, f in [(10**6, 0.05), (10**7, 0.05), (10**7, 0.2), (10**8, 0.2)]), 2)}"
    macros['JumpCliffSmall'] = f"{hu(100 * a(10**6, 0.05, 'cliff', 'jump'), 1)}\\%"
    macros['JumpCliffFive'] = f"{hu(a(10**7, 0.05, 'cliff', 'jump'), 2)}"
    macros['JumpCliffTwenty'] = f"{hu(100 * a(10**7, 0.2, 'cliff', 'jump'), 0)}\\%"
    macros['JumpScliffTwenty'] = f"{hu(100 * a(10**7, 0.2, 'scliff', 'jump'), 0)}\\%"
    macros['JumpScliffBig'] = f"{hu(100 * a(10**8, 0.2, 'scliff', 'jump'), 0)}\\%"
    macros['JumpMgaTwenty'] = f"{hu(100 * a(10**7, 0.2, 'maxval', 'jump'), 0)}\\%"
    macros['JumpMgaFive'] = f"{hu(a(10**7, 0.05, 'maxval', 'jump'), 2)}"
    macros['JumpLowMax'] = f"{hu(100 * max(a(n, f, 'low', d) for n, f in [(10**6, 0.05), (10**7, 0.05), (10**7, 0.2), (10**8, 0.2)] for d in ['jump', 'range', 'bound']), 1)}\\%"
    macros['BoundCliffFive'] = f"{hu(100 * a(10**7, 0.05, 'cliff', 'bound'), 0)}\\%"
    macros['BoundFPRMax'] = f"{hu(100 * max(a(n, f, 'none', 'bound') for n, f in [(10**6, 0.05), (10**7, 0.05), (10**7, 0.2), (10**8, 0.2)]), 0)}\\%"
    macros['JumpDCliff'] = f"{hu(J['audit|100000000|0.2|cliff']['D_mean'], 4)}"
    macros['JumpDLow'] = f"{hu(J['audit|100000000|0.2|low']['D_mean'], 4)}"
    macros['JumpDClean'] = f"{hu(J['audit|100000000|0.2|none']['D_mean'], 4)}"
    hl = [J[f'hit|{n}|low']['hit'] for n in [10**6, 10**7, 10**8]]
    hc = [J[f'hit|{n}|cliff']['hit'] for n in [10**6, 10**7, 10**8]]
    macros['LowHitSmall'] = f"{hu(hl[0], 3)}"; macros['LowHitMid'] = f"{hu(hl[1], 3)}"; macros['LowHitBig'] = f"{hu(hl[2], 3)}"
    macros['CliffHitSmallJ'] = f"{hu(hc[0], 2)}"
    macros['ExactNbsOne'] = f"{hu(100 * J['hit|1000000|cliff']['exact'], 0)}\\%"
    macros['LowTopMid'] = f"{hu(J['hit|10000000|low']['v'], 2)}"
    ctq0 = [v['safe_mean'] for k, v in r.items() if k.startswith('ctq|') and v['mult'] == 1.0 and v['mode'] == 'ctq']
    macros['RqFiveFirstMin'] = f"{hu(100 * min(ctq0), 0)}\\%"
    for _p, _t in [('nbs', 'Nbs'), ('bayess', 'Bay'), ('sgd', 'Sgd')]:
        _x = [v['safe_mean'] for k, v in r.items() if k.startswith('ctq|') and v['mult'] == 1.0 and v['mode'] == 'ctq' and v['proto'] == _p]
        macros[f'RqFiveFixed{_t}Lo'] = f"{hu(min(_x), 2)}"; macros[f'RqFiveFixed{_t}Hi'] = f"{hu(max(_x), 2)}"
    rd = load('rq5d')
    if rd:
        macros['RqFiveAdaptMin'] = f"{hu(min(v['safe_mean'] for v in rd.values()), 2)}"
    re_ = load('rq5e')
    if re_:
        macros['RqFiveSclMin'] = f"{hu(min(v['safe_mean'] for v in re_.values()), 2)}"
    # CTQ table
    lines = []
    for eps in [0.5, 1.0, 2.0]:
        first = True
        for bbar in [0.005, 0.01, 0.02]:
            cells = []
            for proto in ['nbs', 'bayess', 'sgd']:
                c0 = r[f'ctq|{eps}|{bbar}|{proto}|0.0|ctq']
                c1 = r[f'ctq|{eps}|{bbar}|{proto}|1.0|ctq']
                c2 = r[f'ctq|{eps}|{bbar}|{proto}|2.0|ctq']
                def ic(v):
                    col = 'good' if v['infl_mean'] <= 1.1 else ('mid' if v['infl_mean'] <= 2 else 'bad')
                    return f'\\cellcolor{{{col}}}' + pm(v['infl_mean'], v['infl_std'], 2)
                cells += [pm(c0['rank_mean'], c0['rank_std'], 3), ic(c1), ic(c2)]
            qd = r[f'ctq|{eps}|{bbar}|nbs|0.0|ctq']['qdag']
            el = f'\\multirow{{3}}{{*}}{{{eps:g}}}' if first else ''
            first = False
            lines.append(f'{el} & {100 * bbar:g}\\% & {hu(qd, 3)} & ' + ' & '.join(cells) + r'\\')
        lines.append(r'\midrule')
    lines = lines[:-1]
    # plain baseline row (no CTQ) at bbar=1%
    base = []
    for eps in [0.5, 1.0, 2.0]:
        for proto in ['nbs', 'bayess', 'sgd']:
            c1 = r[f'ctq|{eps}|0.01|{proto}|1.0|plain']
            base.append(c1['infl_mean'])
    macros['RqFivePlainInflMin'] = f"{hu(min(base), 1)}"
    macros['RqFivePlainInflMax'] = f"{hu(max(base), 1)}"
    s2 = r"""\begin{table*}[t]
\centering
\caption{RQ5: \ctq{} at $q^\dagger$ on CreditCard ($n=10^6$). \NBS{}$^\dagger$: certified $\eta_{\mathrm{NBS}}$; other blocks: oracle $\eta$. Arrows: defender's view.}
\label{tab:rq5b}
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{ccc ccc ccc ccc}
\toprule
\hd & & & \multicolumn{3}{c}{\NBS{}} & \multicolumn{3}{c}{\BAY{}} & \multicolumn{3}{c}{\SGD{}}\\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}
\hd $\eps$ & $\bar\beta$ & $q^\dagger$ & clean rank & @$\bar\beta$ & @$2\bar\beta$ & clean rank & @$\bar\beta$ & @$2\bar\beta$ & clean rank & @$\bar\beta$ & @$2\bar\beta$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f'{P}/tables/tab_rq5b.tex', 'w').write(s2)


# ------------------------------------------------------------------ RQ6 table
def rq6_out():
    r = load('rq6')
    if r is None:
        return
    lines = []
    for proto in ['nbs', 'bayess', 'sgd']:
        first = True
        for mode in ['plain', 'ctq']:
            cells = []
            for b in [0.0, 0.005, 0.01, 0.02]:
                v = r[f'cc|{proto}|{b}|{mode}']
                cells.append(pm(100 * v['evasion_mean'], 100 * v['evasion_std'], 0))
            v0 = r[f'cc|{proto}|0.0|{mode}']; v1 = r[f'cc|{proto}|0.01|{mode}']
            cells.append(pm(v0['thr_mean'] - 0.05, v0['thr_std'], 0))
            cells.append(pm(v1['thr_mean'] - 0.05, v1['thr_std'], 0))
            pl = f'\\multirow{{2}}{{*}}{{{PNAME[proto]}}}' if first else ''
            first = False
            lines.append(f'{pl} & {"plain" if mode == "plain" else chr(92) + "ctq{}"} & ' + ' & '.join(cells) + r'\\')
        lines.append(r'\midrule')
    lines = lines[:-1]
    s = r"""\begin{table}[t]
\centering
\caption{RQ6(a): attacker transactions that evade review (\%) and review threshold (EUR).}
\label{tab:rq6a}
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{cl cccc cc}
\toprule
\hd & & \multicolumn{4}{c}{evading review (\%)$\uparrow$} & \multicolumn{2}{c}{threshold (EUR)}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}
\hd & & $\beta=0$ & $0.5\%$ & $1\%$ & $2\%$ & clean & $\beta=1\%$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_rq6a.tex', 'w').write(s)
    v = r['cc|bayess|0.01|plain']; macros['RqSixEvBay'] = f"{hu(100 * v['evasion_mean'], 1)}\\%"
    v = r['cc|bayess|0.0|plain']; macros['RqSixEvBayClean'] = f"{hu(100 * v['evasion_mean'], 0)}\\%"
    v = r['cc|bayess|0.01|ctq']; macros['RqSixEvBayCtq'] = f"{hu(100 * v['evasion_mean'], 0)}\\%"
    v0 = r['cc|bayess|0.0|plain']; v1 = r['cc|bayess|0.01|plain']
    macros['RqSixRecallClean'] = f"{hu(100 * v0['recall_mean'], 1)}\\%"; macros['RqSixRecallAtk'] = f"{hu(100 * v1['recall_mean'], 1)}\\%"
    macros['RqSixAlertClean'] = f"{hu(100 * v0['alert_mean'], 2)}\\%"; macros['RqSixAlertAtk'] = f"{hu(100 * v1['alert_mean'], 3)}\\%"
    for p in ['nbs', 'bayess', 'sgd']:
        macros[f'RqSixThrClean{p}'] = f"{r[f'cc|{p}|0.0|plain']['thr_mean'] - 0.05:,.0f}"
        macros[f'RqSixThrAtk{p}'] = f"{r[f'cc|{p}|0.01|plain']['thr_mean'] - 0.05:,.0f}"
        macros[f'RqSixThrCtq{p}'] = f"{r[f'cc|{p}|0.0|ctq']['thr_mean'] - 0.05:,.0f}"
        macros[f'RqSixEvCtqOne{p}'] = f"{hu(100 * r[f'cc|{p}|0.01|ctq']['evasion_mean'], 1)}\\%"
        macros[f'RqSixEvCtqTwo{p}'] = f"{hu(100 * r[f'cc|{p}|0.02|ctq']['evasion_mean'], 0)}\\%"
        macros[f'RqSixSafeCtqOne{p}'] = f"{hu(r[f'cc|{p}|0.01|ctq']['safe_mean'], 2)}"
    macros['RqSixXqEur'] = f"{r['cc_meta']['xq_eur'] - 0.05:,.0f}"
    macros['RqSixRecallCtq'] = f"{hu(100 * r['cc|bayess|0.0|ctq']['recall_mean'], 1)}\\%"
    macros['RqSixAlertCtq'] = f"{hu(100 * r['cc|bayess|0.0|ctq']['alert_mean'], 2)}\\%"
    # flights + CI table (combined)
    lines = []
    for proto in ['nbs', 'bayess', 'sgd']:
        cells = []
        for b in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02]:
            v = r[f'fl|{proto}|{b}']
            cells.append(pm(v['alarm_mean'], v['alarm_std'], 2))
        lines.append(f'{PNAME[proto]} & ' + ' & '.join(cells) + r'\\')
    s = r"""\begin{table}[t]
\centering
\caption{RQ6: rate of false SLO alarms on Flights.}
\label{tab:rq6b}
\scriptsize
\setlength{\tabcolsep}{2.0pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lcccccc}
\toprule
\hd & \multicolumn{6}{c}{alarm rate$\uparrow$}\\
\cmidrule(lr){2-7}
\hd protocol & $\beta=0$ & $0.25\%$ & $0.5\%$ & $0.75\%$ & $1\%$ & $2\%$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_rq6b.tex', 'w').write(s)
    f0 = r['fl|nbs|0.0']
    macros['RqSixFlAlarm'] = f"{hu(f0['alarm_at'], 0)}"; macros['RqSixFlTarget'] = f"{hu(f0['target_delay'], 0)}"; macros['RqSixFlTrue'] = f"{hu(f0['clean_p99'], 0)}"
    cl = [r[f'fl|{p}|0.0']['p99_mean'] for p in ['nbs', 'bayess', 'sgd']]
    at = [r[f'fl|{p}|0.01']['p99_mean'] for p in ['nbs', 'bayess', 'sgd']]
    macros['RqSixFlCleanLo'] = f"{hu(min(cl), 0)}"; macros['RqSixFlCleanHi'] = f"{hu(max(cl), 0)}"
    macros['RqSixFlAtkLo'] = f"{hu(min(at), 0)}"; macros['RqSixFlAtkHi'] = f"{hu(max(at), 0)}"
    a75 = [r[f'fl|{p}|0.0075']['alarm_mean'] for p in ['nbs', 'bayess', 'sgd']]
    a1 = [r[f'fl|{p}|0.01']['alarm_mean'] for p in ['nbs', 'bayess', 'sgd']]
    a0 = [r[f'fl|{p}|0.0']['alarm_mean'] for p in ['nbs', 'bayess', 'sgd']]
    macros['RqSixFlAlSevLo'] = f"{hu(100 * min(a75), 0)}"; macros['RqSixFlAlSevHi'] = f"{hu(100 * max(a75), 0)}\\%"
    macros['RqSixFlAlOneLo'] = f"{hu(100 * min(a1), 0)}"; macros['RqSixFlAlOneHi'] = f"{hu(100 * max(a1), 0)}\\%"
    macros['RqSixFlAlZero'] = f"{hu(100 * max(a0), 0)}\\%"
    lines = []
    for d in ['creditcard', 'flights']:
        for b in [0.0, 0.005, 0.01, 0.02, 0.05]:
            v = r[f'ci|{d}|{b}']
            lines.append(f"{DNAME[d] if b == 0 else ''} & {100 * b:g}\\% & {pm(v['est_infl_mean'], v['est_infl_std'], 2)} & {pm(v['width_mean'], v['width_std'], 3)} & {pm(v['cov_true_mean'], v['cov_true_std'], 2)} & {hitcell(v['cov_tgt_mean'], v['cov_tgt_std'])}\\\\")
        lines.append(r'\midrule')
    lines = lines[:-1]
    s = r"""\begin{table}[t]
\centering
\caption{RQ6(b): confidence hijacking of \SGD{} (100 runs).}
\label{tab:rq6c}
\scriptsize
\setlength{\tabcolsep}{2.0pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lccccc}
\toprule
\hd data & $\beta$ & inflation & CI width & cover $x_q{\downarrow}$ & cover $\mstar{\uparrow}$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_rq6c.tex', 'w').write(s)
    v0 = r['ci|creditcard|0.0']; v2 = r['ci|creditcard|0.02']; v5 = r['ci|creditcard|0.05']
    macros['RqSixCovClean'] = f"{hu(100 * v0['cov_true_mean'], 0)}\\%"
    macros['RqSixCovAtk'] = f"{hu(100 * v2['cov_true_mean'], 0)}\\%"
    macros['RqSixCovTgt'] = f"{hu(100 * v5['cov_tgt_mean'], 0)}\\%"
    macros['RqSixWidthClean'] = f"{hu(v0['width_mean'], 3)}"
    macros['RqSixWidthAtk'] = f"{hu(v2['width_mean'], 3)}"
    v1 = r['ci|creditcard|0.01']
    macros['RqSixWidthOne'] = f"{hu(v1['width_mean'], 2)}"; macros['RqSixInflOne'] = f"{hu(v1['est_infl_mean'], 1)}"
    macros['RqSixWidthFive'] = f"{hu(v5['width_mean'], 3)}"
    cc = [r[f'ci|{d}|0.0']['cov_true_mean'] for d in ['creditcard', 'flights']]
    mh = [1 - r[f'ci|{d}|0.005']['cov_true_mean'] for d in ['creditcard', 'flights']]
    macros['RqSixMissHalfLo'] = f"{hu(100 * min(mh), 0)}"; macros['RqSixMissHalfHi'] = f"{hu(100 * max(mh), 0)}\\%"
    macros['RqSixEvCtqMax'] = f"{hu(100 * max(r[f'cc|{p}|{b}|ctq']['evasion_mean'] for p in ['nbs', 'bayess', 'sgd'] for b in [0.0, 0.0025, 0.005, 0.0075, 0.01]), 2)}\\%"
    macros['RqSixSafeCtqMin'] = f"{hu(100 * min(r[f'cc|{p}|{b}|ctq']['safe_mean'] for p in ['nbs', 'bayess', 'sgd'] for b in [0.0, 0.0025, 0.005, 0.0075, 0.01]), 0)}\\%"
    macros['RqSixCovLo'] = f"{hu(100 * min(cc), 0)}"; macros['RqSixCovHi'] = f"{hu(100 * max(cc), 0)}\\%"


def rq5c_out():
    r = load('rq5c')
    if r is None:
        return
    lines = []
    for eps in [0.5, 1.0, 2.0]:
        first = True
        for bbar in [0.005, 0.01, 0.02]:
            cells = []
            cj = load('cert') or {}
            k0, k1 = f'ctq|1000000|{eps}|{bbar}|0.0', f'ctq|1000000|{eps}|{bbar}|1.0'
            if k0 in cj:
                cells += [pm(cj[k0]['rank_mean'], cj[k0]['rank_std'], 3), hitcell(cj[k1]['safe_mean'], cj[k1]['safe_std'])]
            for proto in ['nbs', 'bayess', 'sgd']:
                c0 = r[f'{eps}|{bbar}|{proto}|0.0']; c1 = r[f'{eps}|{bbar}|{proto}|1.0']
                cells += [pm(c0['rank_mean'], c0['rank_std'], 3), hitcell(c1['safe_mean'], c1['safe_std'])]
            c2 = r[f'{eps}|{bbar}|bayess|2.0']
            col = 'good' if c2['infl_mean'] <= 1.1 else ('mid' if c2['infl_mean'] <= 2 else 'bad')
            cells.append(f'\\cellcolor{{{col}}}' + pm(c2['infl_mean'], c2['infl_std'], 2))
            el = f'\\multirow{{3}}{{*}}{{{eps:g}}}' if first else ''
            first = False
            lines.append(f'{el} & {100 * bbar:g}\\% & ' + ' & '.join(cells) + r'\\')
        lines.append(r'\midrule')
    lines = lines[:-1]
    s2 = r"""\begin{table*}[t]
\centering
\caption{RQ5: \ctq{} at $q^\dagger$ on CreditCard ($n=10^6$). \NBS{}$^\dagger$: certified $\eta_{\mathrm{NBS}}$; other blocks: oracle $\eta$. Arrows: defender's view.}
\label{tab:rq5b}
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{cc cc cc cc cc c}
\toprule
\hd & & \multicolumn{2}{c}{\NBS{}$^\dagger$ (certified)} & \multicolumn{2}{c}{\NBS{}} & \multicolumn{2}{c}{\BAY{}} & \multicolumn{2}{c}{\SGD{}} & \BAY{}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-11}
\hd $\eps$ & $\bar\beta$ & clean rank$\uparrow$ & safe@$\bar\beta{\uparrow}$ & clean rank$\uparrow$ & safe@$\bar\beta{\uparrow}$ & clean rank$\uparrow$ & safe@$\bar\beta{\uparrow}$ & clean rank$\uparrow$ & safe@$\bar\beta{\uparrow}$ & infl.@$2\bar\beta{\downarrow}$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f'{P}/tables/tab_rq5b.tex', 'w').write(s2)
    safe = [r[f'{e}|{b}|{p}|1.0']['safe_mean'] for e in [0.5, 1.0, 2.0] for b in [0.005, 0.01, 0.02] for p in ['nbs', 'bayess', 'sgd']]
    macros['RqFiveSafeMin'] = exs(min(safe))
    macros['RqFiveSafeMean'] = f"{hu(np.mean(safe), 3)}"
    v = r['1.0|0.01|bayess|0.0']; macros['RqFiveCleanBay'] = f"{hu(v['rank_mean'], 3)}"
    macros['RqFiveCleanNbsOne'] = f"{hu(r['1.0|0.01|nbs|0.0']['rank_mean'], 3)}"
    cj = load('cert') or {}
    if cj:
        macros['CertCleanSixOne'] = f"{hu(cj['ctq|1000000|1.0|0.01|0.0']['rank_mean'], 3)}"
        macros['CertCleanSevenOne'] = f"{hu(cj['ctq|10000000|1.0|0.01|0.0']['rank_mean'], 3)}"
        macros['CertSafeMin'] = f"{hu(min(v['safe_mean'] for k_, v in cj.items() if k_.startswith('ctq|') and v['mult'] <= 1.0), 2)}"
        macros['CertBudSix'] = f"{hu(100 * cj['cert|1000000']['beta'], 2)}\\%"
        macros['CertBudEight'] = f"{hu(100 * cj['cert|100000000']['beta'], 2)}\\%"
        macros['CertExactMin'] = f"{hu(100 * min(v['exact'] for k_, v in cj.items() if k_.startswith('cert|')), 1)}\\%"
    r2_ = load('rq2')
    ex = []
    for d_ in ['creditcard', 'flights', 'power', 'pareto']:
        v_ = r2_[f'{d_}|bayess|cliff|0.015']; m_ = np.array(v_['m_raw']); ms_ = v_['mstar']
        h_ = np.abs(m_ - ms_) <= 0.1 * ms_
        ex.append(float(np.mean(m_[h_] == ms_)))
    macros['ExactBayLo'] = f"{hu(100 * min(ex), 0)}\\%"; macros['ExactBayHi'] = f"{hu(100 * max(ex), 0)}\\%"
    tc = load('tc15') or {}
    if tc:
        macros['TcExactLo'] = f"{hu(100 * min(v['exact'] for v in tc.values()), 0)}\\%"; macros['TcExactHi'] = f"{hu(100 * max(v['exact'] for v in tc.values()), 0)}\\%"
    pl = load('plateau') or {}
    if pl:
        macros['PlateauPlan'] = f"{hu(100 * pl['plan'], 2)}\\%"
        macros['PlateauHitOne'] = f"{hu(pl['mult|1.0']['hit_plateau'], 2)}"; macros['PlateauHitMargin'] = f"{hu(pl['mult|1.25']['hit_plateau'], 2)}"
        macros['PlateauSynOne'] = f"{hu(pl['mult|1.0']['hit_synthetic'], 2)}"
        if 'cert' in pl:
            macros['PlateauCertBeta'] = f"{hu(100 * pl['cert']['beta'], 2)}\\%"; macros['PlateauCertExact'] = f"{hu(100 * pl['cert']['exact'], 2)}\\%"
    v = r['1.0|0.01|bayess|0.0']; macros['RqFiveQdagBay'] = f"{hu(v['qdag'], 3)}"


def th_tol(pop, q, ms, eps, kind):
    """Theoretical success threshold for the +-10% hit criterion: the step must clear ceil(0.9 m*)."""
    _, c = consts(eps)
    m9 = int(math.ceil(0.9 * ms))
    F = pop.F(m9 - 1)
    if kind == 'ipa':
        return max(0.0, (F - q) / F)
    return max(0.0, (F - q) / (F + c))


_CC = {}
def CC():
    if 'p' not in _CC:
        _CC['p'] = Pop('creditcard')
    return _CC['p']


def tab_rq3os():
    """One-sided success and edge rate on the RQ3 grid (extended version) + macros."""
    r = load('rq3raw')
    if not r:
        return
    ratios = [1.0, 1.5, 2.0, 3.0]
    atts = [('cliff', r'\cliff{}'), ('scliff', r'\scliff{}'), ('maxval', r'\textsc{MGA}'), ('ipa', 'IPA'), ('fga', r'\textsc{FGA-R}')]
    lines = []
    for proto in ['nbs', 'bayess', 'sgd']:
        first = True
        for att, lab in atts:
            cells = []
            for rr in ratios:
                v = r[f'{proto}|{att}|{rr}']
                cells += [f"{hu(v['hit_mean'], 2)}", f"{hu(v['ge_mean'], 2)}", f"{hu(v['edge_mean'], 2)}"]
            pl = (r'\multirow{5}{*}{' + PNAME[proto] + '}') if first else ''
            first = False
            lines.append(f'{pl} & {lab} & ' + ' & '.join(cells) + r'\\')
        lines.append(r'\midrule')
    lines = lines[:-1]
    s = r"""\begin{table}[t]
\centering
\caption{RQ3: two-sided (hit) versus one-sided (reach) success, and estimates in the top half of the domain (edge).}
\label{tab:rq3os}
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{ll ccc ccc ccc ccc}
\toprule
\hd & & \multicolumn{3}{c}{$1\times$} & \multicolumn{3}{c}{$1.5\times$} & \multicolumn{3}{c}{$2\times$} & \multicolumn{3}{c}{$3\times$}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}
\hd & attack & hit & reach & edge & hit & reach & edge & hit & reach & edge & hit & reach & edge\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_rq3os.tex', 'w').write(s)
    g = lambda p_, a_, rr: r[f'{p_}|{a_}|{rr}']
    # one-sided success of MGA vs CLIFF from beta* on (NBS, BayeSS)
    hi = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
    nb = ['nbs', 'bayess']
    macros['OsMgaReachOneLo'] = f"{hu(min(g(p_, 'maxval', 1.0)['ge_mean'] for p_ in nb), 2)}"
    macros['OsMgaReachOneHi'] = f"{hu(max(g(p_, 'maxval', 1.0)['ge_mean'] for p_ in nb), 2)}"
    macros['OsCliffReachOneLo'] = f"{hu(min(g(p_, 'cliff', 1.0)['ge_mean'] for p_ in nb), 2)}"
    macros['OsCliffReachOneHi'] = f"{hu(max(g(p_, 'cliff', 1.0)['ge_mean'] for p_ in nb), 2)}"
    macros['OsBothReachMin'] = f"{hu(min(g(p_, a_, rr)['ge_mean'] for p_ in nb for a_ in ['cliff', 'maxval'] for rr in [1.5, 2.0, 3.0]), 2)}"
    macros['OsMgaEdgeMin'] = f"{hu(min(g(p_, 'maxval', rr)['edge_mean'] for p_ in ['nbs', 'bayess'] for rr in [1.5, 2.0, 3.0]), 2)}"
    macros['OsCliffEdgeMax'] = f"{hu(max(g(p_, a_, rr)['edge_mean'] for p_ in ['nbs', 'bayess', 'sgd'] for a_ in ['cliff', 'scliff'] for rr in [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]), 2)}"
    macros['OsMgaEdgeSgd'] = f"{hu(g('sgd', 'maxval', 3.0)['edge_mean'], 2)}"
    macros['OsMgaEdgeSgdTwo'] = f"{hu(g('sgd', 'maxval', 2.0)['edge_mean'], 2)}"
    macros['OsMgaReachSgd'] = f"{hu(g('sgd', 'maxval', 1.5)['ge_mean'], 2)}"
    macros['OsIpaReachOne'] = f"{hu(max(g(p_, 'ipa', 1.0)['ge_mean'] for p_ in ['nbs', 'bayess']), 2)}"
    macros['OsFgaReachMax'] = f"{hu(max(g(p_, 'fga', rr)['ge_mean'] for p_ in ['nbs', 'bayess'] for rr in [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]), 2)}"
    same = [v.get('same_as_rq3') for k, v in r.items() if isinstance(v, dict) and 'same_as_rq3' in v]
    print('rq3raw reproduces rq3:', sum(bool(x) for x in same), 'of', len(same))


def tab_plan3():
    """Planning with a wrong guess of x_q (extended version) + macros."""
    r = load('plan3')
    r2 = load('plan2')
    if not r:
        return
    rows = [('nbs', 10**6), ('bayess', 10**6), ('sgd', 10**6), ('sgd', 10**7), ('hh', 10**7)]
    kaps = [2, 5, 10, 1000]
    lines = []
    allhit = {k: [] for k in kaps}
    allhit1 = {k: [] for k in kaps}
    ratio = {k: [] for k in kaps}
    byp = {}
    for proto, n in rows:
        cells = []
        for d in ['creditcard', 'flights', 'power']:
            for k in kaps:
                if k == 10:
                    cal = r2.get(f'cal|{proto}|{n}|{d}'); v = r2.get(f'atk|{proto}|{n}|{d}|1.25'); v1 = r2.get(f'atk|{proto}|{n}|{d}|1.0')
                else:
                    cal = r.get(f'cal|{proto}|{n}|{d}|{k}'); v = r.get(f'atk|{proto}|{n}|{d}|{k}|1.25'); v1 = r.get(f'atk|{proto}|{n}|{d}|{k}|1.0')
                if cal is None or v is None:
                    cells.append('--'); continue
                allhit[k].append(v['hit_mean']); allhit1[k].append(v1['hit_mean'] if v1 else float('nan'))
                c10 = r2.get(f'cal|{proto}|{n}|{d}')
                ratio[k].append(cal / c10)
                grp = 'Sgd' if proto == 'sgd' else 'Other'
                byp.setdefault((k, grp), []).append((v['hit_mean'], cal / c10))
                col = 'good' if v['hit_mean'] >= 0.9 else ('mid' if v['hit_mean'] >= 0.5 else 'bad')
                cells.append(f"\\cellcolor{{{col}}}{hu(100 * cal, 2)}/{hu(v['hit_mean'], 2)}")
        lab = PNAME[proto] + (r' \textsc{(T-Cliff)}' if proto == 'hh' else '') + f' ($10^{{{int(math.log10(n))}}}$)'
        lines.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    s = r"""\begin{table*}[t]
\centering
\caption{RQ3: planning with guessed inflation $\hat\kappa$: plan (\%) / hit rate with the $25\%$ margin.}
\label{tab:plan3}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{l cccc cccc cccc}
\toprule
\hd & \multicolumn{4}{c}{CreditCard} & \multicolumn{4}{c}{Flights} & \multicolumn{4}{c}{Power}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
\hd protocol ($n$) & $\hat\kappa{=}2$ & $5$ & $10$ & $1000$ & $2$ & $5$ & $10$ & $1000$ & $2$ & $5$ & $10$ & $1000$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f'{P}/tables/tab_plan3.tex', 'w').write(s)
    for k, nm in [(2, 'Two'), (5, 'Five'), (1000, 'Inf')]:
        if allhit[k]:
            macros[f'PlanK{nm}HitMin'] = f"{hu(min(allhit[k]), 2)}"
            macros[f'PlanK{nm}HitOneMin'] = f"{hu(min(allhit1[k]), 2)}"
            macros[f'PlanK{nm}N'] = f"{sum(h >= 0.9 for h in allhit[k])}"
            macros[f'PlanK{nm}Tot'] = f"{len(allhit[k])}"
            macros[f'PlanK{nm}RatioLo'] = f"{hu(min(ratio[k]), 2)}"
            macros[f'PlanK{nm}RatioHi'] = f"{hu(max(ratio[k]), 2)}"
            for grp in ['Sgd', 'Other']:
                L_ = byp.get((k, grp), [])
                if L_:
                    macros[f'PlanK{nm}{grp}HitLo'] = f"{hu(min(h for h, _ in L_), 2)}"
                    macros[f'PlanK{nm}{grp}HitHi'] = f"{hu(max(h for h, _ in L_), 2)}"
                    macros[f'PlanK{nm}{grp}RatioLo'] = f"{hu(min(x for _, x in L_), 2)}"
                    macros[f'PlanK{nm}{grp}RatioHi'] = f"{hu(max(x for _, x in L_), 2)}"


def tab_plan():
    r = load('plan2')
    if not r:
        return
    rows = [('nbs', 10**6), ('bayess', 10**6), ('sgd', 10**6), ('sgd', 10**7), ('hh', 10**7)]
    lines, one, onefive = [], [], []
    for proto, n in rows:
        cells = []
        for d in ['creditcard', 'flights', 'power']:
            cal = r.get(f'cal|{proto}|{n}|{d}')
            v = r.get(f'atk|{proto}|{n}|{d}|1.0')
            if cal is None or v is None:
                cells += ['', '', '']
                continue
            w = r.get(f'atk|{proto}|{n}|{d}|1.25')
            cells += [f'{hu(100 * cal, 2)}', hitcell(v['hit_mean'], v['hit_std']), hitcell(w['hit_mean'], w['hit_std'])]
            one.append(v['hit_mean'])
            onefive.append(w['hit_mean'])
        lab = PNAME[proto] + (r' \textsc{(T-Cliff)}' if proto == 'hh' else '') + f' ($10^{{{int(math.log10(n))}}}$)'
        lines.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    s = r"""\begin{table}[t]
\centering
\caption{RQ3: budget planned on synthetic data (\%) and hit rate on the real data at $1\times$ and $1.25\times$ the plan.}
\label{tab:plan}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{l ccc ccc ccc}
\toprule
\hd & \multicolumn{3}{c}{CreditCard} & \multicolumn{3}{c}{Flights} & \multicolumn{3}{c}{Power}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\hd & & \multicolumn{2}{c}{hit rate$\uparrow$} & & \multicolumn{2}{c}{hit rate$\uparrow$} & & \multicolumn{2}{c}{hit rate$\uparrow$}\\
\hd protocol ($n$) & plan & $1\times$ & $1.25\times$ & plan & $1\times$ & $1.25\times$ & plan & $1\times$ & $1.25\times$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_plan.tex', 'w').write(s)
    if one:
        macros['RqThreePlanOneLo'] = f"{hu(min(one), 2)}"
        macros['RqThreePlanOneMed'] = f"{hu(float(np.median(one)), 2)}"
        macros['RqThreePlanN'] = f"{sum(x >= 0.9 for x in one)}"
        macros['RqThreePlanTot'] = f"{len(one)}"
    if onefive:
        macros['RqThreePlanFiveLo'] = exs(min(onefive))
    hhp = [r[f'cal|hh|{10**7}|{d}'] for d in ['creditcard', 'flights', 'power'] if f'cal|hh|{10**7}|{d}' in r]
    if hhp:
        macros['RqThreePlanhhbig'] = f"{hu(100 * min(hhp), 2)}--{hu(100 * max(hhp), 2)}\\%"


def extra_macros():
    L = load('lead') or {}
    if L:
        def g(d, b, k):
            return L.get(f'{d}|{b}', {}).get(k)
        macros['LeadLamLo'] = f"{hu(min(v['lam'] for v in L.values()), 2)}"; macros['LeadLamHi'] = f"{hu(max(v['lam'] for v in L.values()), 2)}"
        for b, nm in [(0.01, 'One'), (0.012, 'OneTwo'), (0.014, 'OneFour')]:
            hs = [v['lead_hit'][0] for v in L.values() if abs(v['beta'] - b) < 1e-9]
            ps = [v['plain_hit'][0] for v in L.values() if abs(v['beta'] - b) < 1e-9]
            if hs:
                macros[f'Lead{nm}Lo'] = f"{hu(min(hs), 2)}"; macros[f'Lead{nm}Hi'] = f"{hu(max(hs), 2)}"
                macros[f'Plain{nm}Hi'] = f"{hu(max(ps), 2)}"
    LK = load('leadk') or {}
    if LK:
        hk = [v['hit'] for v in LK.values() if v['khat'] == 1000]
        macros['LeadKInfLo'] = f"{hu(min(hk), 2)}"; macros['LeadKInfHi'] = f"{hu(max(hk), 2)}"
        hm = [v['hit'] for v in LK.values() if v['khat'] in (5, 10, 20)]
        macros['LeadKMidLo'] = f"{hu(min(hm), 2)}"; macros['LeadKMidHi'] = f"{hu(max(hm), 2)}"
        for kk, nm in [(5, 'Five'), (20, 'Twenty'), (100, 'Hundred')]:
            hh = [v['hit'] for v in LK.values() if v['khat'] == kk]
            if hh:
                macros[f'LeadK{nm}Lo'] = f"{hu(min(hh), 2)}"; macros[f'LeadK{nm}Hi'] = f"{hu(max(hh), 2)}"
    H = load('hh8') or {}
    if H:
        ds = ['creditcard', 'flights', 'power', 'pareto']
        macros['HHEightSevenLo'] = f"{hu(min(H[f'{d}|tcliff|0.0075']['hit_mean'] for d in ds), 2)}"
        macros['HHEightHalfHi'] = f"{hu(max(H[f'{d}|tcliff|0.005']['hit_mean'] for d in ds), 2)}"
        macros['HHEightGenCC'] = f"{hu(H['creditcard|cliff|0.01']['hit_mean'], 2)}"
        macros['HHEightClean'] = f"{hu(min(H[f'{d}|none|0.0']['rank_mean'] for d in ds), 3)}--{hu(max(H[f'{d}|none|0.0']['rank_mean'] for d in ds), 3)}"
    Hf = load('half') or {}
    if Hf:
        for n, nm in [(10**6, 'Six'), (10**7, 'Seven'), (10**8, 'Eight')]:
            ok = [v['kappa'] for k, v in Hf.items() if k.startswith(f'kap|{n}|') and v['hit_mean'] >= 0.9]
            macros[f'HalfKap{nm}'] = f"{max(ok):g}" if ok else '1'
            # predicted largest feasible kappa: pred budget <= 0.5%
            pk = []
            for k, v in Hf.items():
                if k.startswith(f'kap|{n}|'):
                    pv = pred(CC(), 0.99, v['mstar'], 1.0, v['eta_lo'], eta_up=v['eta_up'])
                    if pv <= 0.005:
                        pk.append(v['kappa'])
            macros[f'HalfPred{nm}'] = f"{max(pk):g}" if pk else '1'
        for e, nm in [(0.25, 'Quarter'), (0.5, 'Half'), (1.0, 'One')]:
            v = Hf.get(f'eps|{e}')
            if v:
                macros[f'HalfEps{nm}'] = f"{hu(v['hit_mean'], 2)}"


def pred_macros():
    if RQ4ERR:
        macros['PredErrRqFour'] = f"{hu(max(RQ4ERR), 2)}"
        macros['PredErrCentral'] = f"{hu(max(e for q, e in RQ4ERRQ if q <= 0.75), 2)}"
    if T3ERR:
        macros['PredErrTthree'] = f"{hu(max(e for q, e in T3ERR), 2)}"
    big = [(p, n, m, b) for p, n, m, b in NPRED if n >= 3 * 10**5 and p in ('nbs', 'bayess')]
    small = [(p, n, m, b) for p, n, m, b in NPRED if n < 3 * 10**5 and p in ('nbs', 'bayess')]
    sg = [(p, n, m, b) for p, n, m, b in NPRED if n >= 10**6 and p == 'sgd']
    if big:
        macros['PredErrN'] = f"{hu(max(abs(m - b) for p, n, m, b in big), 2)}"
    if small:
        macros['PredErrSmallMax'] = f"{hu(max(abs(m - b) for p, n, m, b in small), 2)}"
    if sg:
        macros['PredErrSgd'] = f"{hu(max(abs(m - b) for p, n, m, b in sg), 2)}"
    sgs = [(m - b) for p, n, m, b in NPRED if p == 'sgd' and n == 3 * 10**5]
    if sgs:
        macros['SgdGapSmall'] = f"{hu(sgs[0], 1)}"
    allerr = [e for q, e in T3ERR] + list(RQ4ERR) + [abs(m - b) for p, n, m, b in big]
    if allerr:
        macros['PredErrAll'] = f"{hu(max(allerr), 2)}"
    if PREDINF:
        macros['PredInfAll'] = 'all' if all(PREDINF) else 'not all'


def round3():
    """Target-specific thresholds, Theorem 2 brackets, budget validation (extended-version table)."""
    from experiments7 import eta_nbs
    q, eps = 0.99, 1.0
    _, c = consts(eps)
    th = {}
    for d in ['creditcard', 'flights', 'power', 'pareto']:
        pop = Pop(d)
        ms = pop.target(q, 10)
        F1 = float(pop.F(ms - 1))
        for e in [0.25, 0.5, 1.0]:
            _, ce = consts(e)
            th[(d, e)] = (F1 - q) / (F1 + ce)
    v1 = [th[(d, 1.0)] for d in ['creditcard', 'flights', 'power', 'pareto']]
    macros['ThrTenLo'] = f"{hu(100 * min(v1), 2)}"
    macros['ThrTenHi'] = f"{hu(100 * max(v1), 2)}\\%"
    macros['ThrTenQuarter'] = f"{hu(100 * th[('creditcard', 0.25)], 2)}\\%"
    macros['ThrTenHalf'] = f"{hu(100 * th[('creditcard', 0.5)], 2)}\\%"
    pop = Pop('creditcard')
    ms = pop.target(q, 10)
    F1, Fm = float(pop.F(ms - 1)), float(pop.F(ms))
    for n, nm in [(10**6, 'Six'), (10**7, 'Seven'), (10**8, 'Eight')]:
        et = eta_nbs(n, eps, pop.B)
        low = (F1 - q - et) / (F1 + c)
        up = max((F1 - q + et) / (F1 + c), (q + et - Fm) / (1 + c - Fm))
        macros[f'OptLow{nm}'] = 'vacuous' if low <= 0 else f"{hu(100 * low, 2)}\\%"
        macros[f'OptUp{nm}'] = f"{hu(100 * up, 2)}\\%"
        macros[f'EtaNbs{nm}'] = f"{hu(100 * et, 2)}\\%"
    r = load('valid')
    if not r:
        return
    ok = [v for v in r.values() if not v['capped']]

    def sel(rt):
        return [v for v in ok if abs(v['ratio'] - rt) < 1e-9]
    h1 = sorted(v['hit'] for v in sel(1.0))
    macros['ValOneLo'] = f"{hu(h1[0], 2)}"
    macros['ValOneHi'] = f"{hu(h1[-1], 2)}"
    macros['ValOneMed'] = f"{hu(float(np.median(h1)), 2)}"
    macros['ValOneBelow'] = str(sum(v['hi'] < 0.9 for v in sel(1.0)))
    h11 = [v['hit'] for v in sel(1.1)]
    macros['ValElevenLo'] = f"{hu(min(h11), 2)}"
    macros['ValElevenN'] = str(sum(v['lo'] > 0.9 for v in sel(1.1)))
    h09 = [v['hit'] for v in sel(0.9)]
    macros['ValNineMed'] = f"{hu(float(np.median(h09)), 2)}"
    macros['ValNineHi'] = f"{hu(max(h09), 2)}"
    macros['ValNineN'] = str(sum(v['hi'] < 0.9 for v in sel(0.9)))
    br = 0
    for v in sel(1.0):
        k = f"{v['q']}|{v['eps']}|{v['kind']}"
        if r[f'{k}|0.9']['hi'] < 0.9 and r[f'{k}|1.1']['lo'] > 0.9:
            br += 1
    macros['ValBracketN'] = str(br)
    macros['ValTot'] = str(len(sel(1.0)))
    macros['ValCapped'] = str(len([v for v in r.values() if v['capped'] and abs(v['ratio'] - 1) < 1e-9]))
    qs, epss = [0.9, 0.95, 0.99, 0.999], [0.25, 0.5, 1.0, 2.0, 4.0]

    def f2(x):
        return f'{hu(x, 2)}' if x >= 0.995 else f'{hu(x, 2)}'.lstrip('0')
    lines = []
    for qq in qs:
        first = True
        for kind, lab in [('cliff', r'\cliff{}'), ('ipa', 'IPA')]:
            cells = []
            for e in epss:
                k = f'{qq}|{e}|{kind}'
                if r[f'{k}|1.0']['capped']:
                    cells.append(r'$>$range')
                    continue
                cells.append('/'.join(f2(r[f'{k}|{rt}']['hit']) for rt in [0.9, 1.0, 1.1]))
            qlab = f'\\multirow{{2}}{{*}}{{{qq:g}}}' if first else ''
            first = False
            lines.append(f"{qlab} & {lab} & " + ' & '.join(cells) + r'\\')
        lines.append(r'\midrule')
    lines = lines[:-1]
    head = ' & '.join([f'$\\eps={e:g}$' for e in epss])
    s = r"""\begin{table}[t]
\centering
\caption{Validation of \cref{tab:rq1b}: $\beta_{90}/\hat\beta$ / $\beta_{\mathrm{val}}/\hat\beta$ (1000 runs per budget); ``pl.'': plateau.}
\label{tab:valid}
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{cl""" + 'c' * len(epss) + r"""}
\toprule
\hd $q$ & attack & """ + head + r"""\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_valid.tex', 'w').write(s)


def round4():
    """Fine-grid budget validation (replaces tab_valid) + SGD-SN control on continuous data."""
    r = load('valid2')
    if r:
        ok = {k: v for k, v in r.items() if not v['capped'] and v['r90'] is not None}
        lowq = [v for v in ok.values() if v['q'] <= 0.99]
        hiq = [v for v in ok.values() if v['q'] > 0.99]
        macros['ValDevLow'] = f"{hu(100 * max(abs(v['r90'] - 1) for v in lowq), 1)}\\%"
        macros['ValDevHigh'] = f"{hu(100 * max(abs(v['r90'] - 1) for v in hiq), 0)}\\%"
        macros['ValRvalLow'] = f"{hu(max(v['rval'] for v in lowq), 2)}"
        macros['ValRvalHigh'] = f"{hu(max(v['rval'] for v in hiq), 2)}"
        macros['ValNLow'] = str(len(lowq))
        pl = r['0.99|0.25|ipa']
        hv = pl['hit'] + [h for _, h in pl.get('extra', [])]
        macros['PlateauIpaLo'] = f"{hu(min(hv), 3)}"
        macros['PlateauIpaHi'] = f"{hu(max(hv), 3)}"
        macros['PlateauGridLo'] = f"{hu(min(pl['hit']), 3)}"
        macros['PlateauGridHi'] = f"{hu(max(pl['hit']), 3)}"
        macros['PlateauGridBLo'] = f"{hu(100 * pl['beta_hat'] * min(pl['ratios']), 1)}\\%"
        macros['PlateauGridBHi'] = f"{hu(100 * pl['beta_hat'] * max(pl['ratios']), 1)}\\%"
        macros['PlateauExtraLo'] = f"{hu(min(h for _, h in pl['extra']), 3)}"
        macros['PlateauExtraHi'] = f"{hu(max(h for _, h in pl['extra']), 3)}"
        macros['PlateauExtraBLo'] = f"{hu(100 * min(b for b, _ in pl['extra']), 0)}\\%"
        macros['PlateauExtraBHi'] = f"{hu(100 * max(b for b, _ in pl['extra']), 0)}\\%"
        caps = {k: v for k, v in r.items() if v['capped']}
        qs, epss = [0.9, 0.95, 0.99, 0.999], [0.25, 0.5, 1.0, 2.0, 4.0]
        lines = []
        for qq in qs:
            first = True
            for kind, lab in [('cliff', r'\cliff{}'), ('ipa', 'IPA')]:
                cells = []
                for e in epss:
                    k = f'{qq}|{e}|{kind}'
                    v = r[k]
                    if v['capped']:
                        hs = [h for _, h in v['pts']]
                        cells.append(f"pl.\\ {hu(min(hs), 2)}--{hu(max(hs), 2)}".replace('0.', '.'))
                    elif v['r90'] is None:
                        cells.append(f"pl.\\ {macros['PlateauIpaLo']}--{macros['PlateauIpaHi']}".replace('0.', '.'))
                    else:
                        cells.append(f"{hu(v['r90'], 2)}/{hu(v['rval'], 2)}")
                qlab = f'\\multirow{{2}}{{*}}{{{qq:g}}}' if first else ''
                first = False
                lines.append(f"{qlab} & {lab} & " + ' & '.join(cells) + r'\\')
            lines.append(r'\midrule')
        lines = lines[:-1]
        head = ' & '.join([f'$\\eps={e:g}$' for e in epss])
        t = r"""\begin{table}[t]
\centering
\caption{Validation of \cref{tab:rq1b}: $\beta_{90}/\hat\beta$ / $\beta_{\mathrm{val}}/\hat\beta$ (1000 runs per budget); ``pl.'': plateau.}
\label{tab:valid}
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{cl""" + 'c' * len(epss) + r"""}
\toprule
\hd $q$ & attack & """ + head + r"""\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
        open(f'{P}/tables/tab_valid.tex', 'w').write(t)
    c = load('sgdctl')
    if c:
        macros['CtlDiscClean'] = f"{hu(c['disc|0.0']['cov'], 3)}"
        macros['CtlLogClean'] = f"{hu(c['log|0.0']['cov'], 3)}"
        macros['CtlRawClean'] = f"{hu(c['raw|0.0']['cov'], 3)}"
        macros['CtlLogHalf'] = f"{hu(c['log|0.005']['cov'], 2)}"
        macros['CtlRawHalf'] = f"{hu(c['raw|0.005']['cov'], 2)}"
        macros['CtlRawMax'] = f"{hu(max(c[f'raw|{b}']['cov'] for b in [0.005, 0.01, 0.02, 0.05]), 2)}"
        macros['CtlRawTgtOne'] = f"{hu(c['raw|0.01']['covt'], 2)}"
        macros['CtlRawWidthOne'] = f"{hu(c['raw|0.01']['width'] / c['raw|0.0']['width'], 0)}"
        macros['CtlLogInflOne'] = f"{hu(c['log|0.01']['infl_mean'], 1)}"
        macros['CtlWidthRatio'] = f"{hu(c['log|0.01']['width'] / c['log|0.0']['width'], 1)}"
        modes = [('disc', 'log, discrete'), ('log', 'log, continuous'), ('raw', 'raw, continuous')]
        rows = []
        for bb in [0.0, 0.005, 0.01, 0.02, 0.05]:
            cells = []
            for mode, _ in modes:
                v = c.get(f'{mode}|{bb}')
                if v is None:
                    cells.append('--')
                elif bb == 0:
                    cells.append(f"{hu(v['cov'], 3)} [{hu(v['cov_lo'], 2)}, {hu(v['cov_hi'], 2)}]")
                else:
                    w0 = c[f'{mode}|0.0']['width']
                    cells.append(f"{hu(v['cov'], 2)} / {hu(v['covt'], 2)} / {hu(v['infl_mean'], 1)} / {hu(v['width'] / w0, 1)}")
            lab = 'clean' if bb == 0 else f'{100 * bb:g}\\%'
            rows.append(lab + ' & ' + ' & '.join(cells) + r'\\')
        t = r"""\begin{table}[t]
\centering
\caption{RQ6(b) control: coverage by \textsf{SGD-SN} with log- and raw-scale iterates.}
\label{tab:sgdctl}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lccc}
\toprule
\hd $\beta$ & log, discrete & log, continuous & raw, continuous\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
        open(f'{P}/tables/tab_sgdctl.tex', 'w').write(t)
    pth = load('path')
    if pth:
        macros['PathTwo'] = f"{hu(100 * pth['b2'], 2)}\\%"
        macros['PathSeven'] = f"{hu(100 * pth['b7'], 2)}\\%"


def round5():
    """HH non-negativity audit: T-CLIFF vs spread T-CLIFF."""
    r = load('hhaudit')
    if not r:
        return
    ds = ['creditcard', 'flights', 'power', 'pareto']
    macros['HHAudTcMin'] = hu(min(r[f'{d}|10000000|0.015|tcliff']['flag'] for d in ds if d != 'creditcard'), 2)
    macros['HHAudTcCC'] = hu(r['creditcard|10000000|0.015|tcliff']['flag'], 2)
    macros['HHAudSpMax'] = hu(max(v['flag'] for v in r.values() if v['attack'] == 'tcliffs'), 2)
    macros['HHAudTcCCBig'] = hu(r['creditcard|100000000|0.0075|tcliff']['flag'], 3)
    macros['HHAudCleanMax'] = hu(max(v['flag'] for v in r.values() if v['attack'] == 'clean'), 3)
    rb = load('hhaudit_big') or {}
    if rb:
        macros['HHAudBigFlag'] = hu(rb['flights|100000000|0.015|tcliffs']['flag'], 2)
        macros['HHAudGigFlag'] = hu(rb['flights|1000000000|0.0075|tcliffs']['flag'], 2)
    macros['HHAudCleanMin'] = hu(min(v['flag'] for v in r.values() if v['attack'] == 'clean'), 3)
    macros['HHAudHitDiff'] = hu(max(abs(r[k]['hit'] - r[k.replace('tcliffs', 'tcliff')]['hit']) for k in r if k.endswith('tcliffs')), 2)
    rows = []
    for d in ds:
        first = True
        for n, b in [(10**7, 0.0), (10**7, 0.01), (10**7, 0.015), (10**8, 0.0), (10**8, 0.0075)]:
            cells = []
            for a in (['clean'] if b == 0 else ['tcliff', 'tcliffs']):
                v = r[f'{d}|{n}|{b}|{a}']
                cells.append(f"{hu(v['hit'], 2)} / {hu(v['flag'], 3)}")
            if b == 0:
                cells = ['---', '---', cells[0]]
            else:
                cells = cells + ['']
            lab = (DNAME[d] if first else '')
            first = False
            rows.append(f"{lab} & $10^{{{int(math.log10(n))}}}$ & {100 * b:g}\\% & " + ' & '.join(cells[:2]) + (f" & {cells[2]}" if cells[2] else ' & ') + r'\\')
        rows.append(r'\midrule')
    if rb:
        for n, b in [(10**8, 0.015), (10**9, 0.0075)]:
            v = rb[f'flights|{n}|{b}|tcliffs']
            rows.append(f"Flights & $10^{{{int(math.log10(n))}}}$ & {100 * b:g}\\% & --- & {hu(v['hit'], 2)} / {hu(v['flag'], 3)} & \\\\")
    else:
        rows = rows[:-1]
    t = r"""\begin{table}[t]
\centering
\caption{\HH{} non-negativity audit: hit rate / fraction of runs flagged.}
\label{tab:hhaudit}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lccccc}
\toprule
\hd dataset & $n$ & $\beta$ & \textsc{T-Cliff} & spread & clean\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    open(f'{P}/tables/tab_hhaudit.tex', 'w').write(t)


def write_macros():
    with open(f'{P}/numbers.tex', 'w') as f:
        f.write('% auto-generated by gen_tables.py\n')
        for k, v in sorted(macros.items()):
            name = ''.join(ch for ch in k if ch.isalpha())
            f.write(f'\\newcommand{{\\{name}}}{{{v}}}\n')
        tg = max(g for q, g in TOLGAP if q >= 0.95)
        f.write(f'\\newcommand{{\\TolGap}}{{{hu(tg, 2)}}}\n')



def _pava(y):
    y = [float(v) for v in y]; w = [1.0] * len(y); blocks = []
    for v in y:
        blocks.append([v, 1.0, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            a, b = blocks[-2], blocks[-1]
            m = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]); blocks[-2:] = [[m, a[1] + b[1], a[2] + b[2]]]
    out = []
    for m, _, c in blocks:
        out += [m] * c
    return out


def round6():
    """Round-8 review fixes: per-kappa 90%-hit budgets (Fig. 2(a,b)) and CTQ's value price."""
    r = load('rq1a')
    if r:
        for proto, tag in [('nbs', 'Nbs'), ('bayess', 'Bay')]:
            for k, kt in [(2, 'Two'), (50, 'Fifty')]:
                pts = sorted((float(key.split('|')[2]), v['hit_mean']) for key, v in r.items() if key.startswith(f'{proto}|{k}|'))
                b = [x for x, _ in pts]; h = _pava([y for _, y in pts])
                j = next(i for i, v in enumerate(h) if v >= 0.9)
                b90 = b[j] if j == 0 or h[j] == h[j - 1] else b[j - 1] + (0.9 - h[j - 1]) * (b[j] - b[j - 1]) / (h[j] - h[j - 1])
                macros[f'KapBninety{tag}{kt}'] = f"{hu(100 * b90, 2)}\\%"
                macros[f'_b90_{proto}_{k}'] = b90
            macros[f'KapBninetyRatio{tag}'] = f"{hu(macros.pop(f'_b90_{proto}_50') / macros.pop(f'_b90_{proto}_2'), 1)}"
    if r:
        _lv = []
        for proto in ['nbs', 'bayess']:
            for k in [2, 5, 10, 50]:
                pts = sorted((float(key.split('|')[2]), v['hit_mean'], v['theory']) for key, v in r.items() if key.startswith(f'{proto}|{k}|'))
                _lv.append([x for x in pts if x[0] < pts[0][2]][-1][1])
        macros['LeftOfLineLo'] = f"{hu(min(_lv), 2)}"; macros['LeftOfLineHi'] = f"{hu(max(_lv), 2)}"
    r4 = load('rq4')
    if r4:
        macros['TailParOneGeo'] = f"{hu(2 ** r4['tail|pareto|1.1|0.0025']['log2infl_mean'], 2)}"
        macros['TailParOnePred'] = f"{hu(r4['tail|pareto|1.1|0.0025']['theory_infl'], 2)}"
    from ldpq import beta_star as _bs3
    from runexp import getpop as _gp3
    _p3 = _gp3('creditcard'); _m3 = _p3.target(0.99, 3)
    macros['HalfThrThree'] = f"{hu(100 * _bs3(_p3, 0.99, int(math.ceil(0.9 * _m3)), 1.0), 2)}\\%"
    _h = load('half')
    if _h:
        macros['HalfTwoHit'] = f"{hu(_h['kap|100000000|2']['hit_mean'], 2)}"; macros['HalfThreeHit'] = f"{hu(_h['kap|100000000|3']['hit_mean'], 2)}"
    c = load('cert')
    if c:
        v = c['ctq|1000000|1.0|0.01|0.0']
        macros['CtqCleanInfl'] = f"{hu(v['infl_mean'], 2)}"
        from runexp import getpop
        pop = getpop('creditcard')
        macros['CtqLowInfl'] = f"{hu(pop.quantile(v['qlow']) / pop.quantile(0.99), 2)}"
        macros['CtqQlow'] = f"{hu(v['qlow'], 3)}"
    from runexp import getpop as _gp
    from ldpq import beta_star as _bs
    import math as _m
    _p = _gp('creditcard')
    _ms = _p.target(0.99, 2)
    macros['ThKappatwoWin'] = f"{hu(100 * _bs(_p, 0.99, int(_m.ceil(0.9 * _ms)), 1.0), 2)}\\%"

if __name__ == '__main__':
    for fn in [tab_data, fig_rq1a, tab_rq1b, tab_rq2, rq3_out, tab_rq3os, rq4_out, rq5_out, rq5c_out, rq6_out, tab_plan, tab_plan3, extra_macros, pred_macros, round3, round4, round5, round6]:
        try:
            fn()
            print('ok', fn.__name__)
        except Exception as e:
            import traceback; traceback.print_exc()
            print('FAIL', fn.__name__, e)
    write_macros()
