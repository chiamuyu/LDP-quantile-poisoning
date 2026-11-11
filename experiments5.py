"""FGA baseline (fine-grained attack of Li et al., USENIX Sec'23, adapted to quantiles) on the RQ3 budget grid."""
import json, time
import numpy as np
from experiments import *
import runexp


def fga_rq3():
    pop = getpop('creditcard')
    ms = pop.target(Q0, KAP0)
    xq = pop.quantile(Q0)
    r = json.load(open(f'{runexp.RES}/rq3.json'))
    bs = r['beta_star']
    cfgs = []
    for proto in ['nbs', 'bayess', 'sgd']:
        for ratio in [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
            cfgs.append(((proto, ratio), tasks_for(proto, 'creditcard', N0, EPS0, Q0, ratio * bs, 'fga', ms, L=xq, seed0=int(ratio * 100) + 9)))
    outs = batch(cfgs)
    for (proto, ratio), o in outs.items():
        s = summarize(o, pop, Q0, ms, TOL)
        s.update(proto=proto, attack='fga', ratio=ratio, beta=ratio * bs)
        r[f'{proto}|fga|{ratio}'] = s
    save('rq3', r)
    log('fga done')


if __name__ == '__main__':
    t = time.time(); fga_rq3(); log('end', f'{time.time() - t:.0f}s')
