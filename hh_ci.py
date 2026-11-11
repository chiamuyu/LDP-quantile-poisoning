"""HH with and without consistency post-processing (CreditCard, CLIFF, n=1e7): isolates the consistency effect."""
import json, time, numpy as np
from multiprocessing import Pool
from runexp import getpop, gethh, summarize, save
from ldpq import run_hh, ATT

def job(a):
    pop = getpop('creditcard'); cache = gethh('creditcard')
    rng = np.random.default_rng(a['seed'])
    m = run_hh(pop, cache, 10**7, 1.0, 0.99, a['beta'], ATT['cliff'] if a['beta'] > 0 else 0, a['ms'], 0, rng=rng, ci=a['ci'])
    return {'m': m}

if __name__ == '__main__':
    pop = getpop('creditcard'); ms = pop.target(0.99, 10)
    res = {}
    t = time.time()
    for ci in [True, False]:
        for b in [0.0, 0.01, 0.02, 0.03]:
            tasks = [dict(seed=1000 + 7919 * i + int(b * 1e4), beta=b, ms=ms, ci=ci) for i in range(40)]
            with Pool(2) as p:
                outs = p.map(job, tasks)
            s = summarize(outs, pop, 0.99, ms, 0.10)
            s.update(ci=ci, beta=b)
            res[f'{ci}|{b}'] = s
            print(ci, b, round(s['rank_mean'], 4), round(s['hit_mean'], 2), round(s['infl_mean'], 2), f'{time.time()-t:.0f}s', flush=True)
    save('hh_ci', res)
