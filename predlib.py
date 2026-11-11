"""Finite-sample budget prediction (Remark 1): Thm. 1 with the protocol's accuracy eta, measured on clean runs of a
synthetic surrogate population (never on the evaluated data)."""
import json, math, os
import numpy as np
from ldpq import Pop, consts

RES = 'results'
_ETA, _POPS = {}, {}


def Pp(d):
    if d not in _POPS:
        _POPS[d] = Pop(d)
    return _POPS[d]


def eta_q(proto, n, eps, q, data='pareto', delta=0.1):
    """(1 - delta)-quantile of the rank error max{F(m-1) - q, q - F(m)} of clean runs at level q (Def. 1 with G = F)."""
    if 'r' not in _ETA:
        p = f'{RES}/eta.json'
        _ETA['r'] = json.load(open(p)) if os.path.exists(p) else {}
    key = f'{proto}|{data}|{n}|{float(eps)}|{q}'
    if key not in _ETA['r']:
        return None
    m = np.array(_ETA['r'][key]); p = Pp(data)
    e = np.maximum(p.F(m - 1) - q, q - p.F(m))
    return float(np.quantile(e, 1 - delta))


def sdv(q, eps):
    rho, _ = consts(eps)
    pi = rho + (1 - 2 * rho) * q
    return math.sqrt(pi * (1 - pi))


def eta_hat(proto, n, eps, q, data='pareto'):
    """Measured at q_ref = min(q, 0.9) -- rank errors above q are capped at 1 - q -- and rescaled by the answer std."""
    qr = min(q, 0.9)
    e = eta_q(proto, n, eps, qr, data)
    return None if e is None else e * sdv(q, eps) / sdv(qr, eps)


def pred(pop, q, ms, eps, eta, kind='cliff', cmult=1.0, eta_up=None):
    """Smallest beta meeting both conditions of the proof of Thm. 1 for the +-10% hit criterion."""
    _, c = consts(eps)
    c *= cmult
    m9 = int(math.ceil(0.9 * ms)); mu = min(pop.B, int(math.floor(1.1 * ms)))
    F9, Fu = pop.F(m9 - 1), pop.F(mu)
    if kind == 'tcliff':
        c = 2 * consts(eps)[1]   # OUE debiases a fake 0 to -2c
    lo = (F9 - q + eta) / (F9 + (c if kind != 'ipa' else 0.0))
    gain = (1 + c - Fu) if kind not in ('ipa', 'tcliff') else (1 - Fu)   # IPA and T-Cliff add only beta(1 - F) above m*
    eu = eta if eta_up is None else eta_up
    up = (q + eu - Fu) / gain if gain > 1e-12 else (0.0 if q + eu <= Fu else float('inf'))
    if mu >= pop.B:   # Remark 1: the upper condition is needed only if u < B
        up = 0.0
    return max(lo, up, 0.0)


_ES = {}


def eta_sides(proto, data, n, eps, q, b, delta=0.1):
    """(eta_lo, eta_up) measured on synthetic step curves (stepcal.py / experiments4.step); None if missing."""
    if 'r' not in _ES:
        p = f'{RES}/eta_step.json'
        _ES['r'] = json.load(open(p)) if os.path.exists(p) else {}
    r = _ES['r']
    k = f'{proto}|{data}|{n}|{float(eps)}|{q}|{b}|'
    lo, up = r.get(k + f'lo|{delta}'), r.get(k + f'up|{delta}')
    return (lo, up) if lo is not None and up is not None else (None, None)


def pred_step(pop, data, proto, n, eps, q, ms, kind='cliff'):
    lo, up = eta_sides(proto, data, n, eps, q, ms)
    if lo is None:
        return None
    return pred(pop, q, ms, eps, lo, kind, eta_up=up)
