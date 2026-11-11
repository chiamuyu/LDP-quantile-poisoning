import numpy as np, pandas as pd, json, sys
raw = 'data_raw'
out = 'data'
import os; os.makedirs(out, exist_ok=True)
B = 2**20
# CreditCard: transaction amount (EUR). unit = 0.05 EUR so max 25691.16 -> 513823 < 2^20
cc = pd.read_csv(f'{raw}/creditcard.csv', usecols=['Amount','Class'])
amt = np.round(cc['Amount'].values/0.05).astype(np.int64) + 1
np.save(f'{out}/creditcard_vals.npy', amt); np.save(f'{out}/creditcard_class.npy', cc['Class'].values.astype(np.int8))
# Flights: arrival delay (minutes); shift so that min maps to 1
fl = pd.read_json(f'{raw}/flights-200k.json')
d = fl['delay'].values.astype(np.int64)
fd = d - d.min() + 1
np.save(f'{out}/flights_vals.npy', fd)
# Power: global active power (kW) -> watts
pw = pd.read_csv(f'{raw}/household_power_consumption.txt', sep=';', usecols=['Global_active_power'], na_values=['?'], low_memory=False)
p = pw['Global_active_power'].dropna().values
pv = np.round(p*1000).astype(np.int64) + 1
np.save(f'{out}/power_vals.npy', pv)
for name, v in [('creditcard', amt), ('flights', fd), ('power', pv)]:
    qs = np.quantile(v, [0.5, 0.9, 0.95, 0.99, 0.999])
    print(name, len(v), 'min', v.min(), 'max', v.max(), 'q50/90/95/99/99.9', qs, 'max/q99', v.max()/qs[3])
print('fraud frac', cc['Class'].mean(), 'fraud amount quantiles', np.quantile(amt[cc['Class'].values==1],[0.5,0.9,0.99]))
print('delay raw min', d.min(), 'raw q99', np.quantile(d,0.99))
