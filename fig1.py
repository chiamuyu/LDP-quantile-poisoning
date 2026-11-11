import numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ldpq import Pop, consts
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
pop=Pop('creditcard'); q=0.99; eps=1.0; beta=0.008
rho,c=consts(eps); xq=pop.quantile(q); ms=pop.target(q,10)
t=np.unique(np.round(np.geomspace(2000,pop.B,1500)).astype(np.int64))
F=pop.F(t)
G=np.where(t<ms,(1-beta)*F-beta*c,(1-beta)*F+beta*(1+c))
fig,ax=plt.subplots(1,2,figsize=(3.45,1.75),gridspec_kw={'width_ratios':[1.55,1]})
a=ax[0]
a.axhspan(q,1.0,color='#d9e8f5',lw=0)
a.plot(t*0.05,F,color='#1f4e79',lw=1.2,label='honest $G=F$')
a.plot(t*0.05,G,color='#c0392b',lw=1.2,label=r'CLIFF, $\beta{=}0.8\%$')
a.axhline(q,color='k',lw=0.6,ls='--')
a.axvline(xq*0.05,color='#1f4e79',lw=0.6,ls=':'); a.axvline(ms*0.05,color='#c0392b',lw=0.6,ls=':')
a.set_xscale('log'); a.set_ylim(0.95,1.012); a.set_xlim(150,52000)
a.set_xlabel('threshold $t$ (EUR)'); a.set_ylabel('debiased $G(t)$')
a.text(1500,0.9925,'band $[q,1]$',fontsize=6,color='#1f4e79')
a.annotate('$x_q$',(xq*0.05*0.93,1.004),fontsize=7,color='#1f4e79',ha='right')
a.annotate('$m^\\star{=}10x_q$',(ms*0.05*0.93,1.0065),fontsize=7,color='#c0392b',ha='right')
a.legend(loc='lower right',frameon=False,handlelength=1.2,borderaxespad=0.2)
b=ax[1]
kap=np.geomspace(1.01,pop.B/xq,200)
mst=np.minimum(pop.B,np.round(kap*xq)).astype(np.int64)
Fm=pop.F(mst-1)
for e,col in [(0.5,'#7d3c98'),(1.0,'#c0392b'),(2.0,'#e67e22')]:
    _,cc=consts(e)
    b.plot(kap,100*(Fm-q)/(Fm+cc),color=col,lw=1.1,label=f'$\\varepsilon={e:g}$')
    b.axhline(100*(1-q)*(1-math.exp(-e)),color=col,lw=0.5,ls='--')
b.plot(kap,100*(Fm-q)/Fm,color='gray',lw=1.0,ls='-.',label='non-private')
b.set_xscale('log'); b.set_xlabel('inflation $m^\\star/x_q$'); b.set_ylabel(r'$\beta^\star$ (%)')
b.set_ylim(0,1.05); b.legend(loc='lower right',frameon=False,handlelength=1.2)
plt.tight_layout(pad=0.15,w_pad=0.4)
import os; os.makedirs('generated/figs', exist_ok=True)
plt.savefig('generated/figs/fig1_insight.pdf')
print('xq',xq,'ms',ms, 'G(ms-1)',(1-beta)*pop.F(ms-1)-beta*c, 'G(ms)',(1-beta)*pop.F(ms)+beta*(1+c))
