"""Publication layouts, exported without computational or filesystem dependencies."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
METHODS = ["reimherr", "wiesenfarth", "jones", "cp"]
LABELS = ["Reimherr", "Wiesenfarth", "Jones", "CP"]
COLORS = ["#0072B2", "#D55E00", "#333333", "#AA4499"]
NORM = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)

def grid_edges(values, positive=False):
    values = np.asarray(sorted(np.unique(values)), dtype=float)
    if positive:
        mids = np.sqrt(values[:-1] * values[1:])
        return np.r_[values[0] ** 2 / mids[0], mids, values[-1] ** 2 / mids[-1]]
    mids = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (mids[0] - values[0]), mids, values[-1] + (values[-1] - mids[-1])]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def save(fig, out, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', bbox_inches='tight', dpi=180)
    plt.close(fig)

def heatmaps(f, targets, rowlabels, out, name, title, xkey, ykey, truth=None, mle=None, nig=False):
    fig, axs = plt.subplots(len(targets), 4, figsize=(13.5, 2.45 * len(targets) + 1.0),
                            sharex=True, sharey=True, squeeze=False)
    for i, (target, rowlabel) in enumerate(zip(targets, rowlabels)):
        for j, label in enumerate(LABELS):
            z = f[(f.target == target) & (f.method == METHODS[j])]
            require(not z.duplicated([xkey,ykey]).any(), 'Duplicate plotted cell')
            p = z.pivot(index=ykey, columns=xkey, values='epss').sort_index().sort_index(axis=1)
            require(not p.isna().any().any(), 'Missing heatmap cell')
            ax = axs[i,j]
            mesh = ax.pcolormesh(grid_edges(p.columns),
                                grid_edges(p.index, positive=nig), p.values,
                                norm=NORM, cmap='RdBu_r', rasterized=True)
            if truth:
                ax.scatter(*truth, marker='*', s=52, facecolor='white' if nig else '#FFD166',
                           edgecolor='#222222', linewidth=.7, zorder=4)
            if mle:
                ax.axvline(mle[0], color='.15', ls='--', lw=.8)
                ax.axhline(mle[1], color='.15', ls='--', lw=.8)
                ax.scatter(*mle, marker='o', s=35, facecolors='none', edgecolors='.15', zorder=4)
            if nig:
                ax.set_yscale('log'); ax.set_ylim(.22,5.6); ax.set_xlim(-4.25,4.25)
                ax.set_yticks([.25,.5,1,2,5], ['0.25','0.5','1','2','5'])
                ax.set_xticks([-4,-2,0,2,4]); ax.minorticks_off()
            else:
                ax.set_xticks([-5,-3,-1,1,3]); ax.set_yticks([-5,-1,3,7,11])
            if i == 0: ax.set_title(label, fontsize=15)
            if j == 0: ax.set_ylabel(rowlabel, fontsize=12)
    fig.suptitle(title, fontsize=16, y=.992)
    fig.supxlabel(r'Prior location $\mu_0$' if nig else r'Intercept prior location $\mu_{\beta_0}$', y=.075, fontsize=14)
    fig.supylabel(r'Prior mean $E_\pi(\sigma^2)$' if nig else r'Slope prior location $\mu_{\beta_1}$', x=.012, fontsize=14)
    fig.subplots_adjust(left=.16, right=.99, bottom=.15, top=.94, wspace=.09, hspace=.16)
    cax = fig.add_axes([.29,.035,.5,.014])
    fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[-20,-10,0,10,20],
                 label='Mean signed EPSS' if 'simulation' in name else 'Signed EPSS')
    save(fig,out,name)

def line_atlas(f, targets, titles, out, name, xkey, title, schools=False, simulation=False, limit=20):
    fig, axs = plt.subplots(len(targets),4,figsize=(14,2.7*len(targets)+.65),sharex=True,sharey=True)
    for i, (target, rowtitle) in enumerate(zip(targets,titles)):
        for j, method in enumerate(METHODS):
            ax = axs[i,j]; z = f[(f.target==target)&(f.method==method)].sort_values(xkey)
            require(len(z)==(100 if schools else 82), 'Missing line coordinates')
            ax.axhline(0,color='.6',lw=.7); ax.grid(alpha=.13)
            ax.spines[['top','right']].set_visible(False)
            ax.set_ylim(-limit,limit)
            ax.plot(z[xkey],z.epss,color=COLORS[j],lw=1.5)
            if simulation:
                ax.fill_between(z[xkey],z.epss-z.sd,z.epss+z.sd,color=COLORS[j],alpha=.18,lw=0)
            if schools:
                ax.set_xlim(.25,50); ax.axvline(25,color='.35',ls=':',lw=.9)
                if simulation: ax.axvline(6,color='#B2182B',ls='--',lw=1.05,zorder=4)
                bounds = [(-6,20),(-20,6),(-20,20),(-6,6)][j]
            else:
                ax.set_xlim(1.3,2.3); ax.set_xticks([1.3,1.8,2.3])
                ax.axvline(1.8044444444444445,color='.65',ls='--',lw=.9)
                ax.axvline(1.9,color='.2',ls=':',lw=1)
                bounds = [(-3,40),(-40,5),(-40,40),(-5,5)][j]
            if schools or method=='cp':
                for b in bounds: ax.axhline(b,color=COLORS[j],ls=':',lw=.6,alpha=.5)
            if not simulation and method in ('reimherr','wiesenfarth'):
                hit = z[z.boundary_rate==1]
                ax.scatter(hit[xkey],hit.epss,color=COLORS[j],marker='x',s=20)
            if i==0: ax.set_title(LABELS[j],fontsize=15)
            if j==0: ax.set_ylabel(rowtitle,fontsize=12)
    fig.suptitle(title,fontsize=16)
    fig.supxlabel(r'Prior scale $\tau_0$' if schools else r'Prior location $\mu_0$ (mm)',fontsize=15,y=.025)
    fig.supylabel('Signed EPSS (schools)' if schools else 'Signed EPSS (observations)',fontsize=14,x=.006)
    fig.tight_layout(rect=(.03,.06,1,.95),w_pad=1.1,h_pad=1.1)
    save(fig,out,name)

