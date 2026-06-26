import matplotlib.pyplot as plt
import numpy as np


def correlation_plot(cis, figsize=(5, 5), figname=None):
    """Heatmap da matriz de correlação de um DataFrame de indicadores compostos.

    Adaptado de ROpt/composite_indicator/plots.py para uso no Streamlit:
    desenha sobre uma figura controlada (``ax.matshow``) e a retorna, para que
    o app possa exibi-la com ``st.pyplot(fig)``.
    """
    correlation = cis.corr()

    mask = np.triu(np.ones_like(correlation, dtype=bool))
    correlation = correlation.mask(mask)

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(correlation, vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90)
    ax.set_yticks(np.arange(len(correlation.columns)))
    ax.set_yticklabels(correlation.columns)

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            value = correlation.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

    fig.tight_layout()

    if figname is not None:
        fig.savefig(figname)

    return fig

def violin_plot(cis, figsize=(5,5), figname=None):
    plt.close("all")
    fig = plt.figure(figsize=figsize)
    plt.violinplot(cis, showextrema=True)#, showmeans=True, quantiles=[[.25,.75]for _ in all_ci.columns])
    
    mins, quartile1, medians, quartile3, maxs = np.percentile(cis, [0, 25, 50, 75, 100],axis=0)
    # print(mins, quartile1, medians, quartile3, maxs)
    plt.grid()
    ax = plt.gca()
    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='w', s=50, zorder=3)
    ax.vlines(inds, mins, maxs, color='blue', linestyle='-', lw=1)
    ax.vlines(inds, quartile1, quartile3, color='darkblue', linestyle='-', lw=5)
    ax.set_xticks(inds, labels=cis.columns,rotation=60)
    ax.set_xlim(0.25, len(cis.columns) + 0.75)
    
    if figname is not None:
        plt.savefig(figname)

    return fig