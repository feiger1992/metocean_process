import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
deps = ['surface', '0.2H', '0.4H', '0.6H', '0.8H', 'bottom']
deps.reverse()
def read_data(name):
    df = pd.read_clipboard()
    df = pd.melt(df, id_vars=['t', 'type', 'month'], var_name='dep', value_name=name)
    name2 = name + '%'
    df[name2] = (df[name] - df[name].mean()) / df[name].mean()
    result = {}
    result.update({'mean': df.pivot_table(index=['month'], columns=['dep'], values=[name], margins=True,
                                               aggfunc='mean').reindex(
        ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'All']).reindex(['surface','0.2H', '0.4H', '0.6H', '0.8H',  'bottom', 'All'],axis = 1,level = 1),
                        'mean_exceed': df.pivot_table(index=['month'], columns=['dep'], values=[name2], margins=True,
                                                      aggfunc='mean').reindex(
                            ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'All']).reindex(['surface','0.2H', '0.4H', '0.6H', '0.8H',  'bottom', 'All'],axis = 1,level = 1),
                        'std': df.pivot_table(index=['month'], columns=['dep'], values=[name], margins=True,
                                              aggfunc='std').reindex(
                            ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'All']).reindex(['surface','0.2H', '0.4H', '0.6H', '0.8H',  'bottom', 'All'],axis = 1,level = 1),
                        'quan_25': df.pivot_table(index=['month'], columns=['dep'], values=[name], margins=True,
                                                  aggfunc=lambda x: np.percentile(x, 25)).reindex(
                            ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'All']).reindex(['surface','0.2H', '0.4H', '0.6H', '0.8H',  'bottom', 'All'],axis = 1,level = 1),
                        'quan_50': df.pivot_table(index=['month'], columns=['dep'], values=[name], margins=True,
                                                  aggfunc=lambda x: np.percentile(x, 50)).reindex(
                            ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'All']).reindex(['surface','0.2H', '0.4H', '0.6H', '0.8H',  'bottom', 'All'],axis = 1,level = 1),
                        'quan_75': df.pivot_table(index=['month'], columns=['dep'], values=[name], margins=True,
                                                  aggfunc=lambda x: np.percentile(x, 75)).reindex(
                            ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'All']).reindex(['surface','0.2H', '0.4H', '0.6H', '0.8H',  'bottom', 'All'],axis = 1,level = 1)})
    return df, result


def one_data2(name, color, unit, min_v=None, max_v=None):
    df, result = read_data(name)
    if not min_v:
        min_v = int(round(df[name].min()) - 1)
    if not max_v:
        max_v = int(round(df[name].max()) + 1)
    gen_fig(df, 'Annual', name, min_v, max_v, color, unit)
def one_data(name, color, unit, min_v=None, max_v=None):
    df, result = read_data(name)
    if not min_v:
        min_v = int(round(df[name].min()) - 1)
    if not max_v:
        max_v = int(round(df[name].max()) + 1)
    mon_g = df.groupby('month')
    for i, j in mon_g:
        gen_fig(j, i, name, min_v, max_v, color, unit)
    gen_fig(df, 'Annual', name, min_v, max_v, color, unit)
def gen_fig(mon_df, month, name, min_v, max_v, color, unit):
    df = mon_df
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    df = pd.DataFrame({k: df.groupby('dep')[name].get_group(k) for k in deps})
    x = df.plot.hist(ax=ax, density=True, stacked=True, fill=True, bins=np.linspace(min_v, max_v, 40), alpha=.8,
                     grid=False, rwidth=0.8, cmap=color)
    alpha = 1/sum(list(x._stacker_pos_prior.values())[0])
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    font = font_manager.FontProperties(family='DejaVu Sans',
                                       style='normal', size=20)
    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    ax.legend(handles=handles, labels=labels, prop=font)
    ax.set_xlabel(name + unit, fontsize=22)
    ax.set_yticklabels(['{:.2%}'.format(x*alpha) for x in ax.get_yticks()])
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    ax.set_ylabel('Frequence', fontsize=22)
    ax.set_title(name + ' Frequence Distribution' +'(' + month +')', fontsize=25)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.set_facecolor('w')
    fig.savefig(r"C:\Users\刘鹏飞\Desktop\pic\\" + name + ' ' + month + ".png", dpi=200, bbox_inches='tight')

o2 = 'Dissolved O₂'
u = '(µmol/l)'
#one_data(o2, 'rainbow', u)
#one_data('Temperature','rainbow','(℃)')
_,temp = read_data('Temp')
print('*'*20)