import pandas as pd
import numpy as np
import os
from windrose import WindroseAxes
from matplotlib import pyplot  as plt
import matplotlib.cm as cm

plt.rcParams['font.sans-serif']=['SimHei']

dir_in_360b = lambda d: (d - 360 if d >= 360 else d) if d > 0 else  360 + d
dir_in_360 = lambda d: dir_in_360b(d) if (dir_in_360b(d) >= 0 and (dir_in_360b(d) < 360)) else dir_in_360b(
    dir_in_360b(d))
EWSN2d = { "NNW":337.5, "NW":315, "WNW":292.5, "W":270, "WSW":247.5, "SW":225, "SSW":202.5, "S":180, "SSE":157.5, "SE":135, "ESE":112.5,"E":90,"ENE":67.5,"NE":45,"NNE":22.5,"N":0}
def d2EWSN(d):
    if d > 360 or d < 0:
        return d2EWSN(dir_in_360(d))
    if 348.75<d<360:
        return 'N'
    elif d >= 326.25:
        return "NNW"
    elif d >= 303.75:
        return "NW"
    elif d > 281.25:
        return "WNW"
    elif d >= 258.75:
        return "W"
    elif d > 236.25:
        return "WSW"
    elif d >= 213.75:
        return "SW"
    elif d >= 191.25:
        return "SSW"
    elif d >= 168.75:
        return "S"
    elif d >= 146.25:
        return "SSE"
    elif d >= 123.75:
        return "SE"
    elif d >= 101.25:
        return "ESE"
    elif d >= 78.75:
        return "E"
    elif d >= 56.25:
        return "ENE"
    elif d >= 33.75:
        return "NE"
    elif d >= 11.25:
        return "NNE"
    elif d <11.25:
        return "N"
def hist_a_n(n):
    def hist_a(a):
        return np.histogram(a,bins = n)
    return hist_a
class Wind_Wave():
    def __init__(self,filename,if_multiple = False,isEWSN = False,is_SI_v = False):
        #is_SI_v 单位全部为米每秒
        self.is_SI_v = is_SI_v
        self.if_multiple = if_multiple
        self.isEWSN = isEWSN
        if if_multiple:
            df = pd.read_excel(filename,header=[0,1])
        else:
            df = pd.read_excel(filename)
        try:
            df['Dir'] = df['d']
            df = df.drop('d')
        except:
            pass

        if if_multiple:
            df['t'] = df.index
        else:
            try:
                df['t'] = pd.to_datetime(df['t'],errors='ignore')
            except:
                df['t'] = pd.to_datetime(df['time'],errors='ignore')

        df = df.drop_duplicates().dropna()
        df['t'] = df['t'].dt.round('min')
        if isEWSN:
            df = self.convert_EWSN2num(df)

        df = self.convert_num(df)

        self.NanValue = df[pd.isnull(df).any(axis=1) == True]
        self.begin_time,self.end_time = df.t.min(),df.t.max()
        self.data = df
        self.font_dict = {'fontsize':20}
        self.group_month = None
        self.units = {}

        if if_multiple:
            self.items = [i for i in df.columns.levels[0] if i!= 't']
        else:
            self.items = [i for i in df.columns if (i != 't') and (i != 'time') and (i != 'Dir')]
        print("载入文件结束")

    def convert_EWSN2num(self,df):
        for x in df.loc[:, df.columns.get_level_values(1) == 'Dir'].columns:
            df.loc[:, x] = df.loc[:, x].apply(EWSN2d.get)
        return df

    def convert_num(self,df):
        for c in range(len(df.columns)):
            c_name = df.columns.levels[0][c] if self.if_multiple else df.columns[c]
            if not (('t' == c_name) or ('time' == c_name)):
                df.iloc[:, c] = pd.to_numeric(df.iloc[:, c], errors='ignore')
                if df.iloc[:, c].dtype != 'float64':
                    try:
                        df.iloc[:, c] = np.array(df.iloc[:, c], dtype='float64')
                    except:
                        raise ValueError("第" + str(c) + "列中含有无法转化为数值的字符,列标题为"+c_name)
            return df

    def set_save_dir(self,dir):
        os.chdir(dir)
    def plot_data_of_one_item(self,item,dir = None,var = None,ang = None,bins = 5,N = None,unit=None,title = None,rmax = None):
        # var 为所描述数值
        # bins 为单个条目的概率区分数列，np.linspace(...)，传入整数时默认为分级个数，默认为5
        # N为方向的区分度，有几圈,默认为5
        # ang 为标注各圈概率值的标签所在数轴角度
        # unit 为所分布值的单位
        # rmax 为最大概率值，即最外圈所表示的百分比
        unit = " (" + self.get_unit(item) + ")"

        if not (isinstance(dir,pd.Series) and isinstance(var,pd.Series)):
            if not self.if_multiple:
                var = self.data[item]
                dir = self.data.Dir
            else:
                var = self.data[item]['Speed']
                dir = self.data[item]['Dir']
            print("打印全年")
        fig = plt.figure(figsize=(16,9),dpi=200)
        ax = WindroseAxes.from_ax(fig=fig,rmax = rmax)

        ax.bar(dir, var,normed=True, opening=0.8, edgecolor='white', bins=bins,N=N,cmap = cm.rainbow )
        #ax.box(dir, var, normed=True, edgecolor='white', bins=bins,  cmap=cm.rainbow)

        item = item.replace('?','_').replace('/','_')
        ax.set_legend(title=item + unit if unit else item,fancybox = False, facecolor = 'ivory', edgecolor = 'black',
                      ncol = bins//5 if isinstance(bins,int) else len(bins)//5
                      ,fontsize = 15,bbox_to_anchor = (0,0))
        ax.set_radii_angle(angle=45 if not ang else ang)
        ax.tick_params(axis='y', direction='inout', colors='darkblue', pad=1)
        ax.tick_params(axis='x', colors='black', labelsize=15, pad=2)
        ax.grid(color='black', linestyle=':', linewidth=1, alpha=1)
        title = item + " "+title if title else item
        ax.set_title(title,self.font_dict)
        fig_file = title +'.png'
        fig.savefig(fig_file,dpi=200, bbox_inches='tight')
        print(fig_file+'保存成功')
        plt.close()

    def diff_month(self):
        self.data['year_month'] = self.data['t'].apply(lambda x: (x.year, x.month))
        self.group_month = self.data.groupby('year_month')

    def set_units(self,**kwargs):
        if kwargs:
            self.units.update(kwargs)
        else:
            for i in self.items:
                if i not in self.units.keys():
                    u = input('请输入 '+i+' 对应单位\n')
                    self.units.update({i:u})

    def get_unit(self,item):
        if self.is_SI_v:
            return "m/s"
        try:
            return self.units[item]
        except:
            self.set_units()
            self.get_unit(item)

    def plot_each_data_of_one_item(self,item,bins = 10,**kwargs):#bins,ang

        unit = self.get_unit(item)
        if not self.group_month:
            self.diff_month()
        for i, j in self.group_month:

            if self.if_multiple:
                var, dir = j[item]['Speed'], j[item]['Dir']

            else:
                var, dir = j[item], j['Dir']

            self.plot_data_of_one_item(item=item, dir=dir, var=var, bins=bins,
                                       title=str(i[0]) + '年' + str(i[1]) + "月", unit=unit, **kwargs)


class Wave_Ensemble(object):
    def __init__(self, list_of_height, Hz=1):
        diff_h = lambda ll: ll.max() - ll.min()
        df = pd.DataFrame({'h': list_of_height})
        df['diff'] = df['h'] - df['h'].mean()
        for i in range(1, len(list_of_height)):
            if ((df.loc[i - 1, 'diff'] < 0) and (df.loc[i, 'diff'] >= 0)):
                df.loc[i, 'if_zero'] = True
        df['H'] = pd.NaT
        df['T'] = pd.NaT
        list_i = df[df['if_zero'] == True].index
        self.N = len(list_i) - 1
        for i in range(1, len(list_i)):
            df.loc[list_i[i], 'H'] = diff_h(df.loc[list_i[i - 1]:list_i[i], 'h'])
            df.loc[list_i[i], 'T'] = (list_i[i] - list_i[i - 1]) / Hz  # 一秒1数据
        self.df_sorted = df.sort_values(by='H', ascending=False, na_position='last')

    def process(self):
        self.Hmax = self.df_sorted['H'].max()
        self.Tmax = self.df_sorted[self.df_sorted['H'] == self.df_sorted['H'].max()]['T'].values[0]
        self.H10 = self.df_sorted.iloc[0:int(self.N / 10), -2].mean()
        self.T10 = self.df_sorted.iloc[0:int(self.N / 10), -1].mean()
        self.H3 = self.df_sorted.iloc[0:int(self.N / 3), -2].mean()
        self.T3 = self.df_sorted.iloc[0:int(self.N / 3), -1].mean()
        self.Hmean = self.df_sorted['H'].mean()
        self.Tmean = self.df_sorted['T'].mean()

    def output(self):
        dic_columns = ["Hmax", "Tmax", "H10", "T10", "H3", "T3", "Hmean", "Tmean"]
        self.outputdf = pd.DataFrame(
            data=[self.Hmax, self.Tmax, self.H10, self.T10, self.H3, self.T3, self.Hmean, self.Tmean],
            index=dic_columns)
        return self.outputdf


if __name__ == "__main__":
    filename = r"E:\codes\metocean_process\debug_files\Wave_test_file.xlsx"
    filename2 = r"E:\codes\metocean_process\debug_files\wind.xlsx"
    w1 = Wind_Wave(filename2, if_multiple=True, isEWSN=True, is_SI_v=True)
    w1.set_save_dir(r"E:\codes\metocean_process\debug_files\wind_fig")
    # units = {'Hmax':'m', 'Tmax':'s', 'Hs':'m', 'Ts':'s', 'Hm':'m', 'Tm':'s', 'H1/10':'m', 'T1/10':'s', 'Tp':'s'}
    # w1.set_units(**units)
    ####****************风力小于一定风速不纳入统计
    for i in w1.items:
        w1.plot_each_data_of_one_item(i, N=5, ang=270, rmax=35, bins=np.linspace(0, 15, 15))
    print('OK')
"""
ax = WindroseAxes.from_ax()
ax.bar(df.Dir, df.Hs,normed=True, opening=0.8)
ax.set_legend(title = 'Hs'+' (m)',fancybox = False,facecolor='silver',edgecolor = 'black',ncol = 3)"""

