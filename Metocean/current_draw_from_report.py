from collections import namedtuple
import pandas as pd
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
unit = "(cm/s)"
item = '流速'


def east(v, d): return v * np.sin(d / 180 * np.pi)


def north(v, d): return v * np.cos(d / 180 * np.pi)


def aoe(fun, v_es, v_ns):
    vs = []
    for i, j in zip(v_es, v_ns):
        v = fun(i, j)
        vs.append(v)
    return vs


class Read_Report():
    def __init__(
            self,
            filename=r"C:\2019-沈家湾三期（GK-2019-0118水）\实测数据\附录C：走航流速流向报表 - 流矢图.xlsx"):
        def read_excel_report(fileName):
            excel = pd.ExcelFile(fileName)
            for i in excel.sheet_names:
                df = excel.parse(sheet_name=i)
                dfs = split_df(df)
                every_page = sort_df(dfs)
                while True:
                    try:
                        ii = next(every_page)
                    except BaseException:
                        print('*' * 10)
                        print(i)
                        break
                    # draw_rose(ii)
                    self.ALL_DATA.append(ii)

        def select_df(df):
            rows_to_select = df.isna().all(axis=1, bool_only=True)
            boundary = [0]
            for i in range(len(rows_to_select)):
                try:
                    if rows_to_select[i] == False and rows_to_select[i + 1] == True:
                        boundary.append(i + 1)
                    if rows_to_select[i - 1] and rows_to_select[i] == False:
                        boundary.append(i)
                except BaseException:
                    continue
            boundary.append(i)
            # if len(boundary) % 2 != 0:
            #    raise ValueError("请检查输入格式")
            # else:
            return sorted(list(set(boundary)))

        def split_df(df):
            d, dfs = select_df(df), []
            d[-1] = d[-1] + 1
            for i in range(round(len(d) / 2)):
                dfs.append(df.iloc[d[2 * i]:d[2 * i + 1], :])
            return dfs

        def identify(df):
            names = []

            def search_row(
                    row, key):
                return row[row.str.contains(key)].values[0]

            for i in range(5):
                for keys in ['潮汛', '潮型', '测站', '测点']:
                    try:
                        names.append(search_row(df.iloc[i, :], keys))
                    except BaseException:
                        continue
            return names

        def get_name(name):
            for i in name:
                def sep(i):
                    try:
                        return i.split(sep=':')[1]
                    except BaseException:
                        return i.split(sep='：')[1]

                if '测' in i:
                    P = sep(i)
                else:
                    T = sep(i)
            return T, P

        def format_df(df):
            df = df.set_index(pd.to_datetime(df.index, errors='coerce'))
            df = df.reset_index().dropna().set_index('index')
            df = df.dropna(axis=1)

            try:
                df.columns = [
                    'depth',
                    'v_0',
                    'd_0',
                    'v_2',
                    'd_2',
                    'v_4',
                    'd_4',
                    'v_6',
                    'd_6',
                    'v_8',
                    'd_8',
                    'v_10',
                    'd_10',
                    'v_max',
                    'd_max',
                    'v',
                    'd']
            except BaseException:
                df.columns = [
                    'depth',
                    'v_0',
                    'd_0',
                    'v_2',
                    'd_2',
                    'v_4',
                    'd_4',
                    'v_6',
                    'd_6',
                    'v_8',
                    'd_8',
                    'v_10',
                    'd_10',
                    'v',
                    'd']
            df = df[['depth', 'v_0', 'd_0', 'v_2', 'd_2', 'v_4',
                     'd_4', 'v_6', 'd_6', 'v_8', 'd_8', 'v_10', 'd_10', ]]

            df['time'] = df.index
            if (df['v_0'] == 0).all():
                return df.reindex(['time',
                                   'depth',
                                   'v_2',
                                   'd_2',
                                   'v_6',
                                   'd_6',
                                   'v_8',
                                   'd_8'],
                                  axis=1).reset_index()
            else:
                return df.reindex(['time',
                                   'depth',
                                   'v_0',
                                   'd_0',
                                   'v_2',
                                   'd_2',
                                   'v_4',
                                   'd_4',
                                   'v_6',
                                   'd_6',
                                   'v_8',
                                   'd_8',
                                   'v_10',
                                   'd_10',
                                   ],
                                  axis=1).reset_index()

        def sort_df(dfs):
            every_df = namedtuple(
                'each_observation', [
                    'Tide_type', 'Point', 'Data'])
            for i in dfs:
                names = identify(i)
                T, P = get_name(names)
                DF = format_df(i)
                yield every_df(T, P, DF)
                print(P + T + '读取完成')

        def draw_rose(ii):
            df = ii.Data
            v, d = [], []
            for i in [0, 2, 4, 6, 8, 10]:
                try:
                    ddf = df[['v_' + str(i), 'd_' + str(i)]].T
                    for i in range(len(ddf.values[0])):
                        v.append(ddf.values[0][i])
                        d.append(ddf.values[1][i])
                except BaseException:
                    pass
            print("共有" + str(len(v)) + "组流速流向数据")
            fig = plt.figure(figsize=(16, 9), dpi=200)
            ax = WindroseAxes.from_ax(fig=fig, rmax=None)
            ax.bar(
                d,
                v,
                normed=True,
                opening=0.8,
                edgecolor='white',
                bins=np.linspace(
                    0,
                    200,
                    11),
                N=10,
                cmap=cm.rainbow)
            # ax.box(dir, var, normed=True, edgecolor='white', bins=bins,  cmap=cm.rainbow)
            ax.set_legend(
                title=item +
                      unit if unit else item,
                fancybox=True,
                facecolor='ivory',
                edgecolor='black',
                fontsize=15,
                bbox_to_anchor=(
                    0,
                    0),
                decimal_places=0,
                ncol=3,
                prop={
                    'size': 9})
            ax.set_radii_angle(angle=20)
            ax.tick_params(
                axis='y',
                direction='inout',
                colors='darkblue',
                pad=1)
            ax.tick_params(axis='x', colors='black', labelsize=15, pad=2)
            ax.grid(color='black', linestyle=':', linewidth=1, alpha=1)
            title = ii.Point + " " + ii.Tide_type + ' 流速流向分布频级图'
            ax.set_title(title, {'fontsize': 20})
            fig_file = title + '.png'
            fig.savefig(
                fig_file.replace(
                    r'/',
                    '_'),
                dpi=200,
                bbox_inches='tight')
            print(fig_file.replace(r'/', '_') + '保存成功')
            plt.close()
            print(ii.Point + ii.Tide_type + 'OK')

        self.ALL_DATA = []
        read_excel_report(fileName=filename)


def Draw_tide_Profile(
        Single_time,
        Tide_type,
        temp_tide_file=r"C:\2019-沈家湾三期（GK-2019-0118水）\报告撰写\潮流对应时间潮位（程序读取）.xlsx"):
    temp_tide_data = pd.read_excel(temp_tide_file,
                                   sheet_name=Tide_type)
    temp_tide_data['time'] = pd.to_datetime(temp_tide_data['time'])

    a2 = plt.axes([.75, .13, .15, .05], facecolor='grey')
    a2.patch.set_alpha(0.3)
    plt.plot(temp_tide_data['tide'], color='blue')
    i = temp_tide_data[temp_tide_data['time'] == Single_time].index[0]
    plt.axvline(x=i, color='r', linestyle='-')
    plt.xticks([])
    plt.yticks([])


def Draw_Profile_Current(e, n, Tide_type, Single_Time, Profile_Name):
    paramater = 0.2 / max(e.max().max(), n.max().max())
    s = str(Single_Time)[:13].replace('T', " ").replace('2019-', "") + ":00"
    title = "断面：" + Profile_Name + " 时间:" + s
    fig, ax = plt.subplots(1, 1, figsize=(
        len(Profiles[Profile_Name]) / 2 + 1, 8))
    for i in range(1, 8):  # 层数，对应高度
        for j in range(1, len(Profiles[Profile_Name]) + 1):
            if not e.iloc[7 - i][j - 1] == n.iloc[7 - i][j - 1] == 0:
                ax.arrow(j,
                         i,
                         e.iloc[7 - i][j - 1] * paramater,
                         n.iloc[7 - i][j - 1] * paramater,
                         head_width=0.1,
                         head_length=0.1,
                         fc='blue',
                         ec='black')
    ax.set_xlim(0, len(Profiles[Profile_Name]) + 1)
    ax.set_ylim(0, 8)
    ax.set_xlabel('点号', fontsize=15)
    ax.yaxis.grid(True)
    plt.xticks(range(1,
                     len(Profiles[Profile_Name]) + 1),
               Profiles[Profile_Name],
               fontsize=10)
    plt.yticks(range(8), ['', '底层', '0.8H', '0.6H',
                          '0.4H', '0.2H', '表层', '垂线平均'], fontsize=10)
    plt.title("断面：" + Profile_Name + " " + s, fontsize=25)

    Draw_tide_Profile(Single_Time, Tide_type)

    plt.savefig(Profile_Name + s.replace(":00", "") + ".png", dpi=300)
    print(title + "COMPLETED!")
    plt.close()


a = Read_Report()

df_profile = pd.read_excel(r"C:\2019-沈家湾三期（GK-2019-0118水）\报告撰写\断面测点对照表.xlsx")
Profiles = {}
g = df_profile.groupby('Profile')

for i, j in g.groups.items():
    points = []
    for jj in j:
        points.append(df_profile.loc[jj, 'Point'])
    Profiles.update({i: points})
    print(i + "断面对应测点为:" + str(points))

    Tide_types = ['大潮', '中潮', '小潮']
for Profile_Name in Profiles.keys():  # 每个断面
    Time_Split_Datas_N, Time_Split_Datas_E = {}, {}
    for Tide_type in Tide_types:  # 每种潮型
        Tide_Profile_Data = []
        for i in a.ALL_DATA:
            if i.Point in Profiles[Profile_Name] and i.Tide_type == Tide_type:
                i2 = i
                for c in [0, 2, 4, 6, 8, 10]:
                    THREE_CELLS = False
                    try:
                        i2.Data['e_' + str(c)] = aoe(east,
                                                     i.Data['v_' + str(c)],
                                                     i.Data['d_' + str(c)])
                        i2.Data['n_' + str(c)] = aoe(north,
                                                     i.Data['v_' + str(c)],
                                                     i.Data['d_' + str(c)])
                    except BaseException:
                        i2.Data['e_' + str(c)] = 0
                        i2.Data['n_' + str(c)] = 0
                        THREE_CELLS = True

                if THREE_CELLS:
                    i2.Data['e'] = (
                                           i2.Data['e_2'] + i2.Data['e_6'] + i2.Data['e_8']) / 3
                    i2.Data['n'] = (
                                           i2.Data['n_2'] + i2.Data['n_6'] + i2.Data['n_8']) / 3
                else:
                    i2.Data['e'] = (i2.Data['e_0'] + i2.Data['e_2'] * 2 + i2.Data['e_4']
                                    * 2 + i2.Data['e_6'] * 2 + i2.Data['e_8'] * 2 + i2.Data['e_10']) / 10
                    i2.Data['n'] = (i2.Data['n_0'] + i2.Data['n_2'] * 2 + i2.Data['n_4']
                                    * 2 + i2.Data['n_6'] * 2 + i2.Data['n_8'] * 2 + i2.Data['n_10']) / 10

                Tide_Profile_Data.append(i2)
        times = i2.Data.time.values

        for Single_Time in times:
            NS_single, ES_single = {}, {}
            for p in Tide_Profile_Data:
                es, ns = [], []
                for c in ['', '_0', '_2', '_4', '_6', '_8', '_10']:
                    es.append(p.Data.loc[p.Data.time ==
                                         Single_Time, 'e' + str(c)].values[0])
                    ns.append(p.Data.loc[p.Data.time ==
                                         Single_Time, 'n' + str(c)].values[0])

                NS_single.update({p.Point: ns})
                ES_single.update({p.Point: es})

            n = pd.DataFrame(NS_single, columns=Profiles[Profile_Name])
            e = pd.DataFrame(ES_single, columns=Profiles[Profile_Name])
            Draw_Profile_Current(e, n, Tide_type, Single_Time, Profile_Name)

            #Time_Split_Datas_N.update({s: n})
            #Time_Split_Datas_E.update({s: e})
