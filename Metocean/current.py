from functools import reduce
from dxfwrite import DXFEngine as dxf
import time
import matplotlib.pyplot as plt
import operator
import pandas as pd
import numpy as np
import datetime
import threading
from collections import namedtuple
import utide
from matplotlib.dates import date2num
from windrose import WindroseAxes
import matplotlib.cm as cm
import pickle, os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.float_format = '{:,.2f}'.format
constit_tide_level = [
    'O1',
    'Q1',
    'P1',
    'K1',
    'N2',
    'M2',
    'K2',
    'S2',
    'M4',
    'MS4',
    'M6']


def east(v, d):
    try:
        return v * np.sin(d / 180 * np.pi)
    except BaseException:
        d = d.astype(np.float64)
        return v * np.sin(d / 180 * np.pi)


def north(v, d):
    try:
        return v * np.cos(d / 180 * np.pi)
    except BaseException:
        d = d.astype(np.float64)
        return v * np.cos(d / 180 * np.pi)


def velocity(v_e, v_n): return np.sqrt(v_e ** 2 + v_n ** 2)


def dir_in_360b(d): return (d - 360 if d >= 360 else d) if d > 0 else 360 + d


def dir_in_360(d): return dir_in_360b(d) if (
    dir_in_360b(d) >= 0 and (
        dir_in_360b(d) < 360)) else dir_in_360b(
    dir_in_360b(d))


def direction(
    v_e,
    v_n): return dir_in_360(
        180 /
        np.pi *
        np.arctan2(
            v_e,
            v_n)) if (
                v_e > 0) else dir_in_360(
                    180 /
                    np.pi *
                    np.arctan2(
                        v_e,
                        v_n) +
    360)


def small_diff_dir(
    d1,
    d2): return (
        d1 -
        d2 if d1 -
        d2 < 180 else d2 +
        360 -
        d1) if d1 > d2 else (
            d2 -
            d1 if d2 -
            d1 < 180 else d1 +
            360 -
    d2)


def is_d1_in_d2_and_d3(
    d1,
    d2,
    d3): return True if (
        small_diff_dir(
            d1,
            d3) +
    small_diff_dir(
            d1,
            d2) == small_diff_dir(
            d2,
            d3)) else False


def str_from_datatime64(t): return str(t)[:-31][-19:].replace('T', ' ')


def str_from_float(t, n=2): return str(round(t, n))


def str_from_df(item): return item.values[0]


def num_from_df(item): return str(round(item.values[0], 2))


def aoe(fun, v_es, v_ns):
    vs = []
    for i, j in zip(v_es, v_ns):
        v = fun(i, j)
        vs.append(v)
    return vs



def is_time_of1(d):
    def raise_or_not(angle1, ang=0):  # 涨/落潮主方向
        if angle1 > 180:
            raise ValueError
            return "Wrong with the angle"
        if angle1 < 90:
            angle1 = angle1 + 90
        else:
            angle1 = angle1 - 90
        angle2 = angle1 + 180
        if min(abs(d - angle1),
               abs(d - angle2),
               360 - abs(d - angle2),
               360 - abs(d - angle1)) < ang:
            return 0  # 转流
        elif angle1 < d < angle2:
            return 1  # 涨/落潮
        else:
            return -1  # 落/涨潮

    return raise_or_not


def is_time_of(d):
    def raise_or_not(angle, ang=0):  # angle 为落潮方向
        if ang > 90:
            raise ValueError("Wrong with the angle")
        d1 = dir_in_360(angle + 90)  # 两个转流流向
        d2 = dir_in_360(angle - 90)
        if min(abs(d - d1),
               abs(d - d2),
               360 - abs(d - d1),
               360 - abs(d - d2)) < ang:
            return 0  # 转流
        elif small_diff_dir(d, angle) < 90:  # 同向
            return 1  # 红色
        else:
            return -1  # 绿色

    return raise_or_not


def get_duration(time_v_d):  # 统计涨落潮时间
    time_v_d = time_v_d[time_v_d.timeof != 0]
    time_v_d.index = range(0, len(time_v_d))
    raise_time = 0  # 此时无法区分涨落，仅作标识用途
    ebb_time = 0
    raise_duration = datetime.timedelta(0)
    ebb_duration = datetime.timedelta(0)
    raise_last = False
    ebb_last = False
    raise_times = []
    ebb_times = []
    for i in range(1, len(time_v_d)):
        if (time_v_d.loc[i, 'timeof'] == 1) and (
                time_v_d.loc[i - 1, 'timeof'] == 1) and raise_last:  # 涨潮last
            raise_duration += (time_v_d.loc[i, 'time'] -
                               time_v_d.loc[i - 1, 'time'])
            continue
        if (time_v_d.loc[i, 'timeof'] == -
            1) and (time_v_d.loc[i -
                                 1, 'timeof'] == -
                    1) and ebb_last:  # 落潮last
            ebb_duration += (time_v_d.loc[i, 'time'] -
                             time_v_d.loc[i - 1, 'time'])
            continue
        if (time_v_d.loc[i, 'timeof'] == -
            1) and (time_v_d.loc[i -
                                 1, 'timeof'] == 1):  # 涨潮end,直接变落潮
            if raise_last:
                raise_time += 0.5
                raise_duration += (time_v_d.loc[i,
                                                'time'] - time_v_d.loc[i - 1,
                                                                       'time']) / 2
                raise_last = False
                raise_times.append(raise_duration)
                raise_duration = datetime.timedelta(0)
            ebb_time += 0.5
            ebb_last = True
            ebb_duration += (time_v_d.loc[i, 'time'] -
                             time_v_d.loc[i - 1, 'time']) / 2
            continue
        if (time_v_d.loc[i, 'timeof'] == 1) and (
                time_v_d.loc[i - 1, 'timeof'] == -1):  # 落潮end,直接变涨潮
            if ebb_last:
                ebb_time += 0.5
                ebb_duration += (time_v_d.loc[i,
                                              'time'] - time_v_d.loc[i - 1,
                                                                     'time']) / 2
                ebb_last = False
                ebb_times.append(ebb_duration)
                ebb_duration = datetime.timedelta(0)
            raise_time += 0.5
            raise_last = True
            raise_duration += (time_v_d.loc[i,
                                            'time'] - time_v_d.loc[i - 1,
                                                                   'time']) / 2
            continue
    # print('raise_time=' + str(raise_time))  # 次数
    # 第一个时间为首次出现的方向对应次数  # 次数
    if len(raise_times) == 0:
        last_time1 = None
    else:
        last_time1 = reduce(operator.add, raise_times) / int(raise_time)
    if len(ebb_times) == 0:
        last_time2 = None
    else:
        last_time2 = reduce(operator.add, ebb_times) / int(ebb_time)
    if last_time1 and last_time2:
        if last_time1 + last_time2 < pd.Timedelta('12h30m'):
            if last_time1 - last_time2 > pd.Timedelta('1m'):
                last_time2 = pd.Timedelta('12h30m') - last_time1
            else:
                last_time1 = pd.Timedelta('12h30m') - last_time2

    return {
        'durations_1': raise_times,
        'durations_2': ebb_times,
        'times1': raise_time,
        'times2': ebb_time,
        'last_time1': last_time1,
        'last_time2': last_time2}


class Current_pre_process(object):
    def __init__(self, filename, sheet_name=None, is_VD=True):
        if 'csv' in filename:
            try:
                self.df = pd.read_csv(filename)
            except BaseException:
                self.df = pd.read_csv(filename, encoding='GB2312')
        if ('xlsx' in filename) or ('xls' in filename):
            f = pd.ExcelFile(filename)
            if not sheet_name:
                sheet_name = f.sheet_names[0]
            self.name = sheet_name
            self.df = f.parse(sheet_name=sheet_name)

        self.df = self.df.replace(to_replace=3276.7, value=np.nan)
        self.df = self.df.drop([i for i in self.df if 'Unnamed' in i], axis=1)

        # self.df = self.df.reset(range(len(self.df)))

        if 't' not in self.df.columns:
            #self.df['t'] = self.df['time']
            if 'time' not in self.df.columns:
                raise ValueError('文件中没有时间数据')
            self.df = self.df.rename(columns={"time": "t"})
        try:
            self.df['t'] = pd.to_datetime(self.df['t'])
            self.df = self.df.dropna(subset=['t'])
        except BaseException:
            pass

        if is_VD:
            d_i = [d for d in self.df.columns if 'd' in d]
            v_i = [v for v in self.df.columns if 'v' in v]
            if not (d_i and v_i):
                raise ValueError('文件中没有流速流向数据')

            self.d = self.df[d_i].apply(self.convert_num, axis=1)
            self.v = self.df[v_i].apply(self.convert_num, axis=1)

            self.l = self.get_less_index(self.d, self.v)
            if -1 in self.l:
                raise ValueError(str(self.t[self.l.index(-1)]) + '首层为空值')
            self.e, self.n = self.convert_to_e_n()

        else:
            e_i = [e for e in self.df.columns if 'e' in e]
            n_i = [n for n in self.df.columns if 'n' in n]
            if not (e_i and n_i):
                raise ValueError('文件中缺少分向流速数据')
            self.n = self.df[n_i].apply(self.convert_num, axis=1)
            self.e = self.df[e_i].apply(self.convert_num, axis=1)

            self.l = self.get_less_index(self.n, self.e)

            if -1 in self.l:
                raise ValueError(str(self.t[self.l.index(-1)]) + '首层为空值')

            self.v, self.d = self.convert_to_v_d()

    def convert_num(self, series):
        try:
            return pd.to_numeric(series, errors=pd.NaT)
        except BaseException:
            return series

    def convert_to_e_n(self):
        e = pd.DataFrame()
        n = pd.DataFrame()
        for i in range(1, len(self.d.columns) + 1):
            try:
                e['e' + str(i)] = east(self.v['v' + str(i)],
                                       self.d['d' + str(i)])
                n['n' + str(i)] = north(self.v['v' + str(i)],
                                        self.d['d' + str(i)])
            except BaseException:
                e['e' +
                  str(i)] = east(self.v['v' +
                                        str(i)].convert_objects(convert_numeric=True), self.d['d' +
                                                                                              str(i)].convert_objects(convert_numeric=True))
                n['n' +
                  str(i)] = north(self.v['v' +
                                         str(i)].convert_objects(convert_numeric=True), self.d['d' +
                                                                                               str(i)].convert_objects(convert_numeric=True))
        return e, n

    def convert_to_v_d(self):
        v = pd.DataFrame()
        d = pd.DataFrame()
        for i in range(1, len(self.e.columns) + 1):
            try:
                v['v' + str(i)] = aoe(velocity,
                                      self.e['e' + str(i)],
                                      self.n['n' + str(i)])
                d['d' + str(i)] = aoe(direction,
                                      self.e['e' + str(i)],
                                      self.n['n' + str(i)])
            except BaseException:
                # self.e.loc['e' + str(i)]= self.e['e' + str(i)].convert_objects(convert_numeric=True)
                # self.n.loc['n' + str(i)]=  self.n['n' + str(i)].convert_objects(convert_numeric=True)
                self.e.loc['e' +
                           str(i)] = pd.to_numeric(self.e['e' +
                                                          str(i)], errors=pd.NaT)
                self.e.loc['n' +
                           str(i)] = pd.to_numeric(self.e['n' +
                                                          str(i)], errors=pd.NaT)
                v['v' + str(i)] = aoe(velocity,
                                      self.e['e' + str(i)],
                                      self.n['n' + str(i)])
                d['n' + str(i)] = aoe(direction,
                                      self.e['e' + str(i)],
                                      self.n['n' + str(i)])
                # v['e' + str(i)] = aoe(velocity,self.e['e' + str(i)].convert_objects(convert_numeric=True),self.n['n' + str(i)].convert_objects(convert_numeric=True))
                # d['e' + str(i)] = direction(self.e['e' + str(i)].convert_objects(convert_numeric=True), self.n['n' + str(i)].convert_objects(convert_numeric=True))
                # d['e' + str(i)] = aoe(direction, self.e['e' + str(i)].convert_objects(convert_numeric=True),self.n['n' + str(i)].convert_objects(convert_numeric=True))
        return v, d

    def convert_to_EN_file(self, outfile):
        if 'xlsx' not in outfile:
            raise ValueError("输出文件为excel文件")
        self.merge_save_csv(self.e, self.n, outfile)

    def convert_to_VD_file(self, outfile):
        if 'xlsx' not in outfile:
            raise ValueError("输出文件为excel文件")
        self.merge_save_csv(self.v, self.d, outfile)

    def merge_save_csv(self, v1, v2, outfile):
        x = pd.merge(
            v1,
            v2,
            how='outer',
            on=None,
            left_index=True,
            right_index=True)
        x = pd.merge(
            pd.DataFrame(
                self.df.t),
            x,
            how='outer',
            left_index=True,
            right_index=True)
        x.to_excel(outfile, sheet_name=self.name)

    def get_less_index(slef, df_1, df_2):
        def find_NaN_index(df):
            l = []
            df = df.reindex(
                sorted(df.columns, key=lambda x: int(x[1:])), axis=1)
            for i in range(len(df)):
                tt = pd.to_numeric(df.loc[i, :])
                s = list(tt.apply(np.isnan))
                if True in s:
                    l.append(s.index(True) - 1)
                else:
                    l.append(len(s) - 1)
            return l  # l的层数从0开始算

        l1 = find_NaN_index(df_1)
        l2 = find_NaN_index(df_2)
        if len(l1) != len(l2):
            raise ValueError
        l = []
        for i, j in zip(l1, l2):
            if i < j:
                l.append(i)
            else:
                l.append(j)
        return l

    def fen_ceng(
            self,
            zongcengshu,
            t,
            bin,
            first_bin,
            top_ratio,
            button_ratio,
            reverse):
        # zongcengshu =  zongcengshu - 1 #20200219 六横用倒数第二层底层数据
        depth = first_bin + bin * (zongcengshu + 1)
        if zongcengshu == 1:  # 只有两层的情况
            h_6_e = self.e['e1'][t]
            h_6_n = self.n['n1'][t]
            button_e = self.e['e' + str(zongcengshu + 1)][t] * button_ratio
            button_n = self.n['n' + str(zongcengshu + 1)][t] * button_ratio
            factor = bin * ((depth ** (7 / 6)) - (depth - first_bin - bin / 2) ** (7 / 6)) / first_bin * (
                (depth - first_bin - bin / 2) ** (7 / 6) - bin / 2)  # 采用幂函数推算，具体见泛际帮助手册中的公式
            top_e1 = (h_6_e + button_e / button_ratio) * factor  # 用于计算流向
            top_n1 = (h_6_n + button_n / button_ratio) * factor
            top_dir = direction(top_e1, top_n1)
            top_velocity = velocity(h_6_e, h_6_n) * top_ratio  # 流速用中间层推算
            top_e = east(top_velocity, top_dir)
            top_n = north(top_velocity, top_dir)

            h_2_e, h_2_n, h_4_e, h_4_n, h_8_e, h_8_n = [0] * 6
            ave_e = (top_e + h_6_e + button_e) / 3
            ave_n = (top_n + h_6_n + button_n) / 3
            return top_e, top_n, h_2_e, h_2_n, h_4_e, h_4_n, h_6_e, h_6_n, h_8_e, h_8_n, button_e, button_n, ave_e, ave_n, depth

        if reverse:
            """if velocity(self.e['e' + str(zongcengshu + 1)][t], self.n['n' + str(zongcengshu + 1)][t]) <=  velocity(
                    self.e['e' + str(zongcengshu)][t], self.n['n' + str(zongcengshu)][t]) * 3 and velocity(self.e['e' + str(zongcengshu + 1)][t], self.n['n' + str(zongcengshu + 1)][t])* velocity(
                    self.e['e' + str(zongcengshu)][t], self.n['n' + str(zongcengshu)][t]) != 0:  # 比较前两层流速大小
                if_first_bin = 1
            else:
                if_first_bin = 0"""
            if zongcengshu > 10:  # and velocity(self.e['e' + str(zongcengshu + 1)][t], self.n['n' + str(zongcengshu + 1)][t]) <=  velocity(
                # self.e['e' + str(zongcengshu)][t], self.n['n' +
                # str(zongcengshu)][t]) *1.5  :
                if_first_bin = 0  # 吉布提表层数据均偏大，直接用第二层数据,应为0
            else:
                if_first_bin = 1
            # if_first_bin = 0  # 6-11 为五月C3分层尝试
            # 第一层流速太大，用第二层
            top_e = self.e['e' +
                           str(zongcengshu + if_first_bin)][t] * top_ratio
            top_n = self.n['n' +
                           str(zongcengshu + if_first_bin)][t] * top_ratio
            button_e = self.e['e1'][t] * button_ratio
            button_n = self.n['n1'][t] * button_ratio

            scatter = np.linspace(
                depth - bin / 2,
                first_bin + bin / 2,
                zongcengshu + 1)

        else:

            top_e = self.e['e1'][t] * top_ratio
            top_n = self.n['n1'][t] * top_ratio

            button_e = self.e['e' + str(zongcengshu + 1)][t] * button_ratio
            button_n = self.n['n' + str(zongcengshu + 1)][t] * button_ratio

            # 2019/3/5 走航底层用第三层
            # if zongcengshu < 4:
            #    zongcengshu = zongcengshu + 4
            #button_e = self.e['e' + str(zongcengshu - 3)][t] * button_ratio
            #button_n = self.n['n' + str(zongcengshu - 3)][t] * button_ratio
            scatter = np.linspace(
                first_bin + bin / 2,
                depth - bin / 2,
                zongcengshu + 1)

        def cal_h(ref): return depth * ref if depth * \
            ref > first_bin + 0.5 * bin else 0

        h_8 = cal_h(0.2)  # 距离水面0.8即为距离水底0.2，下同
        h_6 = cal_h(0.4)
        h_4 = cal_h(0.6)
        h_2 = cal_h(0.8)

        if not reverse:
            h_2, h_4, h_6, h_8 = h_8, h_6, h_4, h_2

        def quan_zhong(x):
            def quan_zhong_dep(bin, x, up_depth, down_depth): return [abs(
                x - up_depth) / bin, abs(down_depth - x) / bin]  # 返回的按顺序分别是到下层的权重以及到上层的权重
            if x != 0 and zongcengshu != 0:
                scatter_add_x = sorted(np.append(scatter, x))

                if reverse:
                    scatter_add_x_r = list(
                        reversed(
                            sorted(
                                np.append(
                                    scatter,
                                    x))))  # 用于计算高度
                    up_x_index = scatter_add_x_r.index(x) - 1
                    down_x_index = scatter_add_x_r.index(x)
                else:
                    up_x_index = scatter_add_x.index(x) - 1
                    down_x_index = scatter_add_x.index(x)
                up_depth = scatter[up_x_index]
                down_depth = scatter[down_x_index]

                x_down, x_up = quan_zhong_dep(bin, x, up_depth, down_depth)
                if reverse:  # 计算层数，正序不需要,并且将计算权重翻转
                    up_x_index = scatter_add_x.index(x) - 1
                    down_x_index = scatter_add_x.index(x)
                    x_down, x_up = x_up, x_down
                return up_x_index + 1, down_x_index + 1, x_up, x_down
            else:
                return 1, 1, 0.5, 0.5

        def jia_quan(df_e, df_n, t, i, j, i_r, j_r):
            return df_e['e' + str(i)][t] * i_r + df_e['e' + str(j)][t] * \
                j_r, df_n['n' + str(i)][t] * i_r + df_n['n' + str(j)][t] * j_r

        i, j, i_r, j_r = quan_zhong(h_2)
        h_2_e, h_2_n = jia_quan(self.e, self.n, t, i, j, i_r, j_r)

        i, j, i_r, j_r = quan_zhong(h_4)
        h_4_e, h_4_n = jia_quan(self.e, self.n, t, i, j, i_r, j_r)

        i, j, i_r, j_r = quan_zhong(h_6)
        h_6_e, h_6_n = jia_quan(self.e, self.n, t, i, j, i_r, j_r)

        i, j, i_r, j_r = quan_zhong(h_8)
        h_8_e, h_8_n = jia_quan(self.e, self.n, t, i, j, i_r, j_r)

        ave_e = (top_e + 2 * h_2_e + 2 * h_4_e + 2 *
                 h_6_e + 2 * h_8_e + button_e) / 10
        ave_n = (top_n + 2 * h_2_n + 2 * h_4_n + 2 *
                 h_6_n + 2 * h_8_n + button_n) / 10
        return top_e, top_n, h_2_e, h_2_n, h_4_e, h_4_n, h_6_e, h_6_n, h_8_e, h_8_n, button_e, button_n, ave_e, ave_n, depth

    def all_fenceng(
            self,
            bin=1,
            first_bin=0.7,
            top_ratio=1.05,
            button_ratio=0.95,
            reverse=False):
        columns = ['top', 'h_2', 'h_4', 'h_6', 'h_8', 'button', 'ave']

        self.ee = pd.DataFrame(columns=columns)
        self.nn = pd.DataFrame(columns=columns)
        self.vv = pd.DataFrame(columns=columns)
        self.dd = pd.DataFrame(columns=columns)
        all_depth = []
        for i in self.df.index:
            top_e, top_n, h_2_e, h_2_n, h_4_e, h_4_n, h_6_e, h_6_n, h_8_e, h_8_n, button_e, button_n, ave_e, ave_n, depth = self.fen_ceng(
                self.l[i], i, bin, first_bin, top_ratio, button_ratio, reverse)
            e = pd.DataFrame(
                [[top_e, h_2_e, h_4_e, h_6_e, h_8_n, button_e, ave_e]], columns=columns)
            n = pd.DataFrame(
                [[top_n, h_2_n, h_4_n, h_6_n, h_8_n, button_n, ave_n]], columns=columns)
            v = pd.DataFrame(
                [
                    [
                        velocity(
                            top_e, top_n), velocity(
                            h_2_e, h_2_n), velocity(
                            h_4_e, h_4_n), velocity(
                            h_6_e, h_6_n), velocity(
                                h_8_e, h_8_n), velocity(
                                    button_e, button_n), velocity(
                                        ave_e, ave_n)]], columns=columns)
            d = pd.DataFrame(
                [
                    [
                        direction(
                            top_e, top_n), direction(
                            h_2_e, h_2_n), direction(
                            h_4_e, h_4_n), direction(
                            h_6_e, h_6_n), direction(
                                h_8_e, h_8_n), direction(
                                    button_e, button_n), direction(
                                        ave_e, ave_n)]], columns=columns)
            self.ee = self.ee.append(e, ignore_index=True)
            self.nn = self.nn.append(n, ignore_index=True)
            self.vv = self.vv.append(v, ignore_index=True)
            self.dd = self.dd.append(d, ignore_index=True)
            all_depth.append(depth)
        self.depth = pd.Series(all_depth)
        columns = ['time', 'top', 'h_2', 'h_4', 'h_6', 'h_8', 'button', 'ave']
        self.ee['time'] = self.df['t']
        self.ee = self.ee[columns]
        self.nn['time'] = self.df['t']
        self.nn = self.nn[columns]
        self.vv['time'] = self.df['t']
        self.vv = self.vv[columns]
        self.dd['time'] = self.df['t']
        self.dd = self.dd[columns]

    def save_result(
            self,
            outfile,
            V_D=True,
            mag_declination=0,
            cenimeter=True):
        # 磁偏角向东为正，向西为负
        if V_D:
            first = self.vv
            v_columns = {
                'top': 'v_0',
                'h_2': 'v_2',
                'h_4': 'v_4',
                'h_6': 'v_6',
                'h_8': 'v_8',
                'button': 'v_10',
                'ave': 'v'}
            second = self.dd
            if mag_declination:
                for i in second.columns.drop('time'):
                    second[i] = second[i].apply(
                        lambda x: dir_in_360(x + mag_declination))
            d_columns = {
                'top': 'd_0',
                'h_2': 'd_2',
                'h_4': 'd_4',
                'h_6': 'd_6',
                'h_8': 'd_8',
                'button': 'd_10',
                'ave': 'd'}
            if cenimeter and first.drop('time', axis=1).max().max() < 15:
                for i in first.columns.drop('time'):
                    first[i] = first[i] * 100
            first = first.rename(columns=v_columns)
            second = second.rename(columns=d_columns)
            # emerge = pd.concat([first, second],axis=1)
            columns = [
                'time',
                'Depth',
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
        else:
            first = self.ee
            e_columns = {
                'top': 'e_0',
                'h_2': 'e_2',
                'h_4': 'e_4',
                'h_6': 'e_6',
                'h_8': 'e_8',
                'button': 'e_10',
                'ave': 'e'}
            second = self.nn
            n_columns = {
                'top': 'n_0',
                'h_2': 'n_2',
                'h_4': 'n_4',
                'h_6': 'n_6',
                'h_8': 'n_8',
                'button': 'n_10',
                'ave': 'n'}
            first = first.rename(columns=e_columns)
            second = second.rename(columns=n_columns)
            # emerge = pd.concat([first,second],axis=1)

            columns = [
                'time',
                'Depth',
                'e_0',
                'n_0',
                'e_2',
                'n_2',
                'e_4',
                'n_4',
                'e_6',
                'n_6',
                'e_8',
                'n_8',
                'e_10',
                'n_10',
                'e',
                'n']
        # emerge = emerge.ix[:, columns]
        # emerge = emerge.T.drop_duplicates().T
        emerge = pd.merge(first, second, how='outer')
        emerge['Depth'] = self.depth
        emerge = emerge.reindex(columns, axis=1)
        self.to_excel_style = emerge
        if "csv" in outfile:
            emerge.to_csv(outfile, sep=',', index=False, encoding='utf-8')
        else:
            emerge.to_excel(outfile, index=False, sheet_name=self.name)


r"""

sheetname = 'ne'
c = current(filename=r"E:\★★★★★项目★★★★★\★★★★★双子山油品储运贸易基地陆域形成工程★★★★★\实测数据\大潮\C1刘鹏飞\ne.xlsx",sheet_name=sheetname,is_VD=False)
c.all_fenceng()
c.save_result(r"E:\★★★★★项目★★★★★\★★★★★双子山油品储运贸易基地陆域形成工程★★★★★\实测数据\大潮\C1刘鹏飞\处理结果.csv")
"""
"""
c.all_fenceng()
c.save_result(b)
"""


class One_Current_Point(object):
    def __init__(self, point, angle, ang=0, zhang_or_luo=False, cengshu=6):
        self.point = point
        self.angle = angle
        self.ang = ang
        self.zhang_or_luo = zhang_or_luo
        self.cengshu = cengshu

    def location(self, longitude=None, latitude=None, x=None, y=None):
        if longitude and latitude:
            self.longitude = longitude
            self.latitude = latitude
        if x and y:
            self.x = x
            self.y = y

    def set_angle(self, angle):
        if angle < 0:
            self.zhang_or_luo = True

        self.angle = abs(angle)


class Single_Tide_Point(One_Current_Point):
    def __init__(self, filename, tide_type, point, angle, ang=0, zhang_or_luo=False, cengshu=6, format_report=False,
                 data=None):  # False时，默认给出为落潮流向
        One_Current_Point.__init__(
            self, point, angle, ang, zhang_or_luo, cengshu)
        self.tide_type = tide_type
        self.filename = filename
        self.format_report = format_report
        self.data = data

    def get_data(self):
        if 'csv' in self.filename:
            try:
                data = pd.read_csv(self.filename)
            except BaseException:
                data = pd.read_csv(self.filename, encoding='GB2312')
        else:
            data = pd.read_excel(self.filename)
        return data

    def preprocess(self):
        if not self.format_report:
            data = self.get_data()
        else:
            data = self.data

        data = data.loc[data['time'].dropna().index, :]

        self.cengs = []
        if ((self.cengshu == 6) or (self.cengshu == 3) or (self.cengshu == 2)):
            try:
                self.cengs.append(data[['time', 'v', 'd']])
            except BaseException:
                try:
                    self.cengs.append(data[['t', 'v', 'd']])
                    self.cengs[-1]['time'] = self.cengs[-1]['t']
                except BaseException:
                    pass
            for i in range(0, 12, 2):

                try:
                    self.cengs.append(
                        data[['time', 'v_' + str(i), 'd_' + str(i)]])
                except BaseException:
                    self.cengs.append(
                        data[['time', 'v' + str(i), 'd' + str(i)]])
                    self.cengs[-1]['v_' +
                                   str(i)] = self.cengs[-1]['v' + str(i)]
                    self.cengs[-1]['d_' +
                                   str(i)] = self.cengs[-1]['d' + str(i)]
                # 0为垂线，后面的依次是从上往下的层数
        self.ceng_processed = []
        for i in self.cengs:
            self.ceng_processed.append(
                time_v_d(
                    i,
                    self.angle,
                    ang=self.ang,
                    zhang_or_luo=self.zhang_or_luo))

        try:
            self.cal_time_of_ave()
        except BaseException:
            print('涨落潮时间计算有误')
            pass

    def cal_time_of_ave(self):
        self.time = self.cal_time(self.cengs[0])
        self.raise_time = self.time['last_time1']
        self.ebb_time = self.time['last_time2']
        self.first_ebb = False
        if self.zhang_or_luo:  # 给出为涨潮流向时
            if not self.cengs[0].loc[0]['timeof'] == 1:  # 初次为落潮
                self.first_ebb = True
                # else:
                #    First_ebb = False #初次为涨潮
        else:
            if self.cengs[0].loc[0]['timeof'] == 1:
                self.first_ebb = True
                # else:
                #    First_ebb  = False#初次为涨潮
        if not self.first_ebb:
            self.raise_time, self.ebb_time = self.ebb_time, self.raise_time

    def cal_time(self, tvd):
        tvd.loc[tvd.index, 'timeof'] = tvd['d'].apply(
            lambda x: is_time_of(x)(self.angle, self.ang))

        tvd = add_convert_row_to_tvd_and_timeof(tvd, self.angle)

        def remove_small_duration(group, time_v_d):
            one_time_direction_index = []
            group2 = group.copy()
            for i in range(len(group) - 1):
                if (group.index[i] -
                    1 in group.index) and (group.index[i] +
                                           1 in group.index):
                    group2 = group2.drop(group.index[i])
                if (group.index[i] -
                    1 not in group.index) and (group.index[i] +
                                               1 not in group.index):
                    one_time_direction_index.append(group.index[i])
                    group2 = group2.drop(group.index[i])
            n = int(len(group2) / 2)
            if n * 2 != len(group2):
                print('*' * 10 + '筛选短周期方向过程有问题' + '*' * 10)
            for i in range(n):
                if group2.loc[group2.index[i * 2 + 1],
                              't'] - group2.loc[group2.index[i * 2],
                                                't'] < pd.Timedelta('2h'):
                    time_v_d.loc[group2.index[i * 2]:group2.index[i * 2 + 1], 'timeof'] = 0
            return time_v_d, one_time_direction_index

        def both_dir_remove_small_duration(tvd):
            g = tvd.groupby('timeof')
            tvd, one_time_direction1 = remove_small_duration(
                tvd.loc[g.groups[1].copy(), :], tvd)
            tvd, one_time_direction2 = remove_small_duration(
                tvd.loc[g.groups[-1].copy(), :], tvd)
            tvd = tvd.drop(one_time_direction1, axis=0)
            tvd = tvd.drop(one_time_direction2, axis=0)
            return tvd

        try:
            tvd = both_dir_remove_small_duration(
                tvd)  # 返回的第二个值为1则流向全部一致，否则是其他情况
        except BaseException:
            print('Error: 请检查是否流向全部一致')
            return {
                'durations_1': None,
                'durations_2': None,
                'times1': None,
                'times2': None,
                'last_time1': None,
                'last_time2': None}

        while tvd.loc[0, 'timeof'] == 0:  # 去掉开始和结束时候就是转流的情况
            tvd = tvd.drop(0, axis=0)
            tvd = tvd.reset_index(drop=True)
        while tvd.loc[len(tvd) - 1, 'timeof'] == 0:
            tvd = tvd.drop(len(tvd) - 1, axis=0)
            tvd = tvd.reset_index(drop=True)

        return get_duration(tvd)

        # self.ave_of_vertical = self.cengs[0]
        # self.ave_of_vertical['timeof'] = self.ave_of_vertical['d'].apply(lambda x : is_time_of(x)(angle,ang))
        # self.ave_of_vertical = add_convert_row_to_tvd_and_timeof(***,ang)

        # self.time = get_duration(self.ave_of_vertical)

    def out_put(self, ):
        ceng_index = ['垂线平均', '表层', '0.2H', '0.4H', '0.6H', '0.8H', '底层', ]
        columns = ['层数', '平均流速', '平均流向', '最大流速', '最大流速对应方向', '出现时间']
        zhang = pd.DataFrame(columns=columns)
        luo = pd.DataFrame(columns=columns)
        iii = 1
        for i, j in enumerate(ceng_index):
            ceng = self.ceng_processed[i]
            try:
                zhang = zhang.append(
                    pd.DataFrame(
                        {
                            "层数": j,
                            "平均流速": ceng.zhang.mean,
                            "平均流向": ceng.zhang.mean_d,
                            "最大流速": ceng.zhang.extreme_v,
                            "最大流速对应方向": ceng.zhang.extreme_d,
                            "出现时间": ceng.zhang.extreme_t}))
            except BaseException:
                zhang = zhang.append(pd.DataFrame({"层数": j, "平均流速": 0, "平均流向": 0,
                                                   "最大流速": 0, "最大流速对应方向": 0, "出现时间": 0}, index=[iii]))
                iii += 1
            try:
                luo = luo.append(pd.DataFrame({"层数": j,
                                               "最大流速": ceng.luo.extreme_v,
                                               "最大流速对应方向": ceng.luo.extreme_d,
                                               "出现时间": ceng.luo.extreme_t,
                                               "平均流速": ceng.luo.mean,
                                               "平均流向": ceng.luo.mean_d}))
            except BaseException:
                luo = luo.append(pd.DataFrame({"层数": j, "平均流速": 0, "平均流向": 0,
                                               "最大流速": 0, "最大流速对应方向": 0, "出现时间": 0}, index=[iii]))
                iii += 1
        return zhang, luo

    def output_all(self):
        zhang, luo = self.out_put()
        zhang['涨落潮'] = '涨潮'
        luo['涨落潮'] = '落潮'
        statistics = zhang.append(luo, ignore_index=True)
        statistics['Point'] = self.point
        statistics['潮型'] = self.tide_type
        statistics['来源文件'] = self.filename
        self.out_data = statistics
        return self.out_data

    def out(self):
        try:
            try:
                return self.out_data
            except BaseException:
                self.preprocess()
                return self.out_data
        except BaseException:
            self.output_all()
            return self.out_data

    def out_ave_str_method(self):
        try:
            return self.out_ave_str
        except BaseException:
            self.out_ave_str_generate()
            return self.out_ave_str

    def out_txt_p(slef, row):
        return str_from_df(row['Point']) + '观测点在' + \
            str_from_df(row['潮型']) + '时'

    def out_ave_str_generate(self):
        self.out_ave_str = ''
        ceng_list = ['表层', '0.2H', '0.4H', '0.6H', '0.8H', '底层', '垂线平均']
        r_or_e = ['涨潮', '落潮']

        def out_ave_txt_c(row):
            return str_from_df(row['层数']) + '的平均流速为' + num_from_df(
                row['平均流速']) + 'cm/s（' + num_from_df(row['平均流向']) + '°）,'

        for _, p in self.out().groupby('Point'):
            self.out_ave_str += self.out_txt_p(p[0:1])
            # for _, c in p.groupby('层数'):
            # self.out_ave_str += out_ave_txt_c(c)
            for i_e in r_or_e:
                self.out_ave_str += '在' + i_e + '时,'
                for ceng in ceng_list:
                    r = p[(p['层数'] == ceng) & (p['涨落潮'] == i_e)]
                    self.out_ave_str += out_ave_txt_c(r)

    def out_times(self):

        def str_time2(x): return str(x.seconds // 3600) + '小时' + str(
            x.seconds % 3600 // 60 + 1) + '分钟' if x.seconds % 3600 % 60 > 30 else str(x.seconds // 3600) + '小时' + str(
            x.seconds % 3600 // 60) + '分钟'

        def str_time(x):
            try:
                return str_time2(x)
            except BaseException:
                return '00****00'

        content = '平均涨潮历时为' + \
            str_time(self.raise_time) + '，平均落潮历时为' + str_time(self.ebb_time) + '。'
        return content

    def out_times2(self):
        content = '平均涨潮历时为' + str(self.raise_time) + \
            '平均落潮历时为' + str(self.ebb_time)
        return content

    def change_one_dir_values(self, timeof=1, parameter=0.8):
        changed = []
        for i in self.ceng_processed:
            changed.append(
                i.change_one_dir_values(
                    timeof=timeof,
                    parameter=parameter))
        for i in range(0, 6):
            changed[i + 1] = changed[i +
                                     1].rename(columns={'v': 'v_' + str(i * 2), 'd': 'd_' + str(i * 2)})
        self.changed_out = pd.concat(changed, axis=1)
        self.changed_out = self.changed_out.T.drop_duplicates().T
        columns = [
            't',
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
        self.changed_out = self.changed_out[columns]
        self.changed_out = self.changed_out.reindex_axis(columns, axis=1)
        self.changed_out = self.changed_out.rename(columns={'t': 'time'})
        return self.changed_out

    def draw_dxf(self, parameter=10, ceng=0, drawing=None, *args):

        def plot_line(x, y, vs, ds, cengshu, parameter):
            for v, d in zip(vs, ds):
                line = dxf.line((x, y), end_point(x, y, v, d, parameter))

                drawing.add(line)
                # layer_name = dxf.layer(cengshu+self.tide_type)
                # drawing.layers.add(layer_name)
                line['layer'] = cengshu + self.tide_type

        def end_point(x, y, velocity, direction, parameter):
            v_east = velocity * np.sin(direction / 180 * np.pi)
            v_north = velocity * np.cos(direction / 180 * np.pi)
            return [x + v_east * parameter, y + v_north * parameter]

        data = self.ceng_processed[ceng].data
        data = data[data['t'].apply(lambda t: t.minute) == 0]
        # if self.cengshu == 6:
        cengshu_name = ['垂线平均', '表层', '0.2层', '0.4层', '0.6层', '0.8层', '底层']
        # else:
        #     cengshu_name = ['垂线平均', '表层', '中层', '底层']
        if not drawing:
            drawing = dxf.drawing(self.point + '流速矢量图.dxf')
        else:
            drawing = dxf.drawing(drawing)
        plot_line(
            self.x,
            self.y,
            data['v'],
            data['d'],
            cengshu_name[ceng],
            parameter)
        drawing.save()

    def display(self, fig_file=r"C:\fig.png"):

        try:
            parameter = 0.95 / np.nanmax(self.ceng_processed[1].data['v'].values)
        except:
            parameter = 0.95 / np.nanmax(self.ceng_processed[2].data['v'].values)
        if parameter == np.nan:
            print('绘图参数为Nan，请检查数据')
        if parameter == np.inf:
            parameter = 0.95 / self.ceng_processed[0].data['v'].values.max()
        if parameter == np.inf:
            parameter = 0.95 / self.ceng_processed[2].data['v'].values.max()

        def plot(data, e, n, timeof):
            nn = len(data)
            if nn > 1000:
                e1, e2, n1, n2, timeof1, timeof2 = [], [], [], [], [], []
                nn2 = int(nn / 2)
                data1 = data[:nn2]
                data2 = data[nn2:]
                for ccc in e:
                    e1.append(ccc[:nn2])
                    e2.append(ccc[nn2:])
                for ccc in n:
                    n1.append(ccc[:nn2])
                    n2.append(ccc[nn2:])
                for ccc in timeof:
                    timeof1.append(ccc[:nn2])
                    timeof2.append(ccc[nn2:])
                plot(data1, e1, n1, timeof1)
                plot(data2, e2, n2, timeof2)
                return 0
            fig, ax = plt.subplots(1, 1, figsize=(len(data) / 2 + 1, 8))
            for i in range(1, 8):
                for j in range(1, len(data) + 1):
                    try:
                        if timeof[7 - i][j - 1] == -1:
                            color = 'green'
                        elif timeof[7 - i][j - 1] == 1:
                            color = 'blue'
                        else:
                            color = 'black'
                    except BaseException:
                        color = 'black'
                    ax.arrow(j,
                             i,
                             e[7 - i][j - 1],
                             n[7 - i][j - 1],
                             head_width=0.1,
                             head_length=0.1,
                             fc=color,
                             ec=color)
            ax.set_xlim(0, len(data) + 1)
            ax.set_ylim(0, len(plotdatas) + 1)
            ax.set_xlabel('时间', fontsize=20)
            ax.yaxis.grid(True)
            if (data['t'].max() - data['t'].min()).days > 2:
                plt.xticks(
                    range(
                        1,
                        len(data) + 2),
                    data['t'].dt.strftime('%H:%M\n%m-%d'))
            else:
                plt.xticks(
                    range(
                        1,
                        len(data) + 2),
                    data['t'].dt.strftime('%H:%M'))
            plt.yticks(range(8), ['', '底层', '0.8H', '0.6H',
                                  '0.4H', '0.2H', '表层', '垂线平均'], fontsize=20)
            plt.title(self.point +
                      self.tide_type +
                      "(" +
                      min(data['t']).strftime("%m/%d - ") +
                      max(data['t']).strftime("%m/%d") +
                      "）", fontsize=25)
            plt.tight_layout()
            fig.savefig(fig_file +
                        "(" +
                        min(data['t']).strftime("%m_%d") +
                        max(data['t']).strftime("%m_%d") +
                        "）.png", dpi=fig.dpi *
                        2 /
                        3)
            print("(" +
                  min(data['t']).strftime("%m_%d ") +
                  max(data['t']).strftime("%m_%d") +
                  "）.png 文件输出结束")
            plt.close()
            # return fig

        plotdatas = self.ceng_processed
        e = []
        n = []
        timeof = []
        for i in plotdatas:
            data = i.data
            e.append(east(data['v'] * parameter, data['d']).values)
            n.append(north(data['v'] * parameter, data['d']).values)
            timeof.append(data['timeof'].values)

        plot(data, e, n, timeof)

    def output_proper_txt(self):
        return self.output_all().to_string() + '\n' + self.out_times()

    def disturbution_plot(self):
        v, d = [], []
        unit = "cm/s"
        for i in range(1, 7):
            for vi in self.ceng_processed[i].data['v'].values:
                v.append(vi)
            for di in self.ceng_processed[i].data['d'].values:
                d.append(di)
        print("共有" + str(len(v)) + "组流速流向数据")
        fig = plt.figure(figsize=(16, 9), dpi=200)
        ax = WindroseAxes.from_ax(fig=fig, rmax=None)
        ax.bar(d, v, normed=True,
               opening=0.8, edgecolor='white', bins=np.linspace(0, 200, 11), N=10, cmap=cm.rainbow)
        # ax.box(dir, var, normed=True, edgecolor='white', bins=bins,  cmap=cm.rainbow)
        ax.set_legend(title="流速" + unit if unit else item,
                      fancybox=True, facecolor='ivory', edgecolor='black', fontsize=15, bbox_to_anchor=(0,
                                                                                                        0),
                      decimal_places=0, ncol=3,
                      prop={'size': 9})
        ax.set_radii_angle(angle=20)
        ax.tick_params(
            axis='y',
            direction='inout',
            colors='darkblue',
            pad=1)
        ax.tick_params(axis='x', colors='black', labelsize=15, pad=2)
        ax.grid(color='black', linestyle=':', linewidth=1, alpha=1)
        title = self.point + " " + self.tide_type + ' 流速流向分布频级图'
        ax.set_title(title, {'fontsize': 20})
        fig_file = self.filename[-10:] + title + '.png'
        fig.savefig(
            fig_file.replace(
                r'/',
                '_'),
            dpi=200,
            bbox_inches='tight')
        print(fig_file.replace(r'/', '_') + '保存成功')
        plt.close()
        print(self.point + self.tide_type + 'OK')

    def cal_crosscurrent(self, dock_dir):
        current_time = self.ceng_processed[0].data['t']
        Cross = self.ceng_processed[0].cal_crosscurrent_of_df(dock_dir)
        return pd.Series(data=Cross, index=current_time)


class Read_Report():
    def __init__(self, filename, sediment=False):
        def search_row(row, key):
            got = []
            for i, s in enumerate(row.astype(str).tolist()):
                if key in s:
                    got.append(row[i])
            return got

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
            d[-1] += 1  # 报表最后一列数据
            for i in range(round(len(d) / 2)):
                dfs.append(df.iloc[d[2 * i]:d[2 * i + 1], :])
            return dfs

        def identify(df):
            names = []

            for i in range(5):
                for keys in ['潮汛', '潮型', '测站', '测点']:
                    got = search_row(df.iloc[i, :], keys)
                    for x in got:
                        names.append(x)
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

        if not sediment:
            def format_df(df):
                df = df.set_index(pd.to_datetime(df.iloc[:, 1], errors='coerce'))  # 不用将时间放到第一列即可
                # df = df.reset_index().dropna().set_index('index') 读入只有三层数据时候会全部删除数据，因此改为一下四行
                df = df.loc[df.index.dropna(), :]  # 去除无效时间/多余行
                # df = df.iloc[:,1:]#去除第一列
                df = df.drop(df.columns[1], axis=1)
                if not len(df.T) == 17:
                    df = df.dropna(axis=1)  # 去除多余列
                df.loc[:, df.isnull().sum() != 0] = 0.00  # 非六层情况时，填充无数据部分

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
                df['time'] = df.index
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
                                   'v',
                                   'd'],
                                  axis=1).reset_index()
        else:
            def format_df(df):
                df = df.set_index(pd.to_datetime(df.iloc[:, 2], errors='coerce'))  # 时间在第三列
                # df = df.reset_index().dropna().set_index('index') 读入只有三层数据时候会全部删除数据，因此改为一下四行
                df = df.loc[df.index.dropna(), :]  # 去除无效时间/多余行
                # df = df.iloc[:,1:]#去除第一列
                df = df.drop(df.columns[:3], axis=1)
                if not len(df.T) in [4, 7]:
                    df = df.dropna(axis=1)  # 去除多余列
                if len(df.T) == 7:
                    df.columns = [
                        'Surface',
                        '0.2H',
                        '0.4H',
                        '0.6H',
                        '0.8H',
                        'Bottom',
                        'Ave']
                if len(df.T) == 4:  # 正常时为表中底，水浅时为268
                    df.columns = [
                        'Surface',
                        '0.6H',
                        'Bottom',
                        'Ave']
                    # df.columns = [
                    #     '0.2H',
                    #     '0.6H',
                    #     '0.8H',
                    #     'Ave']
                df['time'] = df.index
                return df.reindex(['time',
                                   'Surface',
                                   '0.2H',
                                   '0.4H',
                                   '0.6H',
                                   '0.8H',
                                   'Bottom',
                                   'Ave'],
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

        def read_excel_report(fileName):
            excel = pd.ExcelFile(fileName)
            c = {}
            for i in excel.sheet_names:
                df = excel.parse(sheet_name=i)
                dfs = split_df(df)
                every_page = sort_df(dfs)
                cengshu = 6
                while True:
                    try:
                        ii = next(every_page)
                    except BaseException:
                        print('*' * 10)
                        print(i)
                        break
                    if (((ii.Data != 0).sum() == 0)).sum() == 6:
                        cengshu = 3
                    if (((ii.Data != 0).sum() == 0)).sum() == 8:
                        cengshu = 2

                    Single_Tide_Point_data = Single_Tide_Point(
                        filename=filename,
                        point=ii.Point,
                        tide_type=ii.Tide_type,
                        angle=None,
                        format_report=True,
                        data=ii.Data,
                        cengshu=cengshu)
                    if ii.Point in c:
                        c[ii.Point].update(
                            {ii.Tide_type: Single_Tide_Point_data})
                    else:
                        c.update(
                            {ii.Point: {ii.Tide_type: Single_Tide_Point_data}})
                    print(ii.Point + ii.Tide_type + str(cengshu) + '层载入OK')
            self.points = sorted(list(c.keys()))
            self.tide_type = list(c[self.points[0]].keys())
            for i in self.points:
                if len(c[i].keys()) == len(self.tide_type):
                    pass
                else:
                    print('点' +
                          str(i) +
                          '所载入潮型为' +
                          str(c[i].keys()) +
                          ',与其余测点的' +
                          str(self.tide_type) +
                          '不符')
                    raise ValueError('各点所载入潮型不等')
            return c

        self.data = read_excel_report(filename)

    def process(self):
        c = threading.active_count()
        self.list_of_thread = []
        self.xx = []
        for _, j in self.data.items():
            if isinstance(j, dict):
                for __, jj in j.items():
                    process_thread = threading.Thread(target=jj.preprocess)
                    self.list_of_thread.append(process_thread)
            else:
                process_thread = threading.Thread(target=j.preprocess)
                self.list_of_thread.append(process_thread)
        for i in self.list_of_thread:
            i.start()
        while threading.active_count() != c:
            time.sleep(1)
            self.is_processed = False
        self.is_processed = True

    def setPoint_ang(self, Point, ang):
        for _, data in self.data[Point].items():
            data.set_angle(angle=ang)


class time_v_d(object):
    def __init__(self, data, angle, ang=0, zhang_or_luo=False):  # angle为落潮流向,ang为转流区域角度
        for i in data.columns:
            if 'v' in i:
                # data['v'] = data[i]
                data.loc[data.index, 'v'] = data[i]
            if 'd' in i:
                data.loc[data.index, 'd'] = data[i]
            if 't' in i:
                # data['t'] = pd.to_datetime(data[i])
                data.loc[data.index, 't'] = pd.to_datetime(data[i])

        self.zhang, self.luo = None, None  # 防止出现全涨/落的情况，后续可能引发bug 2019/3/4

        self.data = data[['v', 'd', 't']]
        self.data.loc[self.data.index, 'timeof'] = self.data['d'].apply(
            lambda x: is_time_of(x)(angle, ang))

        fenzhangluo = self.data.groupby('timeof')
        for ii, jj in fenzhangluo:
            if ii == 1:  #
                self.zhang = v_and_d_of_one_dir(jj)
            if ii == -1:
                self.luo = v_and_d_of_one_dir(jj)

        if not zhang_or_luo:
            self.zhang, self.luo = self.luo, self.zhang

    def change_one_dir_values(self, timeof=1, parameter=0.8):
        list_to_change = self.data[self.data['timeof'] == timeof]
        self.data.loc[list_to_change.index,
                      'v'] = list_to_change['v'] * parameter
        return self.data

    def harmonic_analysis(self):
        self.e = east(self.data['v'], self.data['d'])
        self.n = north(self.data['v'], self.data['d'])
        times = date2num(self.data['t'])
        result = utide.solve(
            times,
            self.e.values,
            self.n.values,
            lat=30,
            constit=constit_tide_level)
        self.constit_result = pd.DataFrame(
            data=[
                result['Lsmaj'],
                result['Lsmaj'],
                result['Lsmin'],
                result['theta'],
                result['g']],
            index=[
                'Lsmaj',
                'Lsmaj',
                'Lsmin',
                'theta',
                'g'],
            columns=result['diagn']['name'])

    def cal_crosscurrent_of_df(self, dock_dir):  # 码头角度
        cal_corss_current = lambda v, d: abs(v * np.sin((d - dock_dir) / 180 * np.pi))
        return aoe(cal_corss_current, self.data['v'], self.data['d'])
        # 需要加入最大可能流速和余流的计算

class v_and_d_of_one_dir(object):
    def __init__(self, v_and_d):
        self.data = v_and_d
        self.extreme_v = v_and_d.v.max()
        self.extreme_d = v_and_d.loc[v_and_d['v'] == self.extreme_v]['d']
        t = (v_and_d.loc[v_and_d['v'] == self.extreme_v]['t'])
        self.extreme_t = t.values[0]
        # v_and_d['x'] = v_and_d['t'].apply(lambda x: x.minute)
        v_and_d.loc[v_and_d.index, 'x'] = v_and_d['t'].apply(
            lambda x: x.minute)
        v_and_d2 = v_and_d[v_and_d['x'] == 0]  # 删除非整点数据
        if v_and_d2.empty:  # 如果没有整点数据，则全部参与计算
            v_and_d2 = v_and_d
        # v_and_d2['v_e'] = v_and_d2.apply(lambda df: east(df['v'], df['d']), axis=1)
        v_and_d2.loc[v_and_d2.index, 'v_e'] = v_and_d2.apply(
            lambda df: east(df['v'], df['d']), axis=1)
        # v_and_d2['v_n'] = v_and_d2.apply(lambda df: north(df['v'], df['d']),
        # axis=1)10-23版本跟进pd
        v_and_d2.loc[v_and_d2.index, 'v_n'] = v_and_d2.apply(
            lambda df: north(df['v'], df['d']), axis=1)
        self.mean = velocity(v_and_d2.v_e.mean(),
                             v_and_d2.v_n.mean())  # 平均流速为矢量各分量平均之后的合成流速
        self.mean_d = direction(v_and_d2.v_e.mean(), v_and_d2.v_n.mean())

    def output_str(self, zhang_or_luo):
        if zhang_or_luo == "zhang":
            index = "涨潮"
        if zhang_or_luo == "luo":
            index = "落潮"
        result = ""
        result += index + '平均流速为' + \
            str_from_float(self.mean), ',平均流向为' + str_from_float(self.mean_d)
        result += "\n"
        result += index + "流速最大为" + str_from_float(self.extreme_v), ',其对应流向为' + str_from_float(
            self.extreme_d.values[0]), '出现在' + str_from_datatime64(self.extreme_t)
        return result


def time_A2B(time_A, v_A, dir_A, time_B, v_B, dir_B, dir_cal):
    # dir_cal为转流经过的方向，在实际应用中应为涨/落潮方向垂直角度
    if not is_d1_in_d2_and_d3(dir_cal, dir_A, dir_B):
        dir_cal = dir_cal + 180 if dir_cal < 180 else dir_cal - 180

    ang_A = small_diff_dir(dir_A, dir_cal) / 180 * np.pi
    ang_B = small_diff_dir(dir_B, dir_cal) / 180 * np.pi
    projection_A = abs(v_A * np.sin(ang_A))
    projection_B = abs(v_B * np.sin(ang_B))
    delta_time = time_B - time_A
    mid_time = pd.to_timedelta(
        projection_A / (projection_A + projection_B) * delta_time) + time_A
    return mid_time


def add_convert_row_to_tvd_and_timeof(time_v_d, angle):
    # angle为涨、落潮角度
    while time_v_d.loc[0, 'timeof'] == 0:  # 去掉开始和结束时候就是转流的情况
        time_v_d = time_v_d.drop(0, axis=0)
        time_v_d = time_v_d.reset_index(drop=True)
    while time_v_d.loc[len(time_v_d) - 1, 'timeof'] == 0:
        time_v_d = time_v_d.drop(len(time_v_d) - 1, axis=0)
        time_v_d = time_v_d.reset_index(drop=True)

    turn_group = time_v_d[time_v_d['timeof'] == 0]

    mid_in = []
    first_and_last_of_turn = turn_group.copy()
    for i in range(len(turn_group)):
        if (turn_group.index[i] -
            1 in turn_group.index) and (turn_group.index[i] +
                                        1 in turn_group.index):
            first_and_last_of_turn = first_and_last_of_turn.drop(
                turn_group.index[i])
            mid_in.append(turn_group.index[i])
    for i in first_and_last_of_turn.index:
        time_v_d.loc[i, 'timeof'] = time_v_d.loc[i - 1,
                                                 'timeof'] / 2 + time_v_d.loc[i + 1, 'timeof'] / 2
        if abs(time_v_d.loc[i, 'timeof']) == 1:  # 去除单向内出现转流情况
            first_and_last_of_turn = first_and_last_of_turn.drop(i, axis=0)
        if abs(time_v_d.loc[i, 'timeof']) == 0:  # 去除转流时出现转流的情况
            first_and_last_of_turn = first_and_last_of_turn.drop(i, axis=0)

    time_v_d = time_v_d.sort_values(by='t')
    time_v_d = time_v_d.reset_index(drop=True)

    count_turn = int(len(first_and_last_of_turn) / 2)
    if count_turn * 2 != len(first_and_last_of_turn):
        print('*' * 10 + '转流时间筛选不完整' + '*' * 10)
    for i in range(count_turn):
        index_first = first_and_last_of_turn.index[2 * i] - 1
        index_second = first_and_last_of_turn.index[i * 2 + 1] + 1
        if time_v_d.loc[index_first,
                        'timeof'] == time_v_d.loc[index_second,
                                                  'timeof']:
            time_v_d.loc[index_first:index_second,
                         'timeof'] = time_v_d.loc[index_first, 'timeof']

        else:
            A = time_v_d.loc[index_first]
            B = time_v_d.loc[index_second]
            t = time_A2B(A.t, A.v, A.d, B.t, B.v, B.d, dir_in_360(angle + 90))
            new_row = pd.DataFrame(
                {
                    't': [
                        t -
                        pd.Timedelta(
                            seconds=30),
                        t +
                        pd.Timedelta(
                            seconds=30)],
                    'timeof': [
                        A.timeof,
                        B.timeof]})
            new_row['time'] = new_row['t']
            time_v_d = time_v_d.append(new_row, ignore_index=True)

    time_v_d = time_v_d.drop(time_v_d[time_v_d['timeof'].apply(
        lambda x: x not in [-1, 1])].index, axis=0)
    time_v_d = time_v_d.sort_values(by='t')
    time_v_d = time_v_d.reset_index(drop=True)

    temp = pd.DataFrame(columns=['t', 'v', 'd', 'timeof'])

    """for i in range(0,len(first_and_last_of_turn)):
        A = time_v_d.loc[i - 1]
        B = time_v_d.loc[i + 1]
        if A.timeof*B.timeof >1 :"""

    for i in range(0, len(time_v_d) - 1):
        A = time_v_d.loc[i]
        B = time_v_d.loc[i + 1]
        if A.timeof * B.timeof == -1 and B.t - A.t > pd.Timedelta('10m'):
            t = time_A2B(A.t, A.v, A.d, B.t, B.v, B.d, dir_in_360(angle + 90))
            new_row = pd.DataFrame(
                {
                    't': [
                        t -
                        pd.Timedelta(
                            seconds=30),
                        t +
                        pd.Timedelta(
                            seconds=30)],
                    'timeof': [
                        A.timeof,
                        B.timeof]})
            temp = temp.append(new_row, ignore_index=True)

    time_v_d = time_v_d.append(temp, ignore_index=True)
    time_v_d = time_v_d.sort_values(by='t')
    time_v_d['time'] = time_v_d['t']
    time_v_d = time_v_d.reset_index(drop=True)
    return time_v_d


def draw_dxf(multi_point, parameter, filename):
    drawing = dxf.drawing('filename')
    for i in multi_point:
        i.draw_dxf(drawing=drawing)
    return drawing


days_d = [1, 2, 3, 15, 16, 17]
days_x = [8, 9, 10, 22, 23, 24]


def Spring_or_Neap(day):
    if day in days_d:
        return "Spring"
    if day in days_x:
        return "Neap"
    else:
        return "Moderate"


def plot_sediment(file, outdir=None):
    if outdir:
        os.chdir(outdir)
    c = Read_Report(file, sediment=True)
    for P_name, P in c.data.items():
        for Tidetype, single_survey in P.items():
            countourf_sand_consistency(Tidetype + P_name + "含沙量分布图", single_survey.data)


def countourf_sand_consistency(title, df):
    df = df.dropna(axis=1).set_index('time').iloc[:, 1:-1]
    if len(df.T) == 3:
        y = [1, 0.6, 0]  # 汕尾特殊情况，0.2、0.6、0.8三层，其他地方应该为0，0.6，1
    if len(df.T) == 6:
        y = [0, 0.2, 0.4, 0.6, 0.8, 1]
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(14, 3), dpi=100)
    CS = ax.contourf(x, y, df.T.values, cmap=plt.cm.Greys, vmin=0)
    plt.xticks(range(len(df)), df.index.strftime('%H:%M'), rotation=90)
    Y_index = df.columns.to_list()
    #Y_index.reverse()
    plt.yticks(y, Y_index)  # 倒序的columns
    cbar = fig.colorbar(CS, pad=0.002)
    cbar.set_label('含沙量' + '$(kg/m^3)$')
    # ax.clabel(CS, fk(CS.levels), inline=True, fontsize=10,colors  = 'k', fmt = '%.2f')
    ax.clabel(CS, inline=False, fontsize=10, colors='k', fmt='%.2f')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(title + ".png", dpi=200)
    print(title + "出图完成。")
    plt.close()


def cal_sediment_tranfer(sed_file, current_file, DictOfColumn):
    # DictOfColumn = {'v_2':'0.2H','v_6':'0.6H','v_8':'0.8H'}
    s = Read_Report(sed_file, sediment=True)
    c = Read_Report(current_file)

    for Pname, P in s.data.items():
        for Tide, df in P.items():
            df_current = c.data[Pname][Tide].data.set_index('time')
            df_sediment = df.data.set_index('time')
            for c_current, c_sediment in DictOfColumn.items():
                for index in df_sediment.index:
                    # print(Pname + Tide + column +" " + str(index) + "这就搞定~")
                    df_current.loc[index, c_current] = df_sediment.loc[index, c_sediment]  # 流速换为含沙量
                    # * df_current.loc[index, c_current] 将流速换算为输沙量
            df_current.loc[:, 'v'] = df_current.loc[:, list(DictOfColumn.keys())].T.mean()  # 含沙量垂线平均为算术平均值
            c.data[Pname][Tide].data = df_current.loc[df_sediment.index]
            c.data[Pname][Tide].data = c.data[Pname][Tide].data.fillna(
                {'v_6': 0, 'v_2': 0, 'v_8': 0, 'v_10': 0, 'v_0': 0, 'v_4': 0})

            c.data[Pname][Tide].data.loc[:, 'time'] = c.data[Pname][Tide].data.index
            c.data[Pname][Tide].data.loc[:, 'Unnamed:1'] = c.data[Pname][Tide].data.index

            c.data[Pname][Tide].data = c.data[Pname][Tide].data.reindex(
                ['Unnamed:1', 'time', 'depth', 'v_0', 'd_0', 'v_2', 'd_2', 'v_4', 'd_4', 'v_6', 'd_6', 'v_8', 'd_8',
                 'v_10', 'd_10', 'v', 'd'], axis=1).reset_index(drop=True)
    return c


if __name__ == "__main__":
    # s = r"C:\2020浙江交通集团舟山建筑工业化分公司预制基地码头工程\8. 报告撰写\附录D：含沙量报表.xls"
    # c = r"C:\2020浙江交通集团舟山建筑工业化分公司预制基地码头工程\6. 数据处理\附录B：定点潮流观测报表-整点部分表中底格式.xlsx"
    # D = {'v_0': 'Surface', 'v_2': '0.2H', 'v_6': '0.6H', 'v_8': '0.8H', 'v_10': 'Bottom'}
    # sediment_tranfer_result = cal_sediment_tranfer(s, c, D)
    # f = open(r"C:\2020浙江交通集团舟山建筑工业化分公司预制基地码头工程\8. 报告撰写\含沙量读取结果.python_dump", 'wb')
    # pickle.dump(sediment_tranfer_result, f)
    # f.close()

    plot_sediment(r"C:\2020浙江交通集团舟山建筑工业化分公司预制基地码头工程\8. 报告撰写\附录D：含沙量报表.xls",
                  r"C:\2020浙江交通集团舟山建筑工业化分公司预制基地码头工程\8. 报告撰写\含沙量出图")
