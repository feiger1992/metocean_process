import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools
import datetime
EWSN2d = { "NNW":337.5, "NW":315, "WNW":292.5, "W":270, "WSW":247.5, "SW":225, "SSW":202.5, "S":180, "SSE":157.5, "SE":135, "ESE":112.5,"E":90,"ENE":67.5,"NE":45,"NNE":22.5,"N":0}
EWSN2d_s = pd.Series(EWSN2d)
EWSN2d_s = EWSN2d_s.sort_values()
east = lambda v, d: v * np.sin(d / 180 * np.pi)
north = lambda v, d: v * np.cos(d / 180 * np.pi)
velocity = lambda v_e, v_n: np.sqrt(v_e ** 2 + v_n ** 2)
dir_in_360b = lambda d: (d - 360 if d >= 360 else d) if d > 0 else  360 + d
dir_in_360 = lambda d: dir_in_360b(d) if (dir_in_360b(d) >= 0 and (dir_in_360b(d) < 360)) else dir_in_360b(
    dir_in_360b(d))
direction = lambda v_e, v_n: dir_in_360(180 / np.pi * np.arctan2(v_e, v_n)) if (v_e > 0)else   dir_in_360(
    180 / np.pi * np.arctan2(v_e, v_n) + 360)
fun = lambda x:[y for t in x for y in t]
def wind_power2(v):
    if v <5.4 :
        return 3
    if v <7.9   :
        return 4
    if v <10.7:
        return 5
    if v <13.8:
        return 6
    if v <17.1:
        return 7
    if v <20.7:
        return 8
    if v <24.4:
        return 9
    if v >=24.4:
        return 10
def wind_power(v):
    try:
        return wind_power2(v)
    except:
        try:
            v = pd.to_numeric(v)
            return wind_power2(v)
        except:
            return pd.NaT
def f_EWSN2d(EWSN):
    try:
        return EWSN2d[EWSN]
    except:
        if len(EWSN)==2:
            return EWSN2d[EWSN[::-1]]
        else:
            return EWSN2d[EWSN[0]+EWSN[1:][::-1]]
num2pi = lambda d : d/180*np.pi
def d2EWSN(d):
    if d > 360 or d < 0:
        raise ValueError('请检查方向数据')
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

def process_windFile(file):
    df = pd.read_csv(file)
    for i in df.columns:
        if '向' in i:
            df[i] = df[i].apply(d2EWSN)
    df.to_csv(file.replace('.csv','风向格式转化.csv'))


def polor_d_density(df):
    lables = ["N","NE","E","SE","S","SW","W","NW"]
    g = df.groupby('d')
    r = g.size() / g.size().sum()
    r = r.reindex(EWSN2d_s.index)
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(lables,size = 30)

    bar = ax.fill(EWSN2d_s / 180 * np.pi, r)
    return ax.get_figure()


def split_and_insert(list_of_str, index, length, reverse=True):
    # 将位于index位置上的字符串拆分length个字符移入前|后
    str = list_of_str
    if not reverse:
        str.insert(index, str[index][:length])
        str[index + 1] = str[index + 1][length:]
    else:
        str.insert(index + 1, str[index][-length:])
        str[index] = str[index][:-length]
    return str


add_colon_to_time_data = lambda arg: arg[:2] + ':' + arg[2:]
add_comma_to_str = lambda arg,index: arg[:index]+ ','+arg[index:]



def process(line):
    line = line.replace('-', ' ')
    line = ' '.join(line.split())
    datas = line.split(' ')
    datas = list(filter(None, datas))
    if '///////' in line:
        ls = add_comma_to_str(datas[-2], 3)
        ls = add_comma_to_str(ls, 9)
        ls = add_comma_to_str(ls, 15)
        ls = add_comma_to_str(ls, 20)
        ls = add_comma_to_str(ls, 26)
        datas[-2] = ls
        datas[-4] = add_comma_to_str(datas[-4],2)
        datas[-7] = add_comma_to_str(datas[-7],3)
        datas[-8] = add_comma_to_str(datas[-8], 3)
        datas[-12] = add_comma_to_str(datas[-12], 2)
        datas = ','.join(datas).replace('/' * 28, ',' * 7).replace('/' * 20, ',' * 5).split(sep=',')
        for i in [0,-2,-4,-9,-13,-15,-20]:
            datas[i] = add_colon_to_time_data(datas[i])
        del  datas[21]
        return ','.join(datas) + ',\n'

    if len(datas) == 1:
        return ','.join(datas) + '\n'
    if len(datas[-1:][0]) < 10:
        #print(datas[:-1])
        return ','.join(datas) + '\n'
    #print(datas[0], '*****', len(datas))
    try:
        if len(datas) == 22:
            split_and_insert(datas, len(datas) - 2, 3, False)
            split_and_insert(datas, len(datas) - 2, 4)
            split_and_insert(datas, len(datas) - 3, 5)
            split_and_insert(datas, len(datas) - 4, 4)
            split_and_insert(datas, len(datas) - 5, 5)
        else:
            split_and_insert(datas, len(datas) - 2, 4)
            split_and_insert(datas, len(datas) - 4, 4)
        if len(datas[-5]) == 9:
            split_and_insert(datas, len(datas) - 5, 5)
        if len(datas) == 25:
            split_and_insert(datas, len(datas) - 5, 5)
            split_and_insert(datas, len(datas) - 6, 5)

        split_and_insert(datas, len(datas) - 1, 120)
        split_and_insert(datas, 18, 4)
        split_and_insert(datas, 15, 4)
        split_and_insert(datas, 14, 4)
        split_and_insert(datas, 10, 4)
        split_and_insert(datas, 6, 4)
    except:
        return ','.join(datas) + '\n'
    del datas[21]
    for i in [0, 7, 12, 17, 19, 22, 27, 29]:
        datas[i] = add_colon_to_time_data(datas[i])
    return ','.join(datas) + '\n'

def process_file(filenames,filename_toin,*args):#处理每分钟的气象数据，每个文件里面是一天的气象数据
    with open(filename_toin,mode='w',encoding='utf-8') as f2:
        line1 = '\uFEFFt,2min平均风向,2min平均风速,10min平均风向,10min平均风速,最大风速对应风向,最大风速,最大风速出现时间,分钟内最大瞬时风速对应风向,分钟内最大瞬时风速,极大风向,极大风速,极大风速出现时间,分钟降水量,小时累积降水量,气温,最高气温,最高气温出现时间,最低气温,最低气温出现时间,相对湿度,最小相对湿度,最小相对湿度出现时间,水汽压,露点温度,本站气压,最高本站气压,最高本站气压出现时间,最低本站气压,最低本站气压出现时间,质量控制码,一小时内六十个分钟降水量\n'
        f2.writelines(line1)
        for filename in filenames:
            with open(filename) as file:
                next(file)
                for line in file:
                    try:
                        line = process(line)
                    except IndexError:
                        pass
                    if len(line) >20:
                        datetime_str = os.path.basename(filename).replace('GG','').replace('.DAT','')
                        y = datetime_str[:2]
                        m = datetime_str[2:4]
                        d = datetime_str[4:]
                        f2.writelines(y+'-'+m+'-'+d+' '+line)
                    else:
                        print('*'*10+'+'*10)
                        print(line)
                        print(filename)
                        print('*'*10+'-'*10)
    c = ['t', '2min平均风向', '2min平均风速', '10min平均风向', '10min平均风速',
     '最大风速对应风向', '最大风速', '最大风速出现时间', '极大风向', '极大风速', '极大风速出现时间', '分钟降水量',
     '小时累积降水量', '气温', '最高气温', '最高气温出现时间', '最低气温', '最低气温出现时间', '相对湿度',
     '最小相对湿度', '最小相对湿度出现时间', '本站气压', '最高本站气压', '最高本站气压出现时间', '最低本站气压',
     '最低本站气压出现时间']
    df = pd.read_csv(filename_toin,usecols=c)

    df['t'] = pd.to_datetime(df.t, format='%y-%m-%d %H:%M',errors='coerce')
    df = df.dropna(axis=0, subset=['t'])
    changeDayIndex = df[df.t.apply(lambda x: x.time()) > datetime.time(20, 0)].index
    df.loc[changeDayIndex,'t'] -= pd.tseries.offsets.Day(1)
    df = df.set_index('t',drop = True)
    df = df.sort_index()
    df.to_csv(filename_toin,index_label='北京时间')
mar_haibian_files = [r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180330.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180331.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180301.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180302.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180303.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180304.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180305.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180306.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180307.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180308.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180309.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180310.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180311.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180312.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180313.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180314.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180315.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180316.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180317.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180318.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180319.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180320.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180321.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180322.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180323.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180324.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180325.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180326.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180327.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180328.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\data\GG180329.DAT"]
Mar_conbine_file = r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\3月\海边气象站\1min数据合并.csv"
process_file(mar_haibian_files,Mar_conbine_file)
print("OK")
def processonefile(f,isV=True,df1=None,M1_max_index=None,M10_max_index=None):#处理分方向平均数值
    if not isV:
        s ='风向'
    else:
        s = '风速'
    t1 = pd.Timestamp('20' + f[-10:-4]+'2000') - pd.tseries.offsets.Day(1)
    Df_index = pd.date_range(start=t1, periods=1440, freq='min')
    df_r = pd.read_csv(f, header=None, sep='[ ]*', skipinitialspace=True)
    df_r = df_r.drop(0, axis=1)
    df_r_T = df_r.T
    if isV:
        g = df_r_T.groupby(by=(df_r_T.index - 1) // 4).mean()  # 纵轴为秒，横轴为分钟步进时间
    else:
        g = df_r_T
    mean3 = g.groupby(by=g.index // 3).mean()  # 纵轴三秒一进
    mean15 = g.groupby(by=g.index // 15).mean()  # 纵轴15秒一进
    mean1min = g.mean(axis=0)
    S3 = pd.Series(index=Df_index, data=mean3.iloc[0, :].values, name='三秒平均'+s)
    S15 = pd.Series(index=Df_index, data=mean15.iloc[0, :].values, name='十五秒平均'+s)
    M1 = pd.Series(index=Df_index, data=mean1min.values, name='1分钟平均'+s)
    df = pd.DataFrame([S3, S15, M1]).T
    if isV:
        M1_max = g.max(axis=0)
        #df['maxIn1'] = M1_max.values
        M1_max_index = g[g == M1_max].notnull()

        g_10 = M1_max.groupby(lambda x: x // 10)
        #df['maxIn10'] = g_10.cummax(axis=0).values
        M10_max_index = g[g == g_10.cummax(axis=0)].notnull()
        return df,M1_max_index,M10_max_index
    else:
        g = g.set_index(g.index-1)
        df = pd.merge(df1,df,how = 'outer',left_index=True,right_index=True)
        #df['maxIn1'+s]=g[M1_max_index].bfill()[0:1].values[0]
        #df['maxIn10'+s] = g[M10_max_index].bfill()[0:1].values[0]
        return df

def process_file(speedFilenames,dirFilenames):
    if len(speedFilenames) != len(dirFilenames):
        raise ValueError("输入文件应相互对应")
    dfCombine = pd.DataFrame()
    for fV in speedFilenames:
        fD = [i for i in dirFilenames if fV[-10:-4] in i][0]

        df,M1_max_index,M10_max_index = processonefile(fV)
        df = processonefile(fD,isV=False,df1 = df,M1_max_index = M1_max_index,M10_max_index=M10_max_index)
        dfCombine = dfCombine.append(df)
    dfCombine = dfCombine.sort_index()
    dfCombine.fillna(method='ffill',axis=0,inplace=True)
    Mean2 = dfCombine.resample("2min")['1分钟平均风速', '1分钟平均风向'].mean()
    Mean2 = Mean2[Mean2.index.minute == 0]
    Mean2 = Mean2.rename(index=None, columns={'1分钟平均风速': '2分钟平均风速', '1分钟平均风向': '2分钟平均风向'})
    Mean10 = dfCombine.resample("10min")['1分钟平均风速', '1分钟平均风向'].mean()
    Mean10 = Mean10[Mean10.index.minute == 0]
    Mean10 = Mean10.rename(index=None, columns={'1分钟平均风速': '10分钟平均风速', '1分钟平均风向': '10分钟平均风向'})
    dfCombine = dfCombine[dfCombine.index.minute == 0]
    dfCombine = pd.merge(dfCombine, Mean2, how='outer', left_index=True, right_index=True)
    dfCombine = pd.merge(dfCombine, Mean10, how='outer', left_index=True, right_index=True)
    return dfCombine
dirFilenames= [r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171119.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171120.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171121.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171122.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171123.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171031.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171101.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171102.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171103.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171104.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171105.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171106.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171107.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171108.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171109.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171110.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171111.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171112.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171113.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171114.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171115.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171116.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171117.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WD171118.DAT"]
speedFilenames =[r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171106.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171105.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171104.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171103.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171102.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171101.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171031.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171123.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171122.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171121.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171120.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171119.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171118.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171117.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171116.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171115.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171114.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171113.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171112.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171111.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171110.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171109.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171108.DAT",r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\samp\WS171107.DAT"]
filename_toin = r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\11月\海边\流向数据平均值.csv"
#df = process_file(speedFilenames,dirFilenames)
#df.to_csv(filename_toin,index_label='北京时间')

def process_combine_csv_1(combine_csv):
    c = ['北京时间', '2min平均风向', '2min平均风速', '10min平均风向', '10min平均风速',
     '最大风速对应风向', '最大风速', '最大风速出现时间', '极大风向', '极大风速', '极大风速出现时间', '分钟降水量',
     '小时累积降水量', '气温', '最高气温', '最高气温出现时间', '最低气温', '最低气温出现时间', '相对湿度',
     '最小相对湿度', '最小相对湿度出现时间', '本站气压', '最高本站气压', '最高本站气压出现时间', '最低本站气压',
     '最低本站气压出现时间']
    try:
        df = pd.read_csv(combine_csv,usecols=c)
    except:
        df = pd.read_csv(combine_csv,usecols=c,encoding = 'GB2312')
    print('总汇总文件读取成功')
    df['t'] = pd.to_datetime(df['北京时间'])
    df = df.drop('北京时间',axis=1)

    df.loc[df['本站气压'] < 500 ,'本站气压'] +=10000

        #北京时间转化为UTC时间
    df['t'] = df['t'] - pd.tseries.offsets.Hour(8)
    #df = df[df['t'].apply(lambda t: t.month) == 10]
    df = df.sort_values(by='t')
    df = df.set_index('t', drop=True)
    to_UTCtime = lambda columnName,df: (pd.to_datetime(df[columnName]) - pd.tseries.offsets.Hour(8)).dt.time
    df_h = df.loc[pd.date_range(pd.datetime(2017, 12, 1), pd.datetime(2018, 1, 1), freq='H')]

    df_humidity = df_h[['相对湿度', '最小相对湿度']].copy()
    df_humidity['平均湿度'] = round(df['相对湿度'].groupby(pd.TimeGrouper(freq='H')).mean())
    df_humidity['最小相对湿度出现时间'] = to_UTCtime('最小相对湿度出现时间',df_h)
    df_humidity = df_humidity[['相对湿度','最小相对湿度','最小相对湿度出现时间','平均湿度']]

    df_pressure = df_h[['本站气压', '最高本站气压', '最低本站气压']].copy()/10
    df_pressure['最高本站气压出现时间'],df_pressure['最低本站气压出现时间'] = to_UTCtime('最高本站气压出现时间',df_h),to_UTCtime('最低本站气压出现时间',df_h)
    df_pressure['平均气压'] = round(df['本站气压'].groupby(pd.TimeGrouper(freq='H')).mean())/10
    df_pressure = df_pressure[['本站气压','最高本站气压','最高本站气压出现时间','最低本站气压','最低本站气压出现时间','平均气压']]

    df_rain = df_h[['分钟降水量', '小时累积降水量']].copy()/10

    df_temperatur = df_h[['气温', '最高气温', '最低气温',]].copy()/10
    df_temperatur['最高气温出现时间'],df_temperatur['最低气温出现时间'] =to_UTCtime('最高气温出现时间',df_h),to_UTCtime('最低气温出现时间',df_h)
    df_temperatur = df_temperatur[['气温', '最高气温', '最高气温出现时间', '最低气温', '最低气温出现时间']]
    df_temperatur['小时平均气温'] = df['气温'].groupby(pd.TimeGrouper(freq='H')).mean()/10

    """
    df_wind =df_h[["2min平均风速","10min平均风速","最大风速","极大风速"]].copy()/10
    df_wind["最大风速出现时间"],df_wind["极大风速出现时间"] = to_UTCtime("最大风速出现时间",df_h),to_UTCtime("极大风速出现时间",df_h)
    for i in ["2min平均风向","10min平均风向","极大风向","最大风速对应风向"]:
        df_wind[i] = df_h[i]
        df_wind[i+'__格式'] = df_wind[i].apply(d2EWSN)
    df_wind = df_wind[['2min平均风向__格式','2min平均风速','10min平均风向__格式','10min平均风速']]
    """
    df_humidity.to_csv(r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\12月\门口\12_M2_humidity(UTC).csv")
    df_pressure.to_csv(r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\12月\门口\12_M2_pressure(UTC).csv")
    df_rain.to_csv(r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\12月\门口\12_M2_rain(UTC).csv")
    df_temperatur.to_csv(r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\12月\门口\12_M2_temperature(UTC).csv")
    #df_wind.to_csv(        r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\12月\门口\12_M2_wind(UTC).csv")

#process_combine_csv_1(r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\气象观测\2过程数据\12月\门口\1min数据合并.csv")
def combine(filenames,D_or_V):
   if D_or_V == 'd':
       s = 'WD'
   else:
       s = 'WS'
   d ={}
   for i in filenames:
       d.update({os.path.basename(i).replace(s,'').replace('.DAT',''):np.genfromtxt(i,missing_values='',usemask=True,invalid_raise=False,filling_values=0.0,dtype=float)})
   return d


def select_t1_from_t2(csv_file1,csv_file2):
    to_select = pd.read_csv(csv_file1)
    df_with_item = pd.read_csv(csv_file2)
    def set_time_to_index(df):
        try:
            df['t'] = pd.to_datetime(df['t'])
            df = df.set_index('t')
        except:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        return df
    to_select = set_time_to_index(to_select)
    df_with_item = set_time_to_index(df_with_item)
    return to_select.loc[df_with_item.index]

print("*" * 10)
print("筛选时间")
print("*" * 10)
f2 = r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\1月\1月报表时间.csv"

f_c3 = r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\潮流观测\2过程数据\1月\C3\程序分层结果.csv"
#f_c7 = r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\潮流观测\2过程数据\12月\C7\12月数据【UTC】.csv"
#select_t1_from_t2(f_c3,f2).to_csv(r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\潮流观测\2过程数据\1月\C3\C3_报表内容.csv")

print('*' * 10)
print('OK')