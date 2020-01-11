#encoding = utf-8
import pandas as pd


def density(S, T, P):  # 根据盐度(PSU)、温度(摄氏度)、压强(MPa) 推算对应密度
    # 对应公式在GBT127.63.7 - 2007 海洋调查规范第七部分 海洋调查资料交换 107页
    # 返回密度单位为kg/m^3
    a0 = 999.842594
    a1 = 6.793952E-2
    a2 = -9.095290E-3
    a3 = 1.001685E-4
    a4 = -1.120083E-6
    a5 = 6.536332E-9
    b0 = 8.24493E-1
    b1 = -4.0899E-3
    b2 = 7.6438E-5
    b3 = -8.2467E-7
    b4 = 5.3875E-9
    c0 = -5.72466E-3
    c1 = 1.0227E-4
    c2 = -1.6546E-6
    d0 = 4.8314E-4

    rho_w = a0 + a1 * T + a2 * T ** 2 + a3 * T ** 3 + a4 * T ** 4 + a5 * T ** 5
    rho_S_t_0 = rho_w + (b0 + b1 * T + b2 * T ** 2 + b3 * T ** 3 + b4 * T ** 4) * \
        S + (c0 + c1 * T + c2 * T ** 2) * S ** (3 / 2) + d0 * S ** 2

    e0 = 19652.21
    e1 = 148.4206
    e2 = -2.327105
    e3 = 1.360477E-2
    e4 = -5.155288E-5

    f0 = 54.6746
    f1 = -0.603459
    f2 = 1.09987E-2
    f3 = -6.1670E-5

    g0 = 7.944E-2
    g1 = 1.6483E-2
    g2 = -5.3009E-4

    K_w = e0 + e1 * T + e2 * T ** 2 + e3 * T ** 3 + e4 * T ** 4
    K_S_t_0 = K_w + (f0 + f1 * T + f2 * T ** 2 + f3 * T ** 3) * \
        S + (g0 + g1 * T + g2 * T ** 2) * S ** (3 / 2)

    h0 = 32.39908
    h1 = 1.43713E-2
    h2 = 1.16092E-3
    h3 = -5.77905E-6

    i0 = 2.2838E-2
    i1 = -1.0981E-4
    i2 = -1.6078E-5
    j0 = 1.91075E-3

    A_w = h0 + h1 * T + h2 * T ** 2 + h3 * T ** 3
    A = A_w + (i0 + i1 * T + i2 * T ** 2) * S + j0 * S ** (3 / 2)

    k0 = 8.50935E-3
    k1 = -6.12293E-4
    k2 = 5.2787E-6
    m0 = -9.9348E-5
    m1 = 2.0816E-6
    m2 = 9.1697E-8

    B_w = k0 + k1 * T + k2 * T**2
    B = B_w + (m0 + m1 * T + m2 * T ** 2) * S

    K_S_t_p = K_S_t_0 + A * P + B * P ** 2  # 弹性模量

    return rho_S_t_0 / (1 - 10 * P / K_S_t_p)


filenames = [
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180616.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180617.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180618.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180619.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180620.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180621.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180622.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180623.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180624.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180625.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180626.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180627.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180628.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180629.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180630.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180601.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180602.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180603.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180604.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180605.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180606.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180607.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180609.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180610.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180611.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180613.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180614.xls",
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\1原始数据\18-6月份CTD\20180615.xls"]

c = [
    "Pressure",
    "Timestamp",
    "Temperature",
    "Dissolved O₂",
    "Turbidity",
    "Depth",
    "Salinity"]

cengshu = ['表层', '0.2H', '0.4H', '0.6H', '0.8H', '底层']

dfOUT = []
dfALL = []
for filename in filenames:
    df = pd.read_excel(filename, skiprows=5)[c]

    df['t'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    dfALL.append(df)

dfALL = pd.concat(dfALL).reset_index()
G = dfALL.groupby(dfALL['t'].apply(lambda x: x.date()))

print("数据读取完毕")
for date, df in G:
    g = df.groupby(df.Salinity > 1)
    try:
        df1 = df.loc[g.groups[True], :]
    except BaseException:
        print(str(date) + '没有可用数据')
        continue
    fitValue = df.loc[g.groups[False],
                      'Turbidity'][df.loc[g.groups[False],
                                          :]['Turbidity'] < 0].mean()
    df1.Turbidity -= fitValue
    print(str(date) + "浊度修正值为" + str(fitValue))

    l = g.groups[True].values
    stonetime = []
    for i in l:
        if (i - 1) not in l:
            stonetime.append(range(i - 5, i - 1))
            continue
        if (i + 1) not in l:
            stonetime.append(range(i + 1, i + 5))

    # if len(stonetime) != 4:
        #print(str(date) + "气压修正值有问题")
        # continue

    #fitPressure1 = df.loc[stonetime[0], 'Pressure'].mean() / 2 + df.loc[stonetime[1], 'Pressure'].mean() / 2
    #fitPressure2 = df.loc[stonetime[3], 'Pressure'].mean() / 2 + df.loc[stonetime[2], 'Pressure'].mean() / 2

    df = df1
    a1 = df.iloc[3, -1]
    g2 = df.groupby(df.t.apply(lambda x: abs(x.hour - a1.hour) > 2))
    [dfTemp, dfSalinity, dfTurbidity, dfDissO2] = [pd.DataFrame()] * 4
    dfT2 = pd.DataFrame()
    for _, j in g2:
        depthest = j.Depth.max()
        g5 = j.groupby(round(j.Depth / depthest / 0.2))

        # if  (j.index[0] > stonetime[0]).all() and (j.index[0] < stonetime[1]).all():#第一次入水的时间区间
        #    fitPressure = fitPressure1
        # else:
        #    fitPressure = fitPressure2
        for ceng, jj in g5:
            jj = jj.iloc[1:-1, :]
            O2 = jj["Dissolved O₂"].mean()
            Turbidity = jj['Turbidity'].mean()
            Pressure = jj['Pressure'].mean() / 100  # 压强换算

            while Turbidity < 0:
                Turbidity -= fitValue
            if O2 > 200:
                O2 /= 10
                if O2 < 100:
                    O2 += 100
            if ceng == 5:
                if Turbidity > 11:
                    Turbidity = jj['Turbidity'].max(
                    ) - (round(jj['Turbidity'].max()) - 10)
                    print(str([jj.iloc[[len(jj) // 2], -1]]) +
                          cengshu[int(ceng)] + "浊度太高，为" + str(jj['Turbidity'].mean()))
            else:
                if Turbidity > 3:
                    Turbidity = jj['Turbidity'].max(
                    ) - (round(jj['Turbidity'].max()) - 2)
                    print(str([jj.iloc[[len(jj) // 2], -1]]) +
                          cengshu[int(ceng)] + "浊度太高，为" + str(jj['Turbidity'].mean()))

            try:
                dfT1 = pd.DataFrame(data={'层数': cengshu[int(ceng)], '水深': depthest, '深度': jj['Depth'].mean(), '温度': jj['Temperature'].mean(
                ), '浊度': Turbidity, '溶解氧': O2, '盐度': jj['Salinity'].mean(), '水中压强': Pressure}, index=[jj.iloc[[len(jj) // 2], -1]])

                dfT2 = dfT2.append(dfT1, ignore_index=False)
            except BaseException:
                pass
    dfT2['密度'] = density(dfT2['盐度'], dfT2['温度'], dfT2['水中压强'])
    dfT2['日期'] = date
    dfT2['潮位'] = '高'
    dfT2.loc[dfT2['水深'] < dfT2['水深'].mean(), '潮位'] = '低'
    dfOUT.append(dfT2)
    print(str(date) + '处理完毕')
df = pd.concat(dfOUT).sort_index()
# 开始计算密度
# rho_(S_t_0)


df = df[['日期', '潮位', '水深', '层数', '深度', '温度', '浊度', '盐度', '溶解氧', "密度", "水中压强"]]
df.to_csv(r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\2过程文件\18-6\CTD6月（18）数据处理.csv")
excelWriter = pd.ExcelWriter(
    r"E:\★★★★★项目★★★★★\★★★★★吉布提2017★★★★★\2017吉布提\1长期观测\CTD观测\2过程文件\18-6\分项-1.xlsx")
for i in ['温度', '浊度', '盐度', '溶解氧', "水中压强", "密度"]:

    dft = pd.pivot_table(
        df,
        values=[i],
        index=[
            '日期',
            '水深',
            '潮位'],
        columns=['层数']).reindex_axis(
            cengshu,
            axis=1,
        level=1)
    dft.to_excel(excelWriter, i)
excelWriter.save()

print("*" * 10)
print('全部搞定')
print("*" * 10)
