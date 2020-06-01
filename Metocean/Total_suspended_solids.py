# -*- coding: <utf-8> -*-
# Created by lpf-THEIDI @ 2020/5/20 21:59
import pandas as pd
from Metocean.current import Single_Tide_Point, Current_pre_process, Read_Report
import numpy as np


def copy():
    df = pd.read_clipboard()
    index_name = df.columns[0]
    df[index_name] = pd.to_datetime(df[index_name])
    return df.set_index(index_name)


def read_clipboard_df(name_of_df):
    str = input("请复制含沙量Excel数据并按回车录入数据：")
    if str == '':
        df = copy()
        print("读入数据完成，储存在" + name_of_df + "中")
        return df
    if str != "EXIT":
        print(str)
        print("按回车继续输入Dataframe数据至" + name_of_df)
        read_clipboard_df(name_of_df)


if __name__ == "__main__":
    c = {}
    # for i in range(1,8):
    #     Point_name = "M" + str(i)
    #     c.update({Point_name:{}})
    #     for tide_type in ['大潮','中潮','小潮']:
    #         FileName = Point_name +" "+ tide_type
    #         dataframe = read_clipboard_df(FileName)
    #         c[Point_name].update({tide_type:dataframe})
    # print("数据读取结束，共录入" + str(len(c)) + "组数据")

    C = Read_Report(r"C:\2020汕尾渔港水文测验GK-2020-0030水\实测数据\附录B：潮流观测报表 - 副本.xlsx")
    C.setPoint_ang('M1', ang=250)
    C.setPoint_ang('M2', ang=250)
    C.setPoint_ang('M3', ang=300)
    C.setPoint_ang('M4', ang=300)
    C.setPoint_ang('M5', ang=220)
    C.setPoint_ang('M6', ang=180)
    C.setPoint_ang('M7', ang=180)

# read_clipboard_df()
