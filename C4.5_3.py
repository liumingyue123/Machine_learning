import numpy as np
import math

data = [
    {'id': 1, 'age': 1, 'job': 0, 'house': 0, 'honest': 1, 'result': 0},
    {'id': 2, 'age': 1, 'job': 0, 'house': 0, 'honest': 2, 'result': 0},
    {'id': 3, 'age': 1, 'job': 1, 'house': 0, 'honest': 2, 'result': 1},
    {'id': 4, 'age': 1, 'job': 1, 'house': 1, 'honest': 1, 'result': 1},
    {'id': 5, 'age': 1, 'job': 0, 'house': 0, 'honest': 1, 'result': 0},
    {'id': 6, 'age': 2, 'job': 0, 'house': 0, 'honest': 1, 'result': 0},
    {'id': 7, 'age': 2, 'job': 0, 'house': 0, 'honest': 2, 'result': 0},
    {'id': 8, 'age': 2, 'job': 1, 'house': 1, 'honest': 2, 'result': 1},
    {'id': 9, 'age': 2, 'job': 0, 'house': 1, 'honest': 3, 'result': 1},
    {'id': 10, 'age': 2, 'job': 0, 'house': 1, 'honest': 3, 'result': 1},
    {'id': 11, 'age': 3, 'job': 0, 'house': 1, 'honest': 3, 'result': 1},
    {'id': 12, 'age': 3, 'job': 0, 'house': 1, 'honest': 2, 'result': 1},
    {'id': 13, 'age': 3, 'job': 1, 'house': 0, 'honest': 2, 'result': 1},
    {'id': 14, 'age': 3, 'job': 1, 'house': 0, 'honest': 3, 'result': 1},
    {'id': 15, 'age': 3, 'job': 0, 'house': 0, 'honest': 1, 'result': 0},
]


def getH_D(data,ch, value):#计算经验熵和经验条件熵H(D|A)
    total = 0.0
    count = 0.0
    if ch == 'result':#H(D)
        total = len(data)
        for each in data:
            x = each[ch]
            if x == value:
                count += 1
    else:#H(D|A)
        for each in data:
            if each[ch] == value and each['result']:
                count += 1
            if each[ch] == value:
                total += 1
    return -count / total * math.log(count / total, 2) - ((total - count) / total) * math.log((total - count) / total,
                                                                                              2)


def getH_DA1(ch):
    for each in data:
        x = each['ch']
        y = each['result']


data1 = [
    {'age': 1, 'job': 0, 'house': 0, 'honest': 1, 'result': 0},
    {'age': 1, 'job': 0, 'house': 0, 'honest': 2, 'result': 0},
    {'age': 1, 'job': 1, 'house': 0, 'honest': 2, 'result': 1},
    {'age': 1, 'job': 0, 'house': 0, 'honest': 1, 'result': 0},
    {'age': 2, 'job': 0, 'house': 0, 'honest': 1, 'result': 0},
    {'age': 2, 'job': 0, 'house': 0, 'honest': 2, 'result': 0},
    {'age': 3, 'job': 1, 'house': 0, 'honest': 2, 'result': 1},
    {'age': 3, 'job': 1, 'house': 0, 'honest': 3, 'result': 1},
    {'age': 3, 'job': 0, 'house': 0, 'honest': 1, 'result': 0},

]


if __name__ == "__main__":
    gA1 = getH_D(data,'result', 1) - getH_D(data,'age', 1) / 3 - getH_D(data,'age', 2) / 3 - getH_D(data,'age', 3) / 3
    g_A1 = gA1 / (-5 / 15 * math.log(5 / 15, 2) - 5 / 15 * math.log(5 / 15, 2) - 5 / 15 * math.log(5 / 15, 2))
    gA2 = getH_D(data,'result', 1) - 2 * getH_D(data,'job', 0) / 3
    g_A2 = gA2 / (-5 / 15 * math.log(5 / 15, 2) - 10 / 15 * math.log(10 / 15, 2))
    gA3 = getH_D(data,'result', 1) - 9 * getH_D(data,'house', 0) / 15
    g_A3 = gA3 / (-6 / 15 * math.log(6 / 15, 2) - 9 / 15 * math.log(9 / 15, 2))
    gA4 = getH_D(data,'result', 1) - 5 * getH_D(data,'honest', 1) / 15 - 6 * getH_D(data,'honest', 2) / 15
    g_A4 = gA4 / (-5 / 15 * math.log(5 / 15, 2) - 6 / 15 * math.log(6 / 15, 2) - 4 / 15 * math.log(4 / 15, 2))
    print(g_A1, g_A2, g_A3, g_A4)
    print("第三个特点信息增益比最大")
    # 新的训练集下有无房子作为特征 有房子类别对应同意贷款，新集合为data1
    newgA1=getH_D(data1,'result',1)-4*getH_D(data1,'age', 1) / 9-3*getH_D(data1,'age', 3) / 9
    new_gA1=newgA1/(-4 / 9 * math.log(4 / 9, 2) - 2 /9  * math.log(2 / 9, 2) - 3 / 9 * math.log(3 / 9, 2))
    newgA2 = getH_D(data1, 'result', 1)
    new_gA2 = newgA2 / (-3 / 9 * math.log(3 / 9, 2) - 6 / 9 * math.log(6 / 9, 2) )
    newgA4 = getH_D(data1, 'result', 1) -  4 * getH_D(data1, 'honest', 2) / 9
    new_gA4 = newgA4 / (-1 / 9 * math.log(1 / 9, 2) - 4 / 9 * math.log(4 / 9, 2) - 4 / 9 * math.log(4 / 9, 2))
    print(new_gA1,new_gA2,new_gA4)
    print("新训练集下第二个特征增益比最大")
#          是否有自己的房子
#          /            \
#       有/             \无
#       是            是否有工作
#                      /      \
#                   有/       \无
#                   是          否

#存在的问题，在计算信息增益比时，log0无法计算，手动删掉log0的位置