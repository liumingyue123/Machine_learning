from numpy import *
import numpy as np

x = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]]
y = [1, 1, 1, 1, -1, -1, -1, -1]

data = [
    {'x': [0, 0, 0], 'y': 1},
    {'x': [1, 0, 0], 'y': 1},
    {'x': [1, 0, 1], 'y': 1},
    {'x': [1, 1, 0], 'y': 1},
    {'x': [0, 0, 1], 'y': -1},
    {'x': [0, 1, 1], 'y': -1},
    {'x': [0, 1, 0], 'y': -1},
    {'x': [1, 1, 1], 'y': -1},
]

learning_rate = 1


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def check(data, w=[0, 0, 0], b=0):
    for each in data:
        x, y = each['x'], each['y']
        result=(w[0]*x[0]+w[1]*x[1]+w[2]*x[2])*y
        if result<=0:
            check(data,w=w+learning_rate*)


if __name__ == '__main__':
    check(data, 0, 0)