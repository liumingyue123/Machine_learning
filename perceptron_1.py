from numpy import *
import numpy as np

data = [
    {'x':np.array([0, 0, 0]), 'y': 1},
    {'x':np.array([1, 0, 0]), 'y': 1},
    {'x':np.array([1, 0, 1]), 'y': 1},
    {'x':np.array([1, 1, 0]), 'y': 1},
    {'x':np.array([0, 0, 1]), 'y': -1},
    {'x':np.array([0, 1, 1]), 'y': -1},
    {'x':np.array([0, 1, 0]), 'y': -1},
    {'x':np.array([1, 1, 1]), 'y': -1},
]

learning_rate = 1

def check(data, w=np.array([0, 0, 0]), b=0):
    for each in data:
        x, y = each['x'], each['y']
        result=(w[0]*x[0]+w[1]*x[1]+w[2]*x[2]+b)*y
        if result<=0:
            return check(data,w=w+learning_rate*y*x,b=b+learning_rate*y)
    return w,b

if __name__ == '__main__':
    print(check(data))
