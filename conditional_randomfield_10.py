import numpy as np

m1 = np.array([[0, 0], [0.3, 0.7]])
m2 = np.array([[0.3, 0.7], [0.7, 0.3]])
m3 = np.array([[0.5, 0.5], [0.6, 0.4]])
m4 = np.array([[0, 1], [0, 1]])


def getz():
    t = np.array(m1.dot(m2))
    t1 = np.array(t.dot(m3))
    t2 = np.array(t1.dot(m4))
    return t2


def position(y,goal):
    pos = np.array([])
    for k in range(np.size(y)):
        if (y[k] == goal):
            pos=np.append(pos,k)
    return pos
# def gety():
#     y=np.array([])
#     for i in range(2):
#         for (k,l) in range(2,2):
#             for (m,n) in range(2,2):
#
#                 y.append(m1[1][i]*m2[k][l]*m3[m][n])
#     return y


if __name__ == "__main__":
    y1 = m1[1][0] * m2[0][0] * m3[0][0]  # a01b11c11
    y2 = m1[1][0] * m2[0][0] * m3[0][1]  # a01b11c12
    y3 = m1[1][0] * m2[0][1] * m3[1][0]  # a01b12c21
    y4 = m1[1][0] * m2[0][1] * m3[1][1]  # a01b12c22
    y5 = m1[1][1] * m2[1][0] * m3[0][0]  # a02b21c11
    y6 = m1[1][1] * m2[1][0] * m3[0][1]  # a02b21c12
    y7 = m1[1][1] * m2[1][1] * m3[1][0]  # a02b22c21
    y8 = m1[1][1] * m2[1][1] * m3[0][1]  # a02b22c22
    (i, j) = np.shape(getz())
    z = 0
    z_x = getz()
    for k in range(i):
        for l in range(j):
            z += z_x[k][l]
    print(getz())
    y = np.array([y1 / z, y2 / z, y3 / z, y4 / z, y5 / z, y6 / z, y7 / z, y8 / z])
    print(y)
    print(max(y))
    print(position(y, max(y)))
    print("start=2,stop=2")
    print("概率最大序列为：2-2-1-1-2，2-2-1-2-2")

