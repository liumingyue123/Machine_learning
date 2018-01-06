import numpy as np
import math
Ann = np.array([[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]])#状态转移矩阵
Bnm = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])#观测概率矩阵
Pi = np.array([0.2, 0.3, 0.5])#初始状态概率向量
O = np.array([0, 1, 0,0,1,0,1,1])#观测序列，0表示红，1表示白
import numpy as np


class HMM:
    def __init__(self, An, Bn, pi, O):
        self.A = np.array(An, np.float)
        self.B = np.array(Bn, np.float)
        self.Pi = np.array(pi, np.float)
        self.O = np.array(O, np.int)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]

    def forward(self):  # 前向计算
        T = len(self.O)
        alpha = np.zeros((T, self.N), np.float)

        for i in range(self.N):
            alpha[0][i] = self.Pi[i] * self.B[i][self.O[0]]

        for t in range(T - 1):
            for i in range(self.N):
                summation = 0
                for j in range(self.N):
                    summation += alpha[t][j] * self.A[j][i]#观测序列概率
                alpha[t + 1][i] = summation * self.B[i][self.O[t + 1]]

        summation = 0.0
        for i in range(self.N):
            summation += alpha[T - 1][i]
        Polambda = summation
        return Polambda, alpha

    def backward(self):#后向计算
        T = len(self.O)
        beta = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            beta[T - 1, i] = 1.0

        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                summation = 0.0
                for j in range(self.N):
                    summation += self.A[i][j] * self.B[j][self.O[t + 1]] * beta[t + 1][j]
                beta[t][i] = summation

        Polambda = 0.0
        for i in range(self.N):
            Polambda += self.Pi[i] * self.B[i, self.O[0]] * beta[0, i]
        return Polambda, beta


if __name__ == "__main__":
    hmm = HMM(Ann, Bnm, Pi, O)
    sequence_p1=hmm.forward()[0]
    polamda1=(hmm.forward()[1])
    a=polamda1[3][0]
    polamda2 = (hmm.backward()[1])
    b = polamda2[3][0]
    p=a*b/sequence_p1#前后向概率P(i4=q1|O,lamuda)
    print(p)
    print(hmm.forward()[0])
    print(hmm.forward()[1])
    print(hmm.backward()[0])
    print(hmm.backward()[1])
