import numpy as np


Ann = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])  # 状态转移矩阵
Bnm = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])  # 观测概率矩阵
Pi = np.array([0.2, 0.4, 0.4])  # 初始状态概率向量
O = np.array([0,1,0,1])  # 观测序列，0表示红，1表示白
import numpy as np


class HMM:
    def __init__(self, An, Bn, pi, O):
        self.A = np.array(An, np.float)
        self.B = np.array(Bn, np.float)
        self.Pi = np.array(pi, np.float)
        self.O = np.array(O, np.int)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]

    def viterbi(self):
        # given O,lambda .finding I

        T = len(self.O)
        I = np.zeros(T, np.int)

        delta = np.zeros((T, self.N), np.float)
        psi = np.zeros((T, self.N), np.float)

        for i in range(self.N):#delta1
            delta[0, i] = self.Pi[i] * self.B[i, self.O[0]]
            psi[0, i] = 0

        for t in range(1, T):
            for i in range(self.N):
                delta[t][i] = self.B[i][self.O[t]]* np.array([delta[t - 1][ j] * self.A[j][ i] for j in range(self.N)]).max()
                psi[t][i] = np.array([delta[t - 1][ j] * self.A[j][ i] for j in range(self.N)]).argmax()
        P_T = delta[T - 1, :].max()
        I[T - 1] = delta[T - 1, :].argmax()

        for t in range(T - 2, -1, -1):
            I[t] = psi[t + 1][ I[t + 1]]

        return I,delta
if __name__=="__main__":
    hmm=HMM(Ann,Bnm,Pi,O)
    result=hmm.viterbi()
    print(result[0])#最优路径要在此基础上加一，因为是从0开始
    print(result[1])#3-2-2-2