import math
[pi,p,q]=[0.46,0.55,0.67]
y=[1,1,0,1,0,0,1,0,1,1]

def get_u(pi1,p1,q1,yj):#E步准备
    return pi1*math.pow(p1,yj)*math.pow((1-p1),1-yj)/(pi1*math.pow(p1,yj)*math.pow((1-p1),1-yj)+(1-pi1)*math.pow(q1,yj)*math.pow(1-q1,1-yj))

def estep(pi1,p1,q1,y):#E步
    return [get_u(pi1,p1,q1,y[i]) for i in range(len(y))]

def mstep(u,y):#M步
    pi1=sum(u)/len(y)
    p1 = sum([u[i] * y[i] for i in range(len(u))]) / sum(u)
    q1 = sum([(1 - u[i]) * y[i] for i in range(len(u))]) / sum([1 - u[i] for i in range(len(u))])
    return [pi1, p1, q1]


def run( start_pi, start_p, start_q, iter_num):
    for i in range(iter_num):#迭代
        u= estep(start_pi, start_p, start_q, y)
        print(i, [start_pi, start_p, start_q])
        if [start_pi, start_p, start_q] == mstep(u, y):
            break
        else:
            [start_pi, start_p, start_q] = mstep(u, y)
if __name__=='__main__':
    run(pi, p, q, 100)



