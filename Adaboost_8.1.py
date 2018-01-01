# from numpy import *
# num=10
# x=array([0,1,2,3,4,5,6,7,8,9])
# label=[1,1,1,-1,-1,-1,1,1,1,-1]
#
#
# def init_D():
#     D=[]
#     for i in range(len(x)):
#         D[i]=1/len(x)
#     return D
# def findmax(v):#确定G(x)函数
#     max=0
#     for i in range(int(v)):
#         if label[i]
#
#
#
#
#
# def getv(D):#得到错误率最小的阈值
#     error=0
#     range
#     for i in range(len(x)):
#         v=(2*i+1)/2
#         for j in range(v):
#             if label[j]!=
#
# if __name__=='__main__':
#
from numpy import *


def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


datMat, classLabels = loadSimpData()

# ========================================================================
'''弱分类器之单层决策树(decision stump,也称决策树桩),仅基于单个特征来做决策。  
'''
'''通过阈值比较对数据进行了分类'''


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


'''它会在一个加权数据集中循环，并找到具有最低错误率的单层决策树。伪代码如下:  
将最小错误率minError设为+00  
对数据集中的每一个特征(第一层循环)：  
    对每个步长(第二层循环)：  
        对每个不等号(第三层循环)：  
            建立一棵单层决策树并利用加权数据集对它进行测试  
            如果错误率低于minError，则将当前单层决策树设为最佳单层决策树  
返回最佳单层决策树  
'''


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T;
    m, n = shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity
    for i in range(n):  # loop over all dimensions
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # calc total error multiplied by D
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# D是一个概率分布向量，因此其所有的元素之和为1.0。一开始的所有元素都会被初始化成1/m
D = mat(ones((5, 1)) / 5)

bestStump, minError, bestClasEst = buildStump(datMat, classLabels, D)

'''基于单层决策树的AdaBoost训练过程,伪代码如下:  
对每次迭代：  
    利用buildStump()函数找到最佳的单层决策树  
    将最佳单层决策树加入到单层决策树数组  
    计算alpha  
    计算新的权重向量D  
    更新累计类别估计值  
    如果错误率等于0.0,则退出循环  
'''


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # init D to all equal
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        print
        "D:", D.T
        alpha = float(0.5 * log(
            (1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        print
        "classEst: ", classEst.T
        # 为下一次迭代计算D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()

        # 错误率累加计算
        aggClassEst += alpha * classEst
        print
        "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)

        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


weakClassArr, aggClassEst = adaBoostTrainDS(datMat, classLabels, 9)

'''每个弱分类器的结果以其对应的alpha值作为权重。所有这些弱分类器的结果加权求和就得到了最后的结果。  
datToClass：由一个或者多个待分类样例  
classifierArr：多个弱分类器组成的数组  
predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#c  
'''


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print
        aggClassEst
    return sign(aggClassEst)


'''随着迭代的进行，数据点[0,0]的分类结果越来越强'''
print
'----------------------------'
print
adaClassify([0, 0], weakClassArr)
print
'----------------------------'
print
adaClassify([[5, 5], [0, 0]], weakClassArr)


