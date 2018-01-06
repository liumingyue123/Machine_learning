# -*- coding: utf-8 -*-
"""Excercise 6.2"""
import numpy as np
from libsvm.python import svm, svmutil

import os

dir = os.path.dirname(__file__)
abspath = os.path.join(dir, 'data.txt')
prob = svmutil.svm_read_problem(abspath)
print('Linear\n')
svmutil.svm_train(prob[0], prob[1], '-t 0')
print('RBF\n')
svmutil.svm_train(prob[0], prob[1], '-t 2')
