# -*- coding:utf-8 -*-
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq  投掷骰子，理论参见概率图模型项目相关笔记

import numpy as np
import hmm

# 骰子种类： 4面，6面，8面
dice_num = 3
# 观测变量取值的可能性
x_num = 8
dice_hmm = hmm.DiscreteHMM(3, 8)
# 隐状态的初始概率
dice_hmm.start_prob = np.ones(dice_num) / dice_num
# 状态转移矩阵
dice_hmm.transmat_prob = np.ones((dice_num, dice_num)) / dice_num
# 发射状态矩阵
dice_hmm.emission_prob = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
# 归一化
dice_hmm.emission_prob = dice_hmm.emission_prob / np.repeat(np.sum(dice_hmm.emission_prob, 1), 8).reshape((3, 8))

dice_hmm.trained = True

# 观测值
X = np.array([[1],[6],[3],[5],[2],[7],[3],[5],[2],[4],[3],[6],[1],[5],[4]])

# 解码
# 问题A
Z = dice_hmm.decode(X)
# 问题B
logprob = dice_hmm.X_prob(X)

# 问题C
x_next = np.zeros((x_num, dice_num))
for i in range(x_num):
    c = np.array([i])
    x_next[i] = dice_hmm.predict(X, i)

print("state: ", Z)
print("logprob: ", logprob)
print("prob of x_next: ", x_next)

