import numpy as np
import torch
import logging
import math
from scipy.stats import norm as gaussian

class GMRegularizer():
    '''GM regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''
    def __init__(self, hyperpara=None, gm_num=None, pi_list=None, reg_lambda=None, uptfreq=None):
        self.a, self.b, self.gm_num = hyperpara[0], hyperpara[1], gm_num
        print ("init self.a, self.b, self.gm_num: ", self.a, self.b, self.gm_num)
        for i in range(gm_num):
            pi_list[i] = np.reshape(np.array(pi_list[i]), (1, gm_num))
        self.pi_list = pi_list 
        self.reg_lambda = np.reshape(np.array(reg_lambda), (1, gm_num))
        print ("init self.reg_lambda: ", self.reg_lambda)
        print ("init self.pi_list: ", self.pi_list)
        self.gmuptfreq, self.paramuptfreq = uptfreq[0], uptfreq[1]
        print ("init self.gmuptfreq, self.paramuptfreq: ", self.gmuptfreq, self.paramuptfreq)

    # calc the resposibilities for pj(wi)
    def calcResponsibilityList(self):
        self.responsibilitylist = []
        for i in range(self.gm_num):
            # responsibility normalized with pi
            responsibility = gaussian.pdf(self.w_array_ordered_list[i], loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi_list[i]
            # responsibility normalized with summation(denominator)
            # new array ??
            self.responsibilitylist.append(responsibility/(np.sum(responsibility, axis=1).reshape(self.w_array_ordered_list[i].shape)))
        for i in range(self.gm_num):
            print ("self.responsibilitylist[i] shape: ", self.responsibilitylist[i].shape)
    
    def update_GM_Prior_EM(self, name, step):
        # update reg_lambda
        reg_lambda_numerator = 2 * (self.a - 1)
        for i in range(self.gm_num):
            reg_lambda_numerator = reg_lambda_numerator + np.sum(self.responsibilitylist[i], axis=0)
        reg_lambda_denominator = 2 * self.b
        for i in range(self.gm_num):
            reg_lambda_denominator = reg_lambda_denominator + np.sum(self.responsibilitylist[i] * np.square(self.w_array_ordered_list[i]), axis=0)
        # self.reg_lambda = (2 * (self.a - 1) + np.sum(self.responsibility, axis=0)) / (2 * self.b + np.sum(self.responsibility * np.square(self.w_array), axis=0))
        self.reg_lambda = reg_lambda_numerator / reg_lambda_denominator
        if step % self.gmuptfreq == 0:
            print ("name: ", name)
            # print "np.sum(self.responsibility, axis=0): ", np.sum(self.responsibility, axis=0)
            print ("np.sum(self.responsibility, axis=0): ")
            # print "np.sum(self.responsibility * np.square(self.w[:-1]), axis=0): ", np.sum(self.responsibility * np.square(self.w_array), axis=0)
            print ("np.sum(self.responsibility * np.square(self.w[:-1]), axis=0): ")
            # print "division: ", np.sum(self.responsibility * np.square(self.w_array), axis=0) / np.sum(self.responsibility, axis=0)
            print ("division: ")

    def CalcOrdCorreIdx(self):
        correlation_abs_matrix = np.abs(self.correlation_moving_average)
        self.ordered_correlation_index_matrix = np.zeros(correlation_abs_matrix.shape, dtype=int)
        print ('correlation_abs_matrix: \n', correlation_abs_matrix)
        ## descend ...
        for i in range(self.ordered_correlation_index_matrix.shape[0]):
            self.ordered_correlation_index_matrix[i] = np.argsort(-correlation_abs_matrix[i]) # uisng a small matrix to test the whole process ...
        print ('self.ordered_correlation_index_matrix: ', self.ordered_correlation_index_matrix)
    
    def DivideParam(self):
        self.w_array_ordered = np.zeros(self.w_array.shape)
        for i in range(self.w_array_ordered.shape[0]):
            self.w_array_ordered[i] = self.w_array[i][self.ordered_correlation_index_matrix[i]]
        ## correct ???
        base = int (self.w_array.shape[1] / self.gm_num)
        print ('base: ', base)
        self.w_array_ordered_list = []
        for i in range(self.gm_num-1):
            self.w_array_ordered_list.append(self.w_array_ordered[:, (i*base):((i+1)*base)].reshape((-1,1)))
        self.w_array_ordered_list.append(self.w_array_ordered[:, ((self.gm_num-1)*base):].reshape((-1,1)))
        for i in range(self.gm_num):
            print ('self.w_array_ordered_list[i] shape: ', self.w_array_ordered_list[i].shape)
            print ('self.w_array_ordered_list[i]: ', self.w_array_ordered_list[i])

    def calcGMRegGrad(self):
        self.grad_array_ordered_list = []
        # gm_num happens to be group number
        for i in range(self.gm_num):
            self.grad_array_ordered_list.append(np.sum(self.responsibilitylist[i]*self.reg_lambda, axis=1).reshape(self.w_array_ordered_list[i].shape) * self.w_array_ordered_list[i])
        # reshape
        for i in range(self.gm_num):
            self.grad_array_ordered_list[i] = self.grad_array_ordered_list[i].reshape((self.w_array.shape[0], -1))
            print ('self.grad_array_ordered_list[i] shape: ', self.grad_array_ordered_list[i].shape)
        # concatenate 
        grad_array_ordered_matrix = self.grad_array_ordered_list[0]
        for i in range(1, self.gm_num):
            grad_array_ordered_matrix = np.concatenate((grad_array_ordered_matrix, self.grad_array_ordered_list[i]), axis=1)
        print ("grad_array_ordered_matrix shape: ", grad_array_ordered_matrix.shape)
        self.reg_grad_w = np.zeros(self.w_array.shape)
        print ("self.reg_grad_w shape: ", self.reg_grad_w.shape)
        for i in range(self.reg_grad_w.shape[0]):
            self.reg_grad_w[i][self.ordered_correlation_index_matrix[i]] = grad_array_ordered_matrix[i]


    def apply(self, correlation_moving_average, trainnum, epoch, param, name, step):
        self.w_array = param
        self.correlation_moving_average = correlation_moving_average
        print ('self.w_array shape: ', self.w_array.shape)
        print ('self.correlation_moving_average shape: ', self.correlation_moving_average.shape)
        self.CalcOrdCorreIdx()
        print('after self.ordered_correlation_index_matrix shape: ', self.ordered_correlation_index_matrix.shape)
        print('after self.ordered_correlation_index_matrix: ', self.ordered_correlation_index_matrix)
        self.DivideParam() # divide parameters into different groups according to correlation
        # if epoch < 2 or step % self.paramuptfreq == 0:
        self.calcResponsibilityList()
        self.calcGMRegGrad()
        labelnum = 1
        self.reg_grad_w = self.reg_grad_w / float(trainnum * labelnum)
        if (epoch == 0 and step < 50) or step % self.gmuptfreq == 0:
            print ("step: ", step)
            print ("name: ", name)
            print ("data grad l2 norm: ", np.linalg.norm(param.grad.data.cpu().numpy()))
            print ("reg_grad_w_dev l2 norm: ", np.linalg.norm(self.reg_grad_w))
        if (epoch == 0 and step < 50) or step % self.gmuptfreq == 0:
            print ("w norm: ", np.linalg.norm(param.data.cpu().numpy()))
        if epoch < 2 or step % self.gmuptfreq == 0:
            if epoch >=2 and step % self.paramuptfreq != 0:
                self.calcResponsibilityList()
            self.update_GM_Prior_EM(name, step)
        return self.reg_grad_w


if __name__ == '__main__':
    pi_1 = np.array([0.1, 0.3, 0.6])
    pi_2 = np.array([0.2, 0.2, 0.6])
    pi_3 = np.array([0.1, 0.1, 0.8])
    pi_list = []
    pi_list.append(pi_1)
    pi_list.append(pi_2)
    pi_list.append(pi_3)
    gm_regularizer_instance = GMRegularizer(hyperpara=[1., 2.], gm_num=3, pi_list=pi_list, reg_lambda=[1, 2, 5], uptfreq=[50, 100])
    correlation_moving_average = np.array([[-0.2, 0.3, 0.1, -0.4, 0.6, -0.7, 0.5],
                                           [0.9, -1.1, 1.2, -0.8, -1.4, 1.0, 1.3],
                                           [1.5, 2.1, 1.7, 1.8, -1.6, -1.9, 2.0],
                                           [-2.4, 2.2, -2.8, 2.6, 2.5, -2.7, 2.3]])
    param = np.array([[0.1, -0.2, -0.3, 0.4, -0.5, 0.6, -0.7],
                      [0.8, -0.9, -1.0, 1.1, -1.2, 1.3, 1.4],
                      [1.5, -1.6, 1.7, 1.8, 1.9, 2.0, -2.1],
                      [-2.2, 2.3, -2.4, -2.5, -2.6, 2.7, 2.8]])
    print ('correlation_moving_average: ', correlation_moving_average)
    print ('param: ', param)
    reg_grad_w = gm_regularizer_instance.apply(correlation_moving_average, 100, 1, param, 'test', 10)
