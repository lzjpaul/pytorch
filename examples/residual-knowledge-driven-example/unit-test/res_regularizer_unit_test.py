import numpy as np
import torch

# calc the resposibilities for pj(wi)
def calcCorrelation(feature_matrix):
    feature_correlation = np.corrcoef(feature_matrix, rowvar=False)
    print ("slef.feature_correlation.shape: ", feature_correlation.shape)
    print ("slef.feature_correlation:\n ", feature_correlation)
    return feature_correlation

# singa: (word_num, doc_num)
# pytorch: (doc_num, word_num)
def calcRegGrad(reg_lambda, feature_correlation, w_array):
    correlation_abs_matrix = np.abs(feature_correlation)
    print ('correlation_abs_matrix:\n ', correlation_abs_matrix)
    correlation_abs_avg = np.mean(correlation_abs_matrix)
    print ('correlation_abs_avg: ', correlation_abs_avg) 
    correlation_diff_matrix = correlation_abs_avg - correlation_abs_matrix
    print ('correlation_diff_matrix:\n ', correlation_diff_matrix)
    print ('np.exp(correlation_diff_matrix):\n', np.exp(correlation_diff_matrix))
    print ('2 * reg_lambda * np.exp(correlation_diff_matrix):\n', 2 * reg_lambda * np.exp(correlation_diff_matrix))
    reg_grad_w = 2 * reg_lambda * np.exp(correlation_diff_matrix) * w_array
    print ('w_array:\n ', w_array)
    print ('(2 * reg_lambda * np.exp(correlation_diff_matrix)).dot(w_array):\n ', (2 * reg_lambda * np.exp(correlation_diff_matrix)).dot(w_array))
    print ("reg_grad_w:\n ", reg_grad_w)
    return reg_grad_w

if __name__ == '__main__':
    feature_matrix = np.array([[ 58295.62187335,  45420.95483714,   3398.64920064,    977.22166306, 5515.32801851,  14184.57621022,  16027.2803392 ,  15313.01865824, 6443.2448182 ], [ -143547.79123381,   -22996.69597427,    -2591.56411049, -661.93115277,    -8826.96549102,   -17735.13549851, -11629.13003263,   -14438.33177173,    -6997.89334741], [ 143547.79123381,   22996.69597427,    2591.56411049, 661.93115277,    8826.96549102,   17735.13549851, 11629.13003263,   14438.33177173,    6997.89334741], [ -58295.62187335,  -45420.95483714,   -3398.64920064,    -977.22166306, -5515.32801851,  -14184.57621022,  -16027.2803392 ,  -15313.01865824, -6443.2448182 ], [ 143547.79123381,   22996.69597427,    2591.56411049, 661.93115277,    8826.96549102,   17735.13549851, 11629.13003263,   14438.33177173,    6997.89334741]])
    feature_matrix = np.transpose(feature_matrix)
    print ("self.feature_matrix shape: ", feature_matrix.shape)
    reg_lambda = 0.1
    print ("self.reg_lambda: ", reg_lambda)
    w_array = np.arange(25).reshape((5, 5))
    w_array = np.array([[1,1,1,1,1], [-1,-1,-1,-1,-1], [1,1,1,1,1], [-1,-1,-1,-1,-1], [1,1,1,1,1]])
    print ("self.w_array shape: ", w_array.shape)
    print ("self.w_array:\n ", w_array)
    feature_correlation = calcCorrelation(feature_matrix)
    # self.reg_grad_w = np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w_array.shape) * self.w_array
    reg_grad_w = calcRegGrad(reg_lambda, feature_correlation, w_array)
    # reg_grad_w_dev = (torch.from_numpy(self.reg_grad_w)).float()
    # if (epoch == 0 and step < 50) or step % self.resuptfreq == 0:
    # print ("step: ", step)
    # print ("name: ", name)
    # print ("data grad l2 norm: ", np.linalg.norm(param.grad.data.cpu().numpy()))
    # print ("reg_grad_w_dev l2 norm: ", np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
    # if gpu_id >= 0:
    #     param.grad.data.add_(1.0, reg_grad_w_dev.cuda(gpu_id)) # here3
    # else:
    #     param.grad.data.add_(1.0, reg_grad_w_dev) # here3
    # if (epoch == 0 and step < 50) or step % self.ldauptfreq == 0:
    # print ("delta w (data grad + reg grad) norm: ", np.linalg.norm(param.grad.data.cpu().numpy()))
    # print ("w norm: ", np.linalg.norm(param.data.cpu().numpy()))
