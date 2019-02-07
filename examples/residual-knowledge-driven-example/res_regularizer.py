import numpy as np
import torch
import logging


class ResRegularizer():
    '''Res regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''
    def __init__(self, reg_lambda=None, momentum_mu=None, feature_dim=None):
        self.reg_lambda = reg_lambda
        self.momentum_mu = momentum_mu
        print ("self.reg_lambda: ", self.reg_lambda)
        print ("new self.momentum_mu: ", self.momentum_mu)
        self.correlation_moving_average = np.zeros((feature_dim, feature_dim))
        print ('new self.correlation_moving_average shape: ', self.correlation_moving_average.shape)
    
    # calc correlation using one layer 
    def calcCorrelation(self):
        logger = logging.getLogger('res_reg')
        self.feature_correlation = np.corrcoef(self.feature_matrix, rowvar=False)
        logger.debug ("slef.feature_correlation.shape:")
        logger.debug (self.feature_correlation.shape)

    # calc correlation using two layers 
    def calcCorrelation_two_layers(self):
        logger = logging.getLogger('res_reg')
        feature_dim = self.feature_matrix.shape[1]
        logger.debug ("new using two layers self.feature_dim: %d", feature_dim)
        self.feature_correlation = np.corrcoef(self.feature_matrix, self.second_feature_matrix, rowvar=False)[feature_dim:, 0:feature_dim]
        logger.debug ("new using two layers self.feature_correlation.shape:")
        logger.debug (self.feature_correlation.shape)

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    '''
    def calcRegGrad(self):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.feature_correlation)
        correlation_abs_avg = np.mean(correlation_abs_matrix)
        correlation_diff_matrix = correlation_abs_avg - correlation_abs_matrix
        reg_grad_w = 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * self.w_array
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w
     '''
     
    # calculate regularization using self.correlation_moving_average
    def calcRegGradAvg(self):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.correlation_moving_average) # only this line is different
        correlation_abs_avg = np.mean(correlation_abs_matrix)
        correlation_diff_matrix = correlation_abs_avg - correlation_abs_matrix
        reg_grad_w = 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * self.w_array
        logger.debug ("new reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Exp(self):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.correlation_moving_average)
        reg_grad_w = 2 * self.reg_lambda * np.exp(-correlation_abs_matrix) * self.w_array
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w
   
    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Linear(self):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.correlation_moving_average)
        reg_grad_w = 2 * self.reg_lambda * (1.0 - correlation_abs_matrix) * self.w_array
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w


    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Inverse(self):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.correlation_moving_average)
        reg_grad_w = 2 * self.reg_lambda * (1.0 / correlation_abs_matrix) * self.w_array
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Inverse_Var(self):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.correlation_moving_average)
        reg_grad_w = 2 * self.reg_lambda * (1.0 / (1.0 + correlation_abs_matrix)) * self.w_array
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Gen_Prob(self, labelnum, trainnum):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.correlation_moving_average)
        correlation_abs_matrix_sum = np.sum(correlation_abs_matrix, axis=1).reshape((correlation_abs_matrix.shape[0],1))
        # print ('correlation_abs_matrix_sum shape:', correlation_abs_matrix_sum.shape)
        correlation_abs_matrix_normalize = correlation_abs_matrix / correlation_abs_matrix_sum
        # print ('correlation_abs_matrix_normalize shape: ', correlation_abs_matrix_normalize.shape)
        # print ('np.sum(correlation_abs_matrix_normalize, axis=1)', np.sum(correlation_abs_matrix_normalize, axis=1))
        correlation_abs_matrix_normalize_log = np.log(correlation_abs_matrix_normalize)
        # print ('np.min(correlation_abs_matrix_normalize) after log', np.min(correlation_abs_matrix_normalize_log))
        reg_grad_w = (-self.reg_lambda * np.sign(self.w_array) * correlation_abs_matrix_normalize_log)/float(labelnum * trainnum)
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    def apply(self, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, trainnum, epoch, param, name, step):
        # logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
        logger = logging.getLogger('res_reg')
        self.feature_matrix = features[feature_idx].data.cpu().numpy()
        self.second_feature_matrix = features[feature_idx + 1].data.cpu().numpy()
        logger.debug ("feature_idx: %d", feature_idx)
        logger.debug ("self.feature_matrix shape:")
        logger.debug (self.feature_matrix.shape)
        logger.debug ("self.feature_matrix norm: %f", np.linalg.norm(self.feature_matrix))
        logger.debug ("new self.second_feature_matrix norm: %f", np.linalg.norm(self.second_feature_matrix))
        self.reg_lambda = reg_lambda
        logger.debug ("self.reg_lambda: %f", self.reg_lambda)
        self.w_array = param.data.cpu().numpy()
        logger.debug ("self.w_array shape: ")
        logger.debug (self.w_array.shape)
        # self.calcCorrelation()
        self.calcCorrelation_two_layers()
        self.correlation_moving_average = self.momentum_mu * self.correlation_moving_average + (1-self.momentum_mu) * self.feature_correlation
        # self.reg_grad_w = self.calcRegGrad()
        if reg_method == 0:
            self.reg_grad_w = self.calcRegGradAvg()
        elif reg_method == 1:
            self.reg_grad_w = self.calcRegGradAvg_Exp()
        elif reg_method == 2:
            self.reg_grad_w = self.calcRegGradAvg_Linear()
        elif reg_method == 3:
            self.reg_grad_w = self.calcRegGradAvg_Inverse()
        elif reg_method == 4:
            self.reg_grad_w = self.calcRegGradAvg_Inverse_Var()
        # generation probablity
        elif reg_method == 5:
            self.reg_grad_w = self.calcRegGradAvg_Gen_Prob(labelnum, trainnum)
        else:
            print("Invalid regularization method, exiting...")
            exit()
        reg_grad_w_dev = (torch.from_numpy(self.reg_grad_w)).float()
        if (epoch == 0 and step <= 100) or step % 100 == 0:
            print ('step: ', step)
            print ('name: ', name)
            print ('w norm: ', np.linalg.norm(param.data.cpu().numpy()))
            print ('data grad norm: ', np.linalg.norm(param.grad.data.cpu().numpy()))
            print ('reg_grad_w_dev l2 norm: ', np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        logger.debug ("step: %d", step)
        logger.debug ("name: " +  name)
        logger.debug ("data grad l2 norm: %f", np.linalg.norm(param.grad.data.cpu().numpy()))
        logger.debug ("reg_grad_w_dev l2 norm: %f", np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        if gpu_id >= 0:
            param.grad.data.add_(1.0, reg_grad_w_dev.cuda(gpu_id)) # here3
        else:
            param.grad.data.add_(1.0, reg_grad_w_dev) # here3
        if (epoch == 0 and step <= 100) or step % 100 == 0:
            print ('delta w (data grad + reg grad) norm: ', np.linalg.norm(param.grad.data.cpu().numpy()))
        logger.debug ("delta w (data grad + reg grad) norm: %f", np.linalg.norm(param.grad.data.cpu().numpy()))
        logger.debug ("w norm: %f", np.linalg.norm(param.data.cpu().numpy()))
