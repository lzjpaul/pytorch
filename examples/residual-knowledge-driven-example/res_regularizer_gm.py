import numpy as np
import torch
import logging
import math
from scipy.stats import norm as gaussian

################
#(1) bases for initializing gm need to be changed according to model and fan_in
#(2) two classes commonly use self.w_array, self.reg_lambda, self.reg_grad_w
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
        # for i in range(self.gm_num):
        #     print ("self.responsibilitylist[i] shape: ", self.responsibilitylist[i].shape)
    
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
        self.reg_lambda = self.reg_lambda.reshape((1, self.gm_num))
        if step % self.gmuptfreq == 0:
            print ("name: ", name)
            # print "np.sum(self.responsibility, axis=0): ", np.sum(self.responsibility, axis=0)
            for i in range(self.gm_num):
                print ("np.sum(self.responsibilitylist[i], axis=0): ", np.sum(self.responsibilitylist[i], axis=0))
                print ("np.sum(self.responsibilitylist[i] * np.square(self.w_array_ordered_list[i]), axis=0): ", np.sum(self.responsibilitylist[i] * np.square(self.w_array_ordered_list[i]), axis=0))
                print ("division: ", np.sum(self.responsibilitylist[i] * np.square(self.w_array_ordered_list[i]), axis=0) / np.sum(self.responsibilitylist[i], axis=0))
            print ('self.reg_lambda: ', self.reg_lambda)
            print ("self.pi_list: ", self.pi_list)

    def CalcOrdCorreIdx(self):
        correlation_abs_matrix = np.abs(self.correlation_moving_average)
        self.ordered_correlation_index_matrix = np.zeros(correlation_abs_matrix.shape, dtype=int)
        ## descend ...
        for i in range(self.ordered_correlation_index_matrix.shape[0]):
            self.ordered_correlation_index_matrix[i] = np.argsort(-correlation_abs_matrix[i]) # uisng a small matrix to test the whole process ...
    
    def DivideParam(self):
        self.w_array_ordered = np.zeros(self.w_array.shape)
        for i in range(self.w_array_ordered.shape[0]):
            self.w_array_ordered[i] = self.w_array[i][self.ordered_correlation_index_matrix[i]]
        ## correct ???
        base = int (self.w_array.shape[1] / self.gm_num)
        # print ('base: ', base)
        self.w_array_ordered_list = []
        for i in range(self.gm_num-1):
            self.w_array_ordered_list.append(self.w_array_ordered[:, (i*base):((i+1)*base)].reshape((-1,1)))
        self.w_array_ordered_list.append(self.w_array_ordered[:, ((self.gm_num-1)*base):].reshape((-1,1)))
        # for i in range(self.gm_num):
        #     print ('self.w_array_ordered_list[i] shape: ', self.w_array_ordered_list[i].shape)

    def calcGMRegGrad(self):
        self.grad_array_ordered_list = []
        # gm_num happens to be group number
        # print ('len(self.responsibilitylist: )', len(self.responsibilitylist))
        for i in range(self.gm_num):
            # print ('calcGMRegGrad self.reg_lambda.shape: ', self.reg_lambda.shape)
            self.grad_array_ordered_list.append(np.sum(self.responsibilitylist[i]*self.reg_lambda, axis=1).reshape(self.w_array_ordered_list[i].shape) * self.w_array_ordered_list[i])
        # reshape
        for i in range(self.gm_num):
            self.grad_array_ordered_list[i] = self.grad_array_ordered_list[i].reshape((self.w_array.shape[0], -1))
            # print ('self.grad_array_ordered_list[i] shape: ', self.grad_array_ordered_list[i].shape)
        # concatenate 
        grad_array_ordered_matrix = self.grad_array_ordered_list[0]
        for i in range(1, self.gm_num):
            grad_array_ordered_matrix = np.concatenate((grad_array_ordered_matrix, self.grad_array_ordered_list[i]), axis=1)
        # print ("grad_array_ordered_matrix shape: ", grad_array_ordered_matrix.shape)
        self.reg_grad_w = np.zeros(self.w_array.shape)
        # print ("self.reg_grad_w shape: ", self.reg_grad_w.shape)
        for i in range(self.reg_grad_w.shape[0]):
            self.reg_grad_w[i][self.ordered_correlation_index_matrix[i]] = grad_array_ordered_matrix[i]


    def apply(self, correlation_moving_average, labelnum, trainnum, epoch, param, name, step):
        # print ("calling regularizers of name: ", name)
        self.w_array = param.data.cpu().numpy()
        self.correlation_moving_average = correlation_moving_average
        # print ('self.w_array shape: ', self.w_array.shape)
        # print ('self.correlation_moving_average shape: ', self.correlation_moving_average.shape)
        self.CalcOrdCorreIdx()
        # print('self.ordered_correlation_index_matrix shape: ', self.ordered_correlation_index_matrix.shape)
        self.DivideParam() # divide parameters into different groups according to correlation
        # for i in range(self.gm_num):
        #     print ('after self.w_array_ordered_list[i] shape: ', self.w_array_ordered_list[i].shape)
        if epoch < 2 or step % self.paramuptfreq == 0:
            self.calcResponsibilityList()
            self.calcGMRegGrad()
        # print ("labelnum: ", labelnum)
        self.reg_grad_w = self.reg_grad_w / float(labelnum * trainnum)
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

class GMResRegularizer():
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
        self.gmregularizers = {}
    
    def layer_wise_hyperpara(self, fea_num, hyperpara_list, hyperpara_idx):
        print ("layer_wise fea_num: ", fea_num)
        a_list = hyperpara_list[0]
        b_list = hyperpara_list[1]
        b_val = (b_list[hyperpara_idx[1]]) * fea_num
        a_val = 1. + (b_val * a_list[hyperpara_idx[0]])
        print ("b_val: ", b_val)
        print ("a+val: ", a_val)
        return [a_val, b_val]

    def gen_pi_list(self, gm_num, pi_decay_ratio):
        index_array = np.zeros(101) # [-50, -49, ..., 0, -1, -2, ..., -50]
        for i in range(51):
            index_array[i] = -50 + i
        for i in range(51):
            index_array[50 + i] = -i
        print('index_array: ', index_array)
        pi_list = []
        for i in range(gm_num):
            pi_list.append(index_array[(50-i):(50+gm_num-i)]) # [[0, -1, -2, -3], [-1, 0, -1, -2], [-2, -1, 0, -1], [-3, -2, -1, 0]]
        # decay and normalize 
        for i in range(gm_num):
            pi_list[i] = np.exp(pi_list[i] * pi_decay_ratio)
            pi_list[i] = pi_list[i] / float(np.sum(pi_list[i]))            
        return pi_list

    def gm_register(self, name, param, model_name, hyperpara_list, hyperpara_idx, gm_num, pi_decay_ratio, gm_lambda_ratio, uptfreq):
        print ("param name: ", name)
        print ("param shape: ", param.data.size())
        dims = param.data.size()[0] * param.data.size()[1]
        print ("dims: ", dims)
        layer_hyperpara = self.layer_wise_hyperpara(dims, hyperpara_list, hyperpara_idx) # layerwise initialization of hyper-params
        pi_list = self.gen_pi_list(gm_num, pi_decay_ratio)
        print ("pi_list: ", pi_list)
        k = 1.0 + gm_lambda_ratio
        print ("gm_lambda_ratio: ", gm_lambda_ratio)
        # calculate base, calculate fan_in and fan_out for MLP
        # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#calculate_gain
        if 'mlp' in model_name:
            base = 333.3333 / 10.0
            print ('base model name: ', model_name)
            print ("base: ", base)
        else:
            print ("not mlp, need new fan_in functions")
        # calculate GM initialized lambda (1/variance)
        if gm_lambda_ratio >= 0.0:
            gm_reg_lambda = [base*math.pow(k,_) for _ in  range(gm_num)]
        else:
            gm_reg_lambda_range = base * float(gm_num)
            gm_reg_lambda = np.arange(1.0, gm_reg_lambda_range, gm_reg_lambda_range/gm_num)
        self.gmregularizers[name] = GMRegularizer(hyperpara=layer_hyperpara, gm_num=gm_num, pi_list=pi_list, reg_lambda=gm_reg_lambda, uptfreq=uptfreq)


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
    def calcRegGradAvg_Gen_Prob(self):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.correlation_moving_average)
        correlation_abs_matrix_sum = np.sum(correlation_abs_matrix, axis=1).reshape((correlation_abs_matrix.shape[0],1))
        # print ('correlation_abs_matrix_sum shape:', correlation_abs_matrix_sum.shape)
        correlation_abs_matrix_normalize = correlation_abs_matrix / correlation_abs_matrix_sum
        # print ('correlation_abs_matrix_normalize shape: ', correlation_abs_matrix_normalize.shape)
        # print ('np.sum(correlation_abs_matrix_normalize, axis=1)', np.sum(correlation_abs_matrix_normalize, axis=1))
        correlation_abs_matrix_normalize_log = np.log(correlation_abs_matrix_normalize)
        # print ('np.min(correlation_abs_matrix_normalize) after log', np.min(correlation_abs_matrix_normalize_log))
        reg_grad_w = -self.reg_lambda * np.sign(self.w_array) * correlation_abs_matrix_normalize_log
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    def apply(self, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, trainnum, epoch, param, name, step):
        # logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
        # print ('trainnum: ', trainnum)
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
            self.reg_grad_w = self.calcRegGradAvg_Gen_Prob()
        # gm-based method
        elif reg_method == 6:
            print ('the correlation_moving_average is not layer-wise!!!!')
            self.reg_grad_w = self.gmregularizers[name].apply(self.correlation_moving_average, labelnum, trainnum, epoch, param, name, step)
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
