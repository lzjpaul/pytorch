import numpy as np
import torch
import logging


class ResRegularizer():
    '''Res regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''
    def __init__(self, reg_lambda=None, momentum_mu=None, blocks=None, feature_dim=None):
        self.reg_lambda = reg_lambda
        self.momentum_mu = momentum_mu
        print ("self.reg_lambda: ", self.reg_lambda)
        print ("new self.momentum_mu: ", self.momentum_mu)
        print ("blocks: ", blocks)
        self.correlation_moving_average = []
        for i in range(blocks):
            self.correlation_moving_average.append(np.zeros((feature_dim, feature_dim)))
        print ('new len(self.correlation_moving_average): ', len(self.correlation_moving_average))
   
    def chunk_array(self, arr, chunks, dim):
        if dim == 0:
            chunk_array_list = []
            base = int(arr.shape[0] / chunks)
            for i in range(chunks):
                chunk_array_list.append(arr[i * base: (i+1) * base])
        return chunk_array_list


    # calc correlation using one layer 
    def calcCorrelation(self):
        logger = logging.getLogger('res_reg')
        self.feature_correlation = np.corrcoef(self.feature_matrix, rowvar=False)
        logger.debug ("slef.feature_correlation.shape:")
        logger.debug (self.feature_correlation.shape)

    # calc correlation using two layers 
    def calcCorrelation_two_layers(self):
        self.feature_correlation = []
        logger = logging.getLogger('res_reg')
        # not rnn
        if self.feature_matrix.ndim == 2:
            feature_dim = self.feature_matrix.shape[1]
            logger.debug ("new using two layers self.feature_dim: %d", feature_dim)
            self.feature_correlation.append(np.corrcoef(self.feature_matrix, self.second_feature_matrix, rowvar=False)[feature_dim:, 0:feature_dim])
            logger.debug ("new using two layers self.feature_correlation[0].shape:")
            logger.debug (self.feature_correlation[0].shape)
        # rnn
        else:
            feature_dim = self.feature_matrix.shape[2]
            logger.debug ("rnn new using two layers self.feature_dim: %d", feature_dim)
            logger.debug ("self.batch_first")
            logger.debug (self.batch_first)
            logger.debug ("feature_dim")
            logger.debug (feature_dim)
            if self.batch_first:
                for i in range(self.feature_matrix.shape[1]):
                    self.feature_correlation.append(np.corrcoef(self.feature_matrix[:,i,:], self.second_feature_matrix[:,i,:], rowvar=False)[feature_dim:, 0:feature_dim])
                    logger.debug ('self.feature_matrix[:,i,:] shape:')
                    logger.debug (self.feature_matrix[:,i,:].shape)
                    logger.debug ('self.feature_matrix[:,i,:] norm:')
                    logger.debug (np.linalg.norm(self.feature_matrix[:,i,:]))
                    logger.debug ('self.feature_matrix[:,i,:]:')
                    logger.debug (self.feature_matrix[:,i,:])
                    logger.debug ('self.second_feature_matrix[:,i,:] shape')
                    logger.debug (self.second_feature_matrix[:,i,:].shape)
                    logger.debug ('self.second_feature_matrix[:,i,:] norm:')
                    logger.debug (np.linalg.norm(self.second_feature_matrix[:,i,:]))
                    logger.debug ('self.second_feature_matrix[:,i,:]')
                    logger.debug (self.second_feature_matrix[:,i,:])
                    logger.debug ("new using two layers self.feature_correlation[i].shape:")
                    logger.debug (self.feature_correlation[i].shape)
                    logger.debug ('self.feature_correlation[i] norm')
                    logger.debug (np.linalg.norm(self.feature_correlation[i]))
                    logger.debug ('self.feature_correlation[i]')
                    logger.debug (self.feature_correlation[i])
            else:
                for i in range(self.feature_matrix.shape[0]):
                    self.feature_correlation.append(np.corrcoef(self.feature_matrix[i,:,:], self.second_feature_matrix[i,:,:], rowvar=False)[feature_dim:, 0:feature_dim])
                    logger.debug ("new using two layers self.feature_correlation[i].shape:")
                    logger.debug (self.feature_correlation[i].shape)
    
    # calc moving average
    def calAvgCorrelation(self):
        logger = logging.getLogger('res_reg')
        logger.debug ("len(self.feature_correlation):")
        logger.debug (len(self.feature_correlation))
        logger.debug ("before updating self.correlation_moving_average[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        for i in range(len(self.feature_correlation)):
            logger.debug ("self.feature_idx: ")
            logger.debug (self.feature_idx)
            logger.debug ("self.correlation_moving_average[self.feature_idx] shape: ")
            logger.debug (self.correlation_moving_average[self.feature_idx].shape)
            logger.debug ("before adding self.correlation_moving_average[self.feature_idx] norm: ")
            logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
            logger.debug ('self.feature_correlation[i] shape: ')
            logger.debug (self.feature_correlation[i].shape)
            self.correlation_moving_average[self.feature_idx] = self.momentum_mu * self.correlation_moving_average[self.feature_idx] + (1-self.momentum_mu) * self.feature_correlation[i]
            logger.debug ("after adding self.correlation_moving_average[self.feature_idx] norm: ")
            logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        logger.debug ("self.correlation_moving_average[self.feature_idx].shape:")
        logger.debug (self.correlation_moving_average[self.feature_idx].shape)
        logger.debug ("after updating self.correlation_moving_average[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
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
        logger.debug ('calcRegGradAvg correlation_moving_average self.feature_idx: ')
        logger.debug (self.feature_idx)
        logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx]) # only this line is different
        correlation_abs_avg = np.mean(correlation_abs_matrix)
        correlation_diff_matrix = correlation_abs_avg - correlation_abs_matrix
        if 'lstm' not in self.model_name:
            logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * self.w_array
        else:
            logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * w_array_chunk[0]
            logger.debug ('w_array_chunk[0] shape')
            logger.debug (w_array_chunk[0].shape)
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * w_array_chunk[i]), axis=0)
                logger.debug ("w_array_chunk[i] shape: ")
                logger.debug (w_array_chunk[i].shape)
        logger.debug ("new reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Exp(self):
        logger = logging.getLogger('res_reg')
        logger.debug ('calcRegGradAvg_Exp correlation_moving_average self.feature_idx: ')
        logger.debug (self.feature_idx)
        logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        # print ('calcRegGradAvg_Exp correlation_abs_matrix: ', correlation_abs_matrix)
        if 'lstm' not in self.model_name:
            logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * np.exp(-correlation_abs_matrix) * self.w_array
        else:
            logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * np.exp(-correlation_abs_matrix) * w_array_chunk[0]
            logger.debug ('w_array_chunk[0] shape')
            logger.debug (w_array_chunk[0].shape)
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * np.exp(-correlation_abs_matrix) * w_array_chunk[i]), axis=0)
                logger.debug ("w_array_chunk[i] shape: ")
                logger.debug (w_array_chunk[i].shape)
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w
   
    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Linear(self):
        logger = logging.getLogger('res_reg')
        logger.debug ('calcRegGradAvg_Linear correlation_moving_average self.feature_idx: ')
        logger.debug (self.feature_idx)
        logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        if 'lstm' not in self.model_name:
            logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * (1.0 - correlation_abs_matrix) * self.w_array
        else:
            logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * (1.0 - correlation_abs_matrix) * w_array_chunk[0]
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * (1.0 - correlation_abs_matrix) * w_array_chunk[i]), axis=0)
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w


    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Inverse(self):
        logger = logging.getLogger('res_reg')
        logger.debug ('calcRegGradAvg_Inverse correlation_moving_average self.feature_idx: ')
        logger.debug (self.feature_idx)
        logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        if 'lstm' not in self.model_name:
            logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * (1.0 / correlation_abs_matrix) * self.w_array
        else:
            logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * (1.0 / correlation_abs_matrix) * w_array_chunk[0]
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * (1.0 / correlation_abs_matrix) * w_array_chunk[i]), axis=0)
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Inverse_Var(self):
        logger = logging.getLogger('res_reg')
        logger.debug ('calcRegGradAvg_Inverse_Var correlation_moving_average self.feature_idx: ')
        logger.debug (self.feature_idx)
        logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        if 'lstm' not in self.model_name:
            logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * (1.0 / (1.0 + correlation_abs_matrix)) * self.w_array
        else:
            logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * (1.0 / (1.0 + correlation_abs_matrix)) * w_array_chunk[0]
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * (1.0 / (1.0 + correlation_abs_matrix)) * w_array_chunk[i]), axis=0)
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Gen_Prob(self, labelnum, trainnum):
        logger = logging.getLogger('res_reg')
        logger.debug ('calcRegGradAvg_Gen_prob correlation_moving_average self.feature_idx: ')
        logger.debug (self.feature_idx)
        logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        correlation_abs_matrix_sum = np.sum(correlation_abs_matrix, axis=1).reshape((correlation_abs_matrix.shape[0],1))
        # print ('correlation_abs_matrix_sum shape:', correlation_abs_matrix_sum.shape)
        correlation_abs_matrix_normalize = correlation_abs_matrix / correlation_abs_matrix_sum
        # print ('correlation_abs_matrix_normalize shape: ', correlation_abs_matrix_normalize.shape)
        # print ('np.sum(correlation_abs_matrix_normalize, axis=1)', np.sum(correlation_abs_matrix_normalize, axis=1))
        correlation_abs_matrix_normalize_log = np.log(correlation_abs_matrix_normalize)
        # print ('np.min(correlation_abs_matrix_normalize) after log', np.min(correlation_abs_matrix_normalize_log))
        if 'lstm' not in self.model_name:
            logger.debug ('not lstm')
            reg_grad_w = (-self.reg_lambda * np.sign(self.w_array) * correlation_abs_matrix_normalize_log)/float(labelnum * trainnum)
        else:
            logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = (-self.reg_lambda * np.sign(w_array_chunk[0]) * correlation_abs_matrix_normalize_log)/float(labelnum * trainnum)
            logger.debug ('w_array_chunk[0] shape')
            logger.debug (w_array_chunk[0].shape)
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, (-self.reg_lambda * np.sign(w_array_chunk[i]) * correlation_abs_matrix_normalize_log)/float(labelnum * trainnum)), axis=0)
                logger.debug ("w_array_chunk[i] shape: ")
                logger.debug (w_array_chunk[i].shape)
        logger.debug ("reg_grad_w shape:")
        logger.debug (reg_grad_w.shape)
        return reg_grad_w

    def apply(self, model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, trainnum, epoch, param, name, step, batch_first=True):
        # logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
        logger = logging.getLogger('res_reg')
        self.model_name = model_name
        self.batch_first = batch_first
        self.feature_idx = feature_idx
        self.feature_matrix = features[self.feature_idx].data.cpu().numpy()
        self.second_feature_matrix = features[self.feature_idx + 1].data.cpu().numpy()
        logger.debug ("self.feature_idx: %d", self.feature_idx)
        logger.debug ("self.feature_matrix shape:")
        logger.debug (self.feature_matrix.shape)
        logger.debug ("self.second_feature_matrix shape:")
        logger.debug (self.second_feature_matrix.shape)
        logger.debug ("self.feature_matrix norm: %f", np.linalg.norm(self.feature_matrix))
        logger.debug ("new self.second_feature_matrix norm: %f", np.linalg.norm(self.second_feature_matrix))
        self.reg_lambda = reg_lambda
        logger.debug ("self.reg_lambda: %f", self.reg_lambda)
        self.w_array = param.data.cpu().numpy()
        logger.debug ("self.w_array shape: ")
        logger.debug (self.w_array.shape)
        # self.calcCorrelation()
        self.calcCorrelation_two_layers()
        self.calAvgCorrelation()
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
        if (epoch == 0 and step <= 1000) or step % 1000 == 0:
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
        if (epoch == 0 and step <= 1000) or step % 1000 == 0:
            print ('delta w (data grad + reg grad) norm: ', np.linalg.norm(param.grad.data.cpu().numpy()))
        logger.debug ("delta w (data grad + reg grad) norm: %f", np.linalg.norm(param.grad.data.cpu().numpy()))
        logger.debug ("w norm: %f", np.linalg.norm(param.data.cpu().numpy()))
