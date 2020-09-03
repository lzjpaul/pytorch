import numpy as np
import torch
import logging

# Attention
# (1) self.feature_dim if not equal for two layers, then "for idx in range(self.feature_dim):" in "update_Theta_Current_Layer" need to be changed
small_value = np.finfo(float).eps

class ResRegularizerDiffDim():
    '''Res regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''
    def __init__(self, prior_beta=None, reg_lambda=None, momentum_mu=None, blocks=None, feature_dim_vec=None, model_name=None):
        self.prior_beta = prior_beta
        self.reg_lambda = reg_lambda
        self.momentum_mu = momentum_mu
        self.feature_dim_vec = feature_dim_vec  # this should be a vector, and blocks is len
        # print ("prior model_name: ", model_name)
        # print ("prior self.doc_num: ", self.doc_num)
        # print ("prior self.prior_beta: ", self.prior_beta)
        # print ("prior self.reg_lambda: ", self.reg_lambda)
        # print ("prior new self.momentum_mu: ", self.momentum_mu)
        # print ("diff dim check new self.feature_dim_vec: ", self.feature_dim_vec)
        # print ("diff dim check blocks: ", blocks)
        self.correlation_moving_average = []
        self.reg_grad_w = []
        self.theta_all_layer = []
        for i in range(blocks):
            self.correlation_moving_average.append(np.zeros((self.feature_dim_vec[i+1], self.feature_dim_vec[i])))  # [output*input]
        for i in range(blocks):
            self.reg_grad_w.append(np.zeros((self.feature_dim_vec[i+1], self.feature_dim_vec[i])))  # [output*input]
        for i in range(blocks):
            self.theta_all_layer.append(np.full((self.feature_dim_vec[i+1], self.feature_dim_vec[i]), 1./self.feature_dim_vec[i]))
        # print ('diff dim check len(self.correlation_moving_average): ', len(self.correlation_moving_average))
        # print ('diff dim check len(self.theta_all_layer): ', len(self.theta_all_layer))
        # print ('diff dim check self.theta_all_layer[0]: ', self.theta_all_layer[0])
        # print ('diff dim check self.theta_all_layer[0] shape: ', self.theta_all_layer[0].shape)
   
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
        # logger.debug ("slef.feature_correlation.shape:")
        # logger.debug (self.feature_correlation.shape)

    # calc correlation using two layers 
    def calcCorrelation_two_layers(self):
        self.feature_correlation = []
        logger = logging.getLogger('res_reg')
        # not rnn
        if self.feature_matrix.ndim == 2:
            out_feature_dim = self.feature_dim_vec[self.feature_idx + 1]
            assert self.second_feature_matrix.shape[1] == out_feature_dim
            in_feature_dim = self.feature_dim_vec[self.feature_idx]
            assert self.feature_matrix.shape[1] == in_feature_dim
            # logger.debug ("diff dim check two layers out_feature_dim: %d", out_feature_dim)
            # logger.debug ("diff dim check two layers in_feature_dim: %d", in_feature_dim)
            self.feature_correlation.append(np.corrcoef(self.feature_matrix, self.second_feature_matrix, rowvar=False)[in_feature_dim:, 0:in_feature_dim])
            # logger.debug ("diff dim check using two layers self.feature_correlation[0].shape:")
            # logger.debug (self.feature_correlation[0].shape)
        # rnn
        else:
            print ("Not Implemented for RNN")
    
    # calc moving average
    def calAvgCorrelation(self):
        logger = logging.getLogger('res_reg')
        # logger.debug ("len(self.feature_correlation):")
        # logger.debug (len(self.feature_correlation))
        # logger.debug ("diff dim check before updating self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        for i in range(len(self.feature_correlation)):
            """
            logger.debug ("index i:")
            logger.debug (i)
            logger.debug ("self.feature_idx: ")
            logger.debug (self.feature_idx)
            logger.debug ("self.correlation_moving_average[self.feature_idx] shape: ")
            logger.debug (self.correlation_moving_average[self.feature_idx].shape)
            logger.debug ("before adding self.correlation_moving_average[self.feature_idx] norm: ")
            logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
            logger.debug ('self.feature_correlation[i] shape: ')
            logger.debug (self.feature_correlation[i].shape)
            logger.debug ('self.feature_correlation[i] norm: ')
            logger.debug (np.linalg.norm(self.feature_correlation[i]))
            """
            ## not adding all nan correlation
            if np.isnan(np.linalg.norm(self.feature_correlation[i])):   
                # print ('nan correlation matrix: ', self.feature_correlation[i])
                self.feature_correlation[i][np.isnan(self.feature_correlation[i])]=small_value
                # print ('replace nan correlation matrix: ', self.feature_correlation[i])
            if not np.isnan(np.linalg.norm(self.feature_correlation[i])):
                self.correlation_moving_average[self.feature_idx] = self.momentum_mu * self.correlation_moving_average[self.feature_idx] + (1-self.momentum_mu) * self.feature_correlation[i]
            # logger.debug ("after adding self.correlation_moving_average[self.feature_idx] norm: ")
            # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        # logger.debug ("self.correlation_moving_average[self.feature_idx].shape:")
        # logger.debug (self.correlation_moving_average[self.feature_idx].shape)
        # logger.debug ("after updating self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
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
        # logger.debug ('calcRegGradAvg correlation_moving_average self.feature_idx: ')
        # logger.debug (self.feature_idx)
        # logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx]) # only this line is different
        correlation_abs_avg = np.mean(correlation_abs_matrix)
        correlation_diff_matrix = correlation_abs_avg - correlation_abs_matrix
        if 'lstm' not in self.model_name:
            # logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * self.w_array
        else:
            # logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * w_array_chunk[0]
            # logger.debug ('w_array_chunk[0] shape')
            # logger.debug (w_array_chunk[0].shape)
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * w_array_chunk[i]), axis=0)
                # logger.debug ("w_array_chunk[i] shape: ")
                # logger.debug (w_array_chunk[i].shape)
        # logger.debug ("new reg_grad_w shape:")
        # logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Exp(self):
        logger = logging.getLogger('res_reg')
        # logger.debug ('calcRegGradAvg_Exp correlation_moving_average self.feature_idx: ')
        # logger.debug (self.feature_idx)
        # logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        # print ('calcRegGradAvg_Exp correlation_abs_matrix: ', correlation_abs_matrix)
        if 'lstm' not in self.model_name:
            # logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * np.exp(-correlation_abs_matrix) * self.w_array
        else:
            # logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * np.exp(-correlation_abs_matrix) * w_array_chunk[0]
            # logger.debug ('w_array_chunk[0] shape')
            # logger.debug (w_array_chunk[0].shape)
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * np.exp(-correlation_abs_matrix) * w_array_chunk[i]), axis=0)
                # logger.debug ("w_array_chunk[i] shape: ")
                # logger.debug (w_array_chunk[i].shape)
        # logger.debug ("reg_grad_w shape:")
        # logger.debug (reg_grad_w.shape)
        return reg_grad_w
   
    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Linear(self):
        logger = logging.getLogger('res_reg')
        # logger.debug ('calcRegGradAvg_Linear correlation_moving_average self.feature_idx: ')
        # logger.debug (self.feature_idx)
        # logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        if 'lstm' not in self.model_name:
            # logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * (1.0 - correlation_abs_matrix) * self.w_array
        else:
            # logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * (1.0 - correlation_abs_matrix) * w_array_chunk[0]
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * (1.0 - correlation_abs_matrix) * w_array_chunk[i]), axis=0)
        # logger.debug ("reg_grad_w shape:")
        # logger.debug (reg_grad_w.shape)
        return reg_grad_w


    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Inverse(self):
        logger = logging.getLogger('res_reg')
        # logger.debug ('calcRegGradAvg_Inverse correlation_moving_average self.feature_idx: ')
        # logger.debug (self.feature_idx)
        # logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        if 'lstm' not in self.model_name:
            # logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * (1.0 / correlation_abs_matrix) * self.w_array
        else:
            # logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * (1.0 / correlation_abs_matrix) * w_array_chunk[0]
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * (1.0 / correlation_abs_matrix) * w_array_chunk[i]), axis=0)
        # logger.debug ("reg_grad_w shape:")
        # logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Inverse_Var(self):
        logger = logging.getLogger('res_reg')
        # logger.debug ('calcRegGradAvg_Inverse_Var correlation_moving_average self.feature_idx: ')
        # logger.debug (self.feature_idx)
        # logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        if 'lstm' not in self.model_name:
            # logger.debug ('not lstm')
            reg_grad_w = 2 * self.reg_lambda * (1.0 / (1.0 + correlation_abs_matrix)) * self.w_array
        else:
            # logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = 2 * self.reg_lambda * (1.0 / (1.0 + correlation_abs_matrix)) * w_array_chunk[0]
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, 2 * self.reg_lambda * (1.0 / (1.0 + correlation_abs_matrix)) * w_array_chunk[i]), axis=0)
        # logger.debug ("reg_grad_w shape:")
        # logger.debug (reg_grad_w.shape)
        return reg_grad_w

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGradAvg_Gen_Prob(self, labelnum, seqnum, trainnum, cal_all_timesteps):
        logger = logging.getLogger('res_reg')
        # logger.debug ('calcRegGradAvg_Gen_prob correlation_moving_average self.feature_idx: ')
        # logger.debug (self.feature_idx)
        # logger.debug ("reggrad self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        correlation_abs_matrix_sum = np.sum(correlation_abs_matrix, axis=1).reshape((correlation_abs_matrix.shape[0],1))
        # print ('correlation_abs_matrix_sum shape:', correlation_abs_matrix_sum.shape)
        correlation_abs_matrix_normalize = correlation_abs_matrix / correlation_abs_matrix_sum
        # print ('correlation_abs_matrix_normalize shape: ', correlation_abs_matrix_normalize.shape)
        # print ('np.sum(correlation_abs_matrix_normalize, axis=1)', np.sum(correlation_abs_matrix_normalize, axis=1))
        correlation_abs_matrix_normalize_log = np.log(correlation_abs_matrix_normalize)
        # print ('np.min(correlation_abs_matrix_normalize) after log', np.min(correlation_abs_matrix_normalize_log))
        if cal_all_timesteps:
            normalization_coefficient = float(labelnum * seqnum * trainnum)
        else:
            normalization_coefficient = float(labelnum * trainnum)
        # print ("normalization_coefficient: ", normalization_coefficient)
        """
        logger.debug ("labelnum: ")
        logger.debug (labelnum)
        logger.debug ("seqnum: ")
        logger.debug (seqnum)
        logger.debug ("trainnum: ")
        logger.debug (trainnum)
        logger.debug ("normalization_coefficient: ")
        logger.debug (normalization_coefficient)
        """
        if 'lstm' not in self.model_name:
            # logger.debug ('not lstm')
            reg_grad_w = (-self.reg_lambda * np.sign(self.w_array) * correlation_abs_matrix_normalize_log)/(normalization_coefficient)
        else:
            # logger.debug ('lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = (-self.reg_lambda * np.sign(w_array_chunk[0]) * correlation_abs_matrix_normalize_log)/(normalization_coefficient)
            # logger.debug ('w_array_chunk[0] shape')
            # logger.debug (w_array_chunk[0].shape)
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, (-self.reg_lambda * np.sign(w_array_chunk[i]) * correlation_abs_matrix_normalize_log)/(normalization_coefficient)), axis=0)
                # logger.debug ("w_array_chunk[i] shape: ")
                # logger.debug (w_array_chunk[i].shape)
        # logger.debug ("reg_grad_w shape:")
        # logger.debug (reg_grad_w.shape)
        return reg_grad_w

    def calcRegGradAvg_Gen_Prob_Prior(self, labelnum, seqnum, trainnum, cal_all_timesteps):
        logger = logging.getLogger('res_reg')
        # logger.debug ('diff dim check calcRegGradAvg_Gen_Prob_Prior self.theta_all_layer self.feature_idx: ')
        # logger.debug (self.feature_idx)
        # logger.debug ("diff dim check theta norm reggrad self.theta_all_layer[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.theta_all_layer[self.feature_idx]))
        theta_current_layer_log = np.log(self.theta_all_layer[self.feature_idx])
        # logger.debug ("prior theta_current_layer_log shape: ") 
        # logger.debug (theta_current_layer_log.shape)
        if cal_all_timesteps:
            normalization_coefficient = float(labelnum * seqnum * trainnum)
        else:
            normalization_coefficient = float(labelnum * trainnum)
        # print ("normalization_coefficient: ", normalization_coefficient)
        """
        logger.debug ("labelnum: ")
        logger.debug (labelnum)
        logger.debug ("seqnum: ")
        logger.debug (seqnum)
        logger.debug ("trainnum: ")
        logger.debug (trainnum)
        logger.debug ("normalization_coefficient: ")
        logger.debug (normalization_coefficient)
        """
        '''
        if 'lstm' not in self.model_name:
            logger.debug ('prior not lstm')
        '''
        # logger.debug ("self.reg_lambda: %f",self.reg_lambda)
        # logger.debug ("self.w_array norm: %f", np.linalg.norm(self.w_array))
        # logger.debug ("self.w_array[0]: %s", self.w_array[0])
        # logger.debug ("theta_current_layer_log norm: %f", np.linalg.norm(theta_current_layer_log))
        # logger.debug ("theta_current_layer_log[0]: %s", theta_current_layer_log[0])
        reg_grad_w = (-2 * self.reg_lambda * self.w_array * theta_current_layer_log)/(normalization_coefficient)
        # logger.debug ("reg_grad_w norm: ")
        # logger.debug (np.linalg.norm(reg_grad_w))
        '''
        else:
            logger.debug ('prior lstm')
            w_array_chunk = self.chunk_array(self.w_array,4,0)
            reg_grad_w = (-self.reg_lambda * np.sign(w_array_chunk[0]) * theta_current_layer_log)/(normalization_coefficient)
            logger.debug ('prior w_array_chunk[0] shape')
            logger.debug (w_array_chunk[0].shape)
            for i in range(1,4):
                reg_grad_w = np.concatenate((reg_grad_w, (-self.reg_lambda * np.sign(w_array_chunk[i]) * theta_current_layer_log)/(normalization_coefficient)), axis=0)
                logger.debug ("prior w_array_chunk[i] shape: ")
                logger.debug (w_array_chunk[i].shape)
        '''
        # logger.debug ("prior reg_grad_w shape:")
        # logger.debug (reg_grad_w.shape)
        return reg_grad_w

    def update_Theta_Current_Layer(self, step):
        logger = logging.getLogger('res_reg')
        # logger.debug ('prior calcRegGradAvg_Gen_Prob_Prior correlation_moving_average self.feature_idx: ')
        # logger.debug (self.feature_idx)
        # logger.debug ("prior reggrad self.correlation_moving_average[self.feature_idx] norm:")
        # logger.debug (np.linalg.norm(self.correlation_moving_average[self.feature_idx]))
        correlation_abs_matrix = np.abs(self.correlation_moving_average[self.feature_idx])
        # logger.debug ("correlation_abs_matrix norm")
        # logger.debug (np.linalg.norm(correlation_abs_matrix))
        # logger.debug ("correlation_abs_matrix")
        # logger.debug (correlation_abs_matrix)
        """normalization
        correlation_abs_matrix_sum = np.sum(correlation_abs_matrix, axis=1).reshape((-1,1))
        print("correlation_abs_matrix_sum shape: \n", correlation_abs_matrix_sum.shape)
        print("correlation_abs_matrix_sum: \n", correlation_abs_matrix_sum)
        correlation_abs_matrix_sum = correlation_abs_matrix_sum.astype(float)
        print("after float correlation_abs_matrix_sum: \n", correlation_abs_matrix_sum)
        correlation_abs_matrix = correlation_abs_matrix / correlation_abs_matrix_sum
        print ("after normalization correlation_abs_matrix: \n", correlation_abs_matrix)
        print ("np.sum(correlation_abs_matrix, axis=1): \n", np.sum(correlation_abs_matrix, axis=1))

        """
        self.prior_alpha = 1.0 + self.reg_lambda * self.prior_beta * correlation_abs_matrix
        """
        logger.debug ('prior check self.prior_alpha shape: ')
        logger.debug (self.prior_alpha.shape)
        logger.debug ("prior check self.w_array shape: ")
        logger.debug (self.w_array.shape)
        logger.debug ("self.w_array norm")
        logger.debug (np.linalg.norm(self.w_array))
        """
        # print ("self.w_array[0]: ", self.w_array[0])
        # print ("update theta average self.w_array*self.w_array: ", (np.linalg.norm(self.w_array) * np.linalg.norm(self.w_array) / self.w_array.size))
        # print ("correlation_abs_matrix: ", correlation_abs_matrix)
        # print ("step: ", step)
        # print ("update theta average corr: ", np.sqrt(np.linalg.norm(correlation_abs_matrix) * np.linalg.norm(correlation_abs_matrix) / correlation_abs_matrix.size))
        # logger.debug ("self.w_array")
        # logger.debug (self.w_array)
        # logger.debug ("self.w_array[0:self.feature_dim] norm")
        # logger.debug (np.linalg.norm(self.w_array[0:self.feature_dim]))
        doc_num = self.feature_dim_vec[self.feature_idx + 1]
        in_feature_dim = self.feature_dim_vec[self.feature_idx]
        # logger.debug ("diff dim check doc_num: ")
        # logger.debug (doc_num)
        # logger.debug ("diff dim check in_feature_dim: ")
        # logger.debug (in_feature_dim)
        for doc_idx in range(doc_num):
            # logger.debug ("prior doc_idx")
            # logger.debug (doc_idx)
            theta_doc = (self.reg_lambda * self.w_array[doc_idx, :] * self.w_array[doc_idx, :] + (self.prior_alpha[doc_idx] - 1.0)) / np.sum(self.reg_lambda * self.w_array[doc_idx, :] * self.w_array[doc_idx, :] + (self.prior_alpha[doc_idx] - 1.0)) # here: self.w_array[doc_idx, :]
            # logger.debug ('prior (self.reg_lambda * self.w_array[doc_idx, :] * self.w_array[doc_idx, :] + (self.prior_alpha[doc_idx] - 1.0)) shape: ')
            # logger.debug ((self.reg_lambda * self.w_array[doc_idx, :] * self.w_array[doc_idx, :] + (self.prior_alpha[doc_idx] - 1.0)).shape)
            # logger.debug ("prior check self.theta_all_layer[self.feature_idx] shape: ")
            # logger.debug (self.theta_all_layer[self.feature_idx].shape)
            # logger.debug ("prior check theta_doc shape: ")
            # logger.debug (theta_doc.shape)
            self.theta_all_layer[self.feature_idx][doc_idx] = theta_doc
            '''
            if step % 1000 == 0:
                print ("prior self.feature_idx: ", self.feature_idx)
                print ('prior theta_doc:', theta_doc)
            '''
        """
        logger.debug ("self.feature_idx:")
        logger.debug (self.feature_idx)
        logger.debug ("prior check theta norm update_Theta_Current_Layer self.theta_all_layer[self.feature_idx] norm:")
        logger.debug (np.linalg.norm(self.theta_all_layer[self.feature_idx]))
        logger.debug ("prior update_Theta_Current_Layer np.sum(self.theta_all_layer[self.feature_idx], axis=1):")
        logger.debug (np.sum(self.theta_all_layer[self.feature_idx], axis=1))
        """


    def apply(self, model_name, gpu_id, features, feature_idx, reg_method, reg_lambda, labelnum, seqnum, trainnum, epoch, param, name, step, batch_first=True, cal_all_timesteps=False):
        # logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
        logger = logging.getLogger('res_reg')
        self.model_name = model_name
        self.batch_first = batch_first
        self.feature_idx = feature_idx
        # logger.debug ("apply reg_method: %d", reg_method)
        # logger.debug ("apply reg_lambda: %f", reg_lambda)
        # logger.debug ("apply name: %s", name)
        # logger.debug ("apply step: %d", step)
        # logger.debug ("apply cal_all_timesteps: %s", cal_all_timesteps)
        uptfreq = 1
        if trainnum == 12379:  # MIMIC-III
            uptfreq = 12
        elif trainnum == 5788:  # Movie Review
            uptfreq = 5
        elif trainnum == 60000 and 'mlp' in model_name:  # MNIST + mlp
            uptfreq = 90
        elif 'vgg' in model_name:  # vgg
            uptfreq = 39
        elif 'lenet' in model_name:  # lenet
            uptfreq = 23
        elif 'autoenc' in model_name:  # autoencoder
            uptfreq = 45
        else:
            uptfreq = 1
        # print ("uptfreq: ", uptfreq)
        if 'dropout' not in model_name:
            self.feature_matrix = features[self.feature_idx].data.cpu().numpy()
            self.second_feature_matrix = features[self.feature_idx + 1].data.cpu().numpy()
        else:
            self.feature_matrix = features[2 * self.feature_idx].data.cpu().numpy()
            self.second_feature_matrix = features[2 * self.feature_idx + 1].data.cpu().numpy()
        # logger.debug ("diff dim check self.feature_idx: %d", self.feature_idx)
        # logger.debug ("diff dim check self.feature_matrix shape:")
        # logger.debug (self.feature_matrix.shape)
        # logger.debug ("diff dim check self.second_feature_matrix shape:")
        # logger.debug (self.second_feature_matrix.shape)
        # print ("self.feature_matrix shape:", self.feature_matrix.shape)
        # print ("self.feature_matrix: ", self.feature_matrix)
        # np.savetxt('/hdd2/zhaojing/res-regularization/home_code_edit/coding-area/20-8-7/first_feature_matrix.csv', self.feature_matrix, delimiter=',')
        # print ("self.second_feature_matrix shape:", self.second_feature_matrix.shape)
        # print ("self.second_feature_matrix: ", self.second_feature_matrix)
        # np.savetxt('/hdd2/zhaojing/res-regularization/home_code_edit/coding-area/20-8-7/second_feature_matrix.csv', self.second_feature_matrix, delimiter=',')
        # logger.debug ("diff dim check self.feature_matrix norm: %f", np.linalg.norm(self.feature_matrix))
        # logger.debug ("diff dim check self.second_feature_matrix norm: %f", np.linalg.norm(self.second_feature_matrix))
        self.reg_lambda = reg_lambda
        # logger.debug ("self.reg_lambda: %f", self.reg_lambda)
        self.w_array = param.data.cpu().numpy()
        # logger.debug ("diff dim check self.w_array shape: ")
        # logger.debug (self.w_array.shape)
        # self.calcCorrelation()
        if epoch < 3 or step % uptfreq == 0:
            self.calcCorrelation_two_layers()
            self.calAvgCorrelation()
        # self.reg_grad_w = self.calcRegGrad()
        # print ("trainnum: ", trainnum)
        # print ("seqnum: ", seqnum)
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
            self.reg_grad_w = self.calcRegGradAvg_Gen_Prob(labelnum, seqnum, trainnum, cal_all_timesteps)
        # generation probablity using prior
        elif reg_method == 6:
            # print ("in self.calcRegGradAvg_Gen_Prob_Prior")
            if epoch < 3 or (step-1) % uptfreq == 0:
                self.reg_grad_w[self.feature_idx] = self.calcRegGradAvg_Gen_Prob_Prior(labelnum, seqnum, trainnum, cal_all_timesteps)
        else:
            print("Invalid regularization method, exiting...")
            exit()
        reg_grad_w_dev = (torch.from_numpy(self.reg_grad_w[self.feature_idx])).float()
        """
        if (epoch == 0 and step <= 1000) or step % 1000 == 0:
            print ('step: ', step)
            print ('name: ', name)
            print ('w norm: ', np.linalg.norm(param.data.cpu().numpy()))
            print ('data grad norm: ', np.linalg.norm(param.grad.data.cpu().numpy()))
            print ('reg_grad_w_dev l2 norm: ', np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        """
        # logger.debug ("step: %d", step)
        # logger.debug ("name: " +  name)
        # logger.debug ("data grad l2 norm: %f", np.linalg.norm(param.grad.data.cpu().numpy()))
        # logger.debug ("reg_grad_w_dev l2 norm: %f", np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        if gpu_id >= 0:
            param.grad.data.add_(1.0, reg_grad_w_dev.cuda(gpu_id)) # here3
        else:
            param.grad.data.add_(1.0, reg_grad_w_dev) # here3
        # if (epoch == 0 and step <= 1000) or step % 1000 == 0:
        #     print ('delta w (data grad + reg grad) norm: ', np.linalg.norm(param.grad.data.cpu().numpy()))
        # logger.debug ("delta w (data grad + reg grad) norm: %f", np.linalg.norm(param.grad.data.cpu().numpy()))
        # logger.debug ("w norm: %f", np.linalg.norm(param.data.cpu().numpy()))
        if reg_method == 6:
            if epoch < 3 or step % uptfreq == 0:
                self.update_Theta_Current_Layer(step)
            # if epoch <= 3 or epoch == 50:
            #     print ("prior self.feature_idx: ", self.feature_idx)
            #     print ('prior self.theta_all_layer[0][0, :5]:', self.theta_all_layer[0][0, :5])
