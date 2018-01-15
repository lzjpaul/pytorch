import numpy as np
import torch

class LDARegularizer():
    '''LDA regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''
    def __init__(self, hyperpara=None, ldapara=None, theta=None, phi=None, uptfreq=None):
        self.alpha, self.phi = hyperpara[0], np.copy(phi)
        self.doc_num, self.topic_num, self.word_num = ldapara[0], ldapara[1], ldapara[2]
        print ("self.alpha, self.doc_num, self.topic_num, self.word_num: ", self.alpha, self.doc_num, self.topic_num, self.word_num)
        print ("self.phi shape: ", self.phi.shape)
        self.theta_alldoc = np.zeros((self.doc_num, self.topic_num))
        for doc_idx in range(self.doc_num):
            self.theta_alldoc[doc_idx,:] = np.copy(theta)
        print ("init self.theta_alldoc shape: ", self.theta_alldoc.shape)
        print ("init self.theta_alldoc: ", self.theta_alldoc)
        self.ldauptfreq, self.paramuptfreq = uptfreq[0], uptfreq[1]
        print ("init self.ldauptfreq, self.paramuptfreq: ", self.ldauptfreq, self.paramuptfreq)

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        self.responsibility_all_doc = np.zeros((self.doc_num, self.word_num, self.topic_num))
        for doc_idx in range(self.doc_num):
            responsibility_doc = self.phi*(self.theta_alldoc[doc_idx].reshape((1, -1)))
            # print 'responsibility_doc[10]: ', responsibility_doc[10]
            # responsibility normalized with summation(denominator)
            self.responsibility_all_doc[doc_idx] = responsibility_doc/(np.sum(responsibility_doc, axis=1).reshape(-1,1))
            # for some words, the topic distributions are all zeros, we will fill these word's topic distribution with average value
            zero_idx = np.where((np.sum(responsibility_doc, axis=1)) == 0)[0]
            average_theta_matrix = np.full((len(zero_idx), self.topic_num), 1./self.topic_num)
            self.responsibility_all_doc[doc_idx][zero_idx] = average_theta_matrix
            '''
            print "doc_idx: ", doc_idx
            print "self.responsibility_all_doc[doc_idx][10]: ", self.responsibility_all_doc[doc_idx][10]
            print "self.responsibility_all_doc[doc_idx][zero_idx].shape: ", self.responsibility_all_doc[doc_idx][zero_idx].shape
            print "len(zero_idx): ", len(zero_idx)
            '''
            # print "word responsibility sum[:100]: ", np.sum(self.responsibility_all_doc[doc_idx], axis=1)[:100]
            # print "word responsibility sum shape: ", np.sum(self.responsibility_all_doc[doc_idx], axis=1).shape
            # print "word responsibility sum nan place: ", np.argwhere(np.isnan(np.sum(self.responsibility_all_doc[doc_idx], axis=1)))
            # print "word responsibility sum sum: ", np.sum(np.sum(self.responsibility_all_doc[doc_idx], axis=1))
            # print 'self.responsibility_all_doc sum: ', np.sum(self.responsibility_all_doc)

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGrad(self):
        theta_phi_all_doc = np.zeros((self.doc_num, self.word_num))
        for doc_idx in range(self.doc_num):
            theta_phi_doc = self.phi*(self.theta_alldoc[doc_idx].reshape((1, -1)))
            theta_phi_doc = np.sum(theta_phi_doc, axis=1)
            zero_idx = np.where(theta_phi_doc == 0)[0]
            min_theta_phi_doc = np.full((len(zero_idx),), -100.)
            theta_phi_doc = np.log(theta_phi_doc)
            theta_phi_doc[zero_idx] = min_theta_phi_doc
            # print 'len(zero_idx): ', len(zero_idx)
            # print 'theta_phi_doc[zero_idx]: ', theta_phi_doc[zero_idx]
            theta_phi_all_doc[doc_idx, :] = theta_phi_doc # here: theta_phi_all_doc[doc_idx, :]
        # print 'print theta_phi_all_doc sum: ', np.sum(theta_phi_all_doc)
        # print 'print min: np.sort(np.unique(theta_phi_all_doc.reshape((1,-1)))): ', np.sort(np.unique(theta_phi_all_doc.reshape((1,-1))))
        return -(np.sign(self.w_array) * theta_phi_all_doc)

    def update_LDA_EM(self, name, step):
        self.theta_alldoc = np.zeros((self.doc_num, self.topic_num))
        # update theta_all_doc
        for doc_idx in range(self.doc_num):
            theta_doc = (np.sum((self.responsibility_all_doc[doc_idx] * np.absolute(self.w_array[doc_idx, :]).reshape((-1,1))), axis=0) + (self.alpha - 1)) / np.sum(np.sum((self.responsibility_all_doc[doc_idx] * np.absolute(self.w_array[doc_idx, :]).reshape((-1,1))), axis=0) + (self.alpha - 1)) # here: self.w_array[doc_idx, :]
            self.theta_alldoc[doc_idx] = theta_doc
            if step % self.ldauptfreq == 0:
                print ('theta_doc:', theta_doc)
        # print 'sum: np.sum(self.theta_alldoc): ', np.sum(self.theta_alldoc)

    def apply(self, gpu_id, trainnum, labelnum, epoch, param, name, step):
        self.w_array = param.data.cpu().numpy()
        if epoch < 2 or step % self.paramuptfreq == 0:
            self.calcResponsibility()
            # self.reg_grad_w = np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w_array.shape) * self.w_array
            self.reg_grad_w = self.calcRegGrad()
        reg_grad_w_dev = torch.from_numpy(self.reg_grad_w/float(trainnum * labelnum))
        if gpu_id >= 0:
            reg_grad_w_dev.cuda(gpu_id)
            param.grad.data.cuda(gpu_id)
        if (epoch == 0 and step < 50) or step % self.ldauptfreq == 0:
            print ("step: ", step)
            print ("name: ", name)
            print ("data grad l2 norm: ", np.linalg.norm(param.grad.data.cpu().numpy()))
            print ("reg_grad_w_dev l2 norm: ", np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        param.grad.data.add_(1.0, reg_grad_w_dev) # here3
        if (epoch == 0 and step < 50) or step % self.ldauptfreq == 0:
            print ("delta w norm: ", np.linalg.norm(param.grad.data.cpu().numpy()))
            print ("w norm: ", np.linalg.norm(param.data.cpu().numpy()))
        if epoch < 2 or step % self.ldauptfreq == 0:
            if epoch >=2 and step % self.paramuptfreq != 0:
                self.calcResponsibility()
            self.update_LDA_EM(name, step)
        # return grad
