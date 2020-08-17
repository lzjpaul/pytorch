###########19-2-27
###########temporarily used as lda-prior folder for running calcRegGradAvg_Gen_Prob_Prior()
###########also, developing word language modling (wlm) and mnist

(2) 19-3-2
(2-1) wlm model code
(2-2) data.py --> data_origin.py (original example)


19-3-4:
(1) adding train_lstm_main_hook_resreg_real_wlm.py  in residual-knowledge-driven-example-test-lda-prior for wlm model, but the non-wlm part is not changed and not tested ....

19-4-10:
(1) in "residual-knowledge-driven-example-test-lda-prior", for train_lstm_main_hook_resreg_real_wlm.py
--divided by (label_num * time_step * sample_num)--> since sample_num = (train_data.size(0) * train_data.size(1))/seqnum, the # of mini-batches may not be accurate since the last mini-batch does not contain 100 samples 
--for non-wlm, not changed and not tested yet!!

19-4-11:
(1) adding validation dataset and doing learning rate annealing in "residual-knowledge-driven-example-test-lda-prior" folder

19-4-15:
MNIST
(1) getting mlp_residual_hook_resreg.py from ../residual-knowledge-driven-example/archive-code/19-2-13/mlp_residual_hook_resreg.py, because mlp_residual_hook_resreg.py contains MNIST while 19-2-13 is the latest that has MNIST --> but still running in ../residual-knowledge-driven-example-test-lda-prior/ folder

(3) 19-4-15
(3-1) mlp_residual_hook_resreg_real_mnist.py: adapted for mnist dataset

(4) 19-4-16
(4-1) for MNIST, the loss function is log_softmax

(5) 19-4-17
(5-1) try momentum, dropout, batchsize using train_lstm_main_hook_resreg_real_wlm_no_momentum.py and train_lstm_main_hook_resreg_real_wlm_dropout.py
(5-2) no momentum for wlm

(5-3) now the best code is train_lstm_main_hook_resreg_real_wlm.py and mlp_residual_hook_resreg_real_mnist.py!!!!!

(6) 19-4-20
(6-1) check convergence for wlm
(6-2) train_lstm_main_hook_resreg_real_wlm_tune_hyperparam.py && mlp_residual_hook_resreg_real_mnist_tune_hyperparam.py
      --> using "for" for prior_beta, reg_lambda, weight_decay
from train_lstm_main_hook_resreg_real_wlm.py and mlp_residual_hook_resreg_real_mnist.py

(7) 19-4-21
(7-1) train_lstm_main_hook_resreg_real_wlm_tune_converge.py/train_lstm_main_hook_resreg_real_wlm_tune_hyperparam.py/train_lstm_main_hook_resreg_real_wlm.py --> no gradient clip
(7-2) train_wlm_tune_wd0000110_lambda0000110_beta_1.py ... and .mlp_residual_tune_wd0000110_lambda0000110_beta8.py

--train_wlm_tune_wd0_01_lambda1_beta_1.py: wd from 0 to 0.1, lambda only one value, beta only one value
--train_wlm_tune_wd000010001_lambda0000110_beta8: wd from 0.00001 to 0.0001, lambda drom 0.00001 to 10, beta 8 values

all from train_lstm_main_hook_resreg_real_wlm_tune_hyperparam.py  and mlp_residual_hook_resreg_real_mnist_tune_hyperparam.py 

(8) 19-4-23
(8-1) resume gradient clip!!
(8-2) find  MNISt and wlm 1-3 blocks weight decay

(9) 19-5-5
(9-1) rerun some configurations, tune better wlm and mnist

(10) 19-5-9
(10-1) cal_corr_mean_var for visualization null hypothesis
mv mlp_residual_tune_wd0001_lambda1_beta_1.py mlp_residual_tune_cal_corr_mean_var.py
mv train_wlm_tune_wd0001_lambda1_beta_1.py train_wlm_tune_cal_corr_mean_var.py 

(11) 19-5-12
(11-1) adding 'int': if epoch == 0 or ((epoch+1) % int(max_epoch/4)) == 0:

(12) 19-5-14
(12-1) null hypothesis

(13) 19-5-15
(13-1) train_wlm_tune_cal_corr_mean_var.py solves nan conditions


########################AAAI experiments!! #########################
(14) 20-7-25
(14-1) mlp_residual_hook_resreg_real_mnist_tune_hyperparam.py && train_lstm_main_hook_resreg_real_wlm_tune_hyperparam.py
model.eval()

20-7-26
(1) residual-knowledge-driven-example-aaai-L2
L2 combined with corr-reg, experiments for aaai
(2) res_regularizer.py
coming from residual-knowledge-driven-example-L2/
--change SGD part
--change theta update 
(3) 
vgg_main_origin.py: baseline-cifar-models/vgg-cyfu-final/main.py
lenet_run_main_NLL_GPU_SGD_lr01_origin.py: baseline-lenet-mnist/LeNet-5/run_main_NLL_GPU_SGD_lr01.py
autoencoder_MNIST_main_origin.py: baseline-autoencoder-model/autoencoder_MNIST_main.py

20-7-27:
(1) three models adding features.append(), all models in one main.py, etc --> see github

20-7-28:
(1) res_regularizer_diff_dim.py
different dimensions of feature vectors
(2) all have one more dropout model structure options ... 
both res_regularizer.py and res_regularizer_diff_dim.py:
and dropout takes the (2 times i) and (2 times i + 1) as the first and second feature matrix ...

20-7-29:
(1) debug for L2 corr-reg, three models corr-reg, diff dims and dropout model structures
(2) code for correlation normalization for both res_regularizer.py and res_regularizer_diff_dim.py
(3) also code for L1, maxnorm, etc

20-8-3:
20-8-3-lenet-lr-005.sh/20-8-3-lenet-lr-01.sh/lenet_main_NLL_GPU_SGD_lr..._momentum_reproduce.py: only has one wd and run once

20-8-4:
thinking of transfer autoencoder to adam ...
autoencoder_MNIST_main.py --> autoencoder_MNIST_main_SGD.py

20-8-3-wd-autoencoder-1.sh/autoencoder_MNIST_main_tune_wd_1.py: still SGD, but try to use 1e-10, 1e-9... much smaller wd

20-8-5:
XXXX_origin.py: see 20-7-26

mlp_residual_hook_resreg_real_mnist_wd0001/00001/01.py from mlp_residual_hook_resreg_real_mnist_tune_hyperparam.py 
train_lstm_main_hook_resreg_real_wlm_wd0001/00001.py from train_lstm_main_hook_resreg_real_wlm_tune_hyperparam.py 

20-8-5 overall code evolution logic:

####### list of hyperparameters
mlp_residual_hook_resreg_real_mnist_tune_hyperparam.py
train_lstm_main_hook_resreg_real_wlm_tune_hyperparam.py
autoencoder_MNIST_main_SGD.py
(now using Adam!!!)lenet_main_NLL_GPU_SGD_lr005.py (now using Adam!!!)
vgg_main.py

####### fixed weightdecay for L2 baseline
mlp_residual_hook_resreg_real_mnist_wd00001.py
train_lstm_main_hook_resreg_real_wlm_wd00001.py
autoencoder_MNIST_main_SGD_lr10_wd000001.py
(now using Adam!!!) lenet_main_NLL_GPU_SGD_lr005_wd0001.py (now using Adam!!!)
vgg_main_lr05_wd0005.py

####### then tune Lasso, maxnorm, dropout ...

20-8-6
lenet_main_NLL_GPU_Adam_wd0.py: for lenet, even adam without wd can be 0.993X, similar to dropout (0.9935)...

20-8-7
debug for nan correlation of autoencoder and lenet ...

20-8-13
(1) lazy update ...
(2) res_regularizer_diff_dim_no_lazy.py and res_regularizer_no_lazy.py: previously no lazy version ...
(3) lenet 100 and 200 epochs final results
(4) autoencoder recosntruction loss * 10000

20-8-16
(1) for lazy update, self.reg_grad_w needs to be a list ...
self.reg_grad_w[self.feature_idx]

20-8-17 
(1) visualization of correlation and weights ...
(code in "residual-knowledge-driven-example-aaai-L2-vis" folder)
