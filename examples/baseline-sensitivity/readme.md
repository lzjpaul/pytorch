(1) code for sensitivity driven

1-1) change sigmoid and logsoftmax function to layer
1-2) register_backward_hook

20-7-18
2) sensitivity_regularizer.py

sensitivity regularizer\

3) try reduce=None, what is the grad shape??

CUDA_VISIBLE_DEVICES=0 python mlp_residual_tune_cal_corr_mean_var_wd0001_lambda1_beta1.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname mlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0

reduction=None for loss.item() and f.grad.data
train_loss += (loss.mean().item() * len(data)) # sum over all samples in the mini-batch  # need to use mean().item()

throws "grad can be implicitly created only for scalar outputs"
