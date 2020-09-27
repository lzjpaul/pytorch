## corr_mean_var
### panda2-0
### autoencoder
CUDA_VISIBLE_DEVICES=0 python autoencoder_MNIST_main_SGD_lr10_wd000001_autoencoder_wd_corr_mean_var.py -modelname autoenc -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-12.log
CUDA_VISIBLE_DEVICES=0 python autoencoder_MNIST_main_SGD_lr10_wd000001_autoencoder_corr_reg_corr_mean_var.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-13.log
### lenet
CUDA_VISIBLE_DEVICES=0 python lenet_main_NLL_GPU_Adam_wd0_lenet_wd_corr_mean_var.py -modelname lenet -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-14.log
CUDA_VISIBLE_DEVICES=0 python lenet_main_NLL_GPU_Adam_wd0_lenet_corr_reg_corr_mean_var.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-15.log
