### panda5-1
### autoencoder
CUDA_VISIBLE_DEVICES=1 /hdd1/zhaojing/anaconda3-cuda-10/bin/python autoencoder_MNIST_main_SGD_lr10_wd000001_autoencoder_vis.py -modelname autoenc -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-31.log
CUDA_VISIBLE_DEVICES=1 /hdd1/zhaojing/anaconda3-cuda-10/bin/python autoencoder_MNIST_main_SGD_lr10_wd000001_autoencoder_tune_corr_reg_vis.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-32.log
### lenet
CUDA_VISIBLE_DEVICES=1 /hdd1/zhaojing/anaconda3-cuda-10/bin/python lenet_main_NLL_GPU_Adam_wd0_lenet_vis.py -modelname lenet -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-33.log
## (6)
CUDA_VISIBLE_DEVICES=1 /hdd1/zhaojing/anaconda3-cuda-10/bin/python lenet_main_NLL_GPU_Adam_wd0_lenet_tune_corr_reg_vis.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-34.log
