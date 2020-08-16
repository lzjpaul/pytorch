###### Lenet, panda14-0
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python lenet_main_NLL_GPU_Adam_wd0_tune_corr_reg.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-solve-nan-49.log
