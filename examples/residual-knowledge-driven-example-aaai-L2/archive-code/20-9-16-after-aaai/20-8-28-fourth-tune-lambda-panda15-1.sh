###### Lenet, panda7-0(16 hours: 48)
### panda15-1
CUDA_VISIBLE_DEVICES=1 /hdd1/zhaojing/anaconda3-cuda-10/bin/python lenet_main_NLL_GPU_Adam_wd0_tune_corr_reg_lenet_lambda.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-13.log
############################################ 20-8-29-end !!!!!!!!!!!!!! ##############################################