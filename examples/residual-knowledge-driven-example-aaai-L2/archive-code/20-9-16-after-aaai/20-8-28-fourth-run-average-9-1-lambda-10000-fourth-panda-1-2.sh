###### lenet 18+18+21+21 (16 hours: 48)
### panda1-2
CUDA_VISIBLE_DEVICES=2 python lenet_main_NLL_GPU_Adam_wd0_tune_corr_reg_lenet_e8.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-137.log
CUDA_VISIBLE_DEVICES=2 python lenet_main_NLL_GPU_Adam_wd0_tune_corr_reg_lenet_e876.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-137-876.log
########## 20-9-1-rerun end