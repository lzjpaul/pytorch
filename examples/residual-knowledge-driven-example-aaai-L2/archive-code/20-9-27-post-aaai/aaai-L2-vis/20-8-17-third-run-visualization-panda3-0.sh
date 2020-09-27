### panda3-0
CUDA_VISIBLE_DEVICES=0 python vgg_main_lr05_wd0005_vgg_corr_reg_corr_mean_var_1.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-17.log
