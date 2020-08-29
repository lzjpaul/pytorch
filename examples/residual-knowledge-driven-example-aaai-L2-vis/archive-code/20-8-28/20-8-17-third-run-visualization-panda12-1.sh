### panda12-1
CUDA_VISIBLE_DEVICES=1 /hdd1/zhaojing/anaconda3-cuda-10/bin/python vgg_main_lr05_wd0005_vgg_tune_corr_reg_vis_3.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-38.log
