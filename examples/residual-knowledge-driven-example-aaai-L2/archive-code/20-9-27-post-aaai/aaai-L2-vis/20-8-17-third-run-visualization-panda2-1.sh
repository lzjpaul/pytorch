### panda2-1
### vgg
CUDA_VISIBLE_DEVICES=1 python vgg_main_lr05_wd0005_vgg_wd_corr_mean_var.py -modelname vgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run-visualization/20-8-17-third-run-visualization-16.log
