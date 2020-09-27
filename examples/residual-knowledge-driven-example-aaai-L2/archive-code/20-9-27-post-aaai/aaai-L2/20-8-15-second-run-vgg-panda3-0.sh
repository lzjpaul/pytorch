###### VGG-5, panda3-0
CUDA_VISIBLE_DEVICES=0 python vgg_main_lr05_wd0.py -modelname vgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-15/20-8-15-second-run/20-8-15-second-run-84.log
CUDA_VISIBLE_DEVICES=0 python vgg_main_lr05_wd0005_tune_corr_reg_vgg_4.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-15/20-8-15-second-run/20-8-15-second-run-96.log
