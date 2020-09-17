###### VGG-6, panda3-2
CUDA_VISIBLE_DEVICES=2 python vgg_main_lr05_wd0005.py -modelname dropoutregvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 9 --dropout 0.1 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-15/20-8-15-second-run/20-8-15-second-run-87.log
CUDA_VISIBLE_DEVICES=2 python vgg_main_lr05_wd0005_tune_corr_reg_vgg_5.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-15/20-8-15-second-run/20-8-15-second-run-97.log
