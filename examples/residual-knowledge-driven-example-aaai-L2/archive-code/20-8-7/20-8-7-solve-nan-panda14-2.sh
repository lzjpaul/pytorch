###### VGG-2, panda14-2
CUDA_VISIBLE_DEVICES=2 /hdd1/zhaojing/anaconda3-cuda-10/bin/python vgg_main_lr05_wd0005_tune_corr_reg_2.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-solve-nan-51.log
