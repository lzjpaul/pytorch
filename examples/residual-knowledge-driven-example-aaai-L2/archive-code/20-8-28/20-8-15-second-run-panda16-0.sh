###### VGG-2, panda16-0
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python vgg_main_lr05_wd0005_tune_lasso_86.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 7 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-15/20-8-15-second-run/20-8-15-second-run-85.log
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python vgg_main_lr05_wd0005.py -modelname dropoutregvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 9 --dropout 0.1 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-15/20-8-15-second-run/20-8-15-second-run-87.log
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python vgg_main_lr05_wd0005.py -modelname dropoutregvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 9 --dropout 0.2 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-15/20-8-15-second-run/20-8-15-second-run-88.log
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python vgg_main_lr05_wd0005_tune_corr_reg_vgg_2.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-15/20-8-15-second-run/20-8-15-second-run-94.log
