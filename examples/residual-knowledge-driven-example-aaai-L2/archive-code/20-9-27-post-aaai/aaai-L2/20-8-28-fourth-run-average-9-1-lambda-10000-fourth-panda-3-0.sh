### panda3-0
CUDA_VISIBLE_DEVICES=0 python vgg_main_lr05_wd0005_tune_lasso_6.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 7 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-145.log
CUDA_VISIBLE_DEVICES=0 python vgg_main_lr05_wd0005_tune_lasso_6.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 7 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-146.log
CUDA_VISIBLE_DEVICES=0 python vgg_main_lr05_wd0005_tune_lasso_6.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 7 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-147.log
### and VGG remaining 8 machines ...
########## 20-9-1-rerun end