###### VGG
### panda2-2
CUDA_VISIBLE_DEVICES=2 python vgg_main_lr05_wd0005.py -modelname vgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-139.log
CUDA_VISIBLE_DEVICES=2 python vgg_main_lr05_wd0005.py -modelname vgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-140.log
CUDA_VISIBLE_DEVICES=2 python vgg_main_lr05_wd0005.py -modelname vgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-141.log
########## 20-9-1-rerun end