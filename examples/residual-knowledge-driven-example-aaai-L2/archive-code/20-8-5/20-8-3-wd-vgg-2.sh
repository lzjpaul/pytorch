#### home folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-aaai-L2
###panda12
CUDA_VISIBLE_DEVICES=1 python vgg_main_tune_wd_2.py -modelname vgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 --save-dir=save_vgg16 | tee -a /hdd2/zhaojing/res-regularization/20-8-3/8-3-tune-wd-4.log
