# ncrb1
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 1 -decay 0.00001 -batchsize 64 -regmethod 5 -firstepochs 3 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-7
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 1 -decay 0.0001 -batchsize 64 -regmethod 5 -firstepochs 3 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-8
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 1 -decay 0.001 -batchsize 64 -regmethod 5 -firstepochs 3 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-9
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 1 -decay 0.01 -batchsize 64 -regmethod 5 -firstepochs 3 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-10
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 1 -decay 0.1 -batchsize 64 -regmethod 5 -firstepochs 3 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-11
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 1 -decay 1.0 -batchsize 64 -regmethod 5 -firstepochs 3 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-12
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 1 -decay 10.0 -batchsize 64 -regmethod 5 -firstepochs 3 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-13