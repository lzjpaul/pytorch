# ncrc2
CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 3 -decay 0.00001 -batchsize 64 -regmethod 5 -firstepochs 0 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-28
CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 3 -decay 0.0001 -batchsize 64 -regmethod 5 -firstepochs 0 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-29
CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 3 -decay 0.001 -batchsize 64 -regmethod 5 -firstepochs 0 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-30
CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 3 -decay 0.01 -batchsize 64 -regmethod 5 -firstepochs 0 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-31
CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 3 -decay 0.1 -batchsize 64 -regmethod 5 -firstepochs 0 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-32
CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 3 -decay 1.0 -batchsize 64 -regmethod 5 -firstepochs 0 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-33
CUDA_VISIBLE_DEVICES=2 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 3 -decay 10.0 -batchsize 64 -regmethod 5 -firstepochs 0 -labelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-2-7/2-7-tune-param-34