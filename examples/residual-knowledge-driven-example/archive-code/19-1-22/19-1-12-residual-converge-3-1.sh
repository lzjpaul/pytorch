# ncrb
CUDA_VISIBLE_DEVICES=0 python mlp_residual_hook_resreg.py -datadir . -modelname regmlp -blocks 2 -decay 0.0001 -batchsize 64 -regmethod 0 -firstepochs 0 -maxepoch 2000 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-1-12/1-12-converge-2000epoch-0001-2
