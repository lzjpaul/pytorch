## ncrc
CUDA_VISIBLE_DEVICES=0 python mlp_residual_hook_resreg.py -datadir . -modelname mlp -blocks 4 -decay 0.0 -batchsize 64 -maxepoch 1000 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-1-11/1-11-mlp4-wd-0-1000-epoch
