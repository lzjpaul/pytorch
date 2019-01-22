## ncrb
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg.py -datadir . -modelname mlp -blocks 3 -decay 0.00001 -batchsize 64 -maxepoch 1000 -gpuid 0 | tee -a /hdd1/zhaojing/res-regularization/19-1-11/1-11-mlp3-wd-000001-1000-epoch
