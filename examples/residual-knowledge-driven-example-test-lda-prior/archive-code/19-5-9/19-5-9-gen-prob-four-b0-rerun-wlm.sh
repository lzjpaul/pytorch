# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
########################################################### all one hyper-param, can all run in 12 hours!!
# ncrb0
### MNIST-mlp-regmlp6-correlation
## WLM-lstm-reglstm6-correlation
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_cal_corr_mean_var_wd00001_lambda1_beta1_b.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-9-results/19-5-9-gen-prob-four-11.log
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_cal_corr_mean_var_wd00001_lambda001_beta1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname reglstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-9-results/19-5-9-gen-prob-four-12.log
### MNIST-mlp-regmlp5-regmlp6-best-avg/std --> 24 hours/block
CUDA_VISIBLE_DEVICES=0 python mlp_residual_tune_wd0001_lambda1_beta1.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname mlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-5-9-results/19-5-9-gen-prob-four-31.log
CUDA_VISIBLE_DEVICES=0 python mlp_residual_tune_wd0001_lambda1_beta1.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname mlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-5-9-results/19-5-9-gen-prob-four-32.log
###########################################################
# ncra0
