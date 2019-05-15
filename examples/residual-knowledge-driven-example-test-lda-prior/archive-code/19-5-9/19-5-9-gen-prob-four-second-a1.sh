# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncra1
# 45 params, 48 hours
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd0001_lambda00001100_beta000011000_2.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-5-9-results/19-5-9-gen-prob-four-14.log
### MNIST-mlp-regmlp5-regmlp6-best-avg/std --> 24 hours/block
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd0001_lambda1_beta1.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname mlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-5-9-results/19-5-9-gen-prob-four-31.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd0001_lambda1_beta1.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname mlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-5-9-results/19-5-9-gen-prob-four-32.log

# ncrb1