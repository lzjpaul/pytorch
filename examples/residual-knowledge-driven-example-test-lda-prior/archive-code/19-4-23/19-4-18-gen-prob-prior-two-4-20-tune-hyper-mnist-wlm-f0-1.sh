# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncrf0
CUDA_VISIBLE_DEVICES=0 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_tune_wd0001_lambda0000110_beta8_6.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-13.log
CUDA_VISIBLE_DEVICES=0 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_tune_wd0001_lambda0000110_beta8_6.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 2 -lr 0.01 -batchsize 65 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-22.log