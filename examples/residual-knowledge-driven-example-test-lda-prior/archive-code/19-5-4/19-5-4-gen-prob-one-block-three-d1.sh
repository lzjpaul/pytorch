# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncrd1
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd0001_lambda000011_beta8_3.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-9.log


# ncre0