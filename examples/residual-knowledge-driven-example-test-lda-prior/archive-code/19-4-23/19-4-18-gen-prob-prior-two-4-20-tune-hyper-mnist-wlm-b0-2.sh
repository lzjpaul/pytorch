# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncrb0-2
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-25.log
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 2 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.2557 --seed 1111 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-26.log
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-27.log
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 2 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.2557 --seed 1111 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-28.log
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-29.log
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 2 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.2557 --seed 1111 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-30.log