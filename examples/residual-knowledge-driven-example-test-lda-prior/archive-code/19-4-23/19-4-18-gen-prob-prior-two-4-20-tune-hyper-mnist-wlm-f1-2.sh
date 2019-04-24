# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncrf1-2
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_wlm_tune_wd00001_lambda0000110_beta1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname reglstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-31.log
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_wlm_tune_wd00001_lambda0000110_beta1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname reglstm -blocks 2 -lr 20.0 -batchsize 100 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.2557 --seed 1111 | tee -a 19-4-23-results/19-4-18-gen-prob-prior-two-4-20-tune-hyper-mnist-wlm-40.log