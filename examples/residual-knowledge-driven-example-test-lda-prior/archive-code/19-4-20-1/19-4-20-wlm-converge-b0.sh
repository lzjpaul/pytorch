# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrc
# ncrb0
CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_resreg_real_wlm_tune_converge.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 2 -lr 20.0 -decay 0.00001 -reglambda 0.0 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 4000 -gpuid 0 --priorbeta 0.0 --emsize 200 --nhid 200 --clip 0.2557 --seed 1111 | tee -a /hdd1/zhaojing/res-regularization/19-4-20/4-20-wlm-converge-4000epoch-3
