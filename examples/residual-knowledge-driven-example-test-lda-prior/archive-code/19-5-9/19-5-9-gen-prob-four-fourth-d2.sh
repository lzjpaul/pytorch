##################################################### not run yet !!! --> no GPU memory!!
# ncrd2
# 9 params, 24 hours
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd00001_lambda00001100_beta000011000_5.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname reglstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 80 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-9-results/19-5-9-gen-prob-four-27.log
### WLM-lstm-reglstm5-reglstm6-best-avg/std --> 24 hours/block
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd00001_lambda1_beta1_b.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 80 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-9-results/19-5-9-gen-prob-four-53.log
#ncrc0