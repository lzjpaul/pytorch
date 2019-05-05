# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncrf1
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_wlm_tune_wd00001_lambda000011_beta8_3.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname reglstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-54.log

# ncrg0