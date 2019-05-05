# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncrb2
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-46.log
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-47.log
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-48.log
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-49.log
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd00001_lambda1_beta_1.py -traindatadir ./data/wikitext-2 -trainlabeldir ./data/wikitext-2 -testdatadir ./data/wikitext-2 -testlabeldir ./data/wikitext-2 -seqnum 35 -modelname lstm -blocks 1 -lr 20.0 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 150 -gpuid 0 --emsize 200 --nhid 200 --clip 0.25 --seed 1111 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-50.log
# ncrb2-1
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd0001000001_lambda00101_beta1.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname reglstm -blocks 1 -lr 0.3 -batchsize 100 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-42.log
CUDA_VISIBLE_DEVICES=2 python train_wlm_tune_wd0001000001_lambda00101_beta8_1.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname reglstm -blocks 1 -lr 0.3 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-43.log


# ncrc0