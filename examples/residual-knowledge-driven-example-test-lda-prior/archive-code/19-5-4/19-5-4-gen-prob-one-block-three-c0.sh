# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncrc0
CUDA_VISIBLE_DEVICES=0 python mlp_residual_tune_wd0001_lambda000011_beta8_1.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-7.log
# ncrc0-1
CUDA_VISIBLE_DEVICES=0 python train_wlm_tune_wd0001000001_lambda00101_beta8_2.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname reglstm -blocks 1 -lr 0.3 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-44.log

# ncrc1