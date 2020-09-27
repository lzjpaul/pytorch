### LSTM-Movie-Review, panda1-1
CUDA_VISIBLE_DEVICES=1 python train_lstm_main_hook_resreg_real_wlm_wd00001_vis.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname lstm -blocks 1 -lr 0.3 -batchsize 100 -regmethod 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run/20-8-17-third-run-lstm-movie-wd.log

### LSTM-Movie-Review, panda1-2
CUDA_VISIBLE_DEVICES=2 python train_lstm_main_hook_resreg_real_wlm_wd00001_tune_corr_reg_vis.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname reglstm -blocks 1 -lr 0.3 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run/20-8-17-third-run-lstm-movie-corr.log


### autoencoder, panda9-0
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python autoencoder_MNIST_main_SGD_lr10_wd000001_vis.py -modelname autoenc -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run/20-8-17-third-run-autoenc-wd.log

### autoencoder, panda12-0
CUDA_VISIBLE_DEVICES=0 /hdd1/zhaojing/anaconda3-cuda-10/bin/python autoencoder_MNIST_main_SGD_lr10_wd000001_tune_corr_reg_vis.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-17/20-8-17-third-run/20-8-17-third-run-autoenc-corr.log

