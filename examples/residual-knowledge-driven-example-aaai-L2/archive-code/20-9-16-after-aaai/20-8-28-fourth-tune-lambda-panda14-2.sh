###### LSTM-Movie-Review-1 (16 hours: 28)
### panda14-2
CUDA_VISIBLE_DEVICES=2 /hdd1/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real_wlm_wd00001_tune_corr_reg_lstm_movie_lambda_1.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname reglstm -blocks 1 -lr 0.3 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-tune-lambda/20-8-28-fourth-run-tune-lambda-11.log
############################################ 20-8-29-end !!!!!!!!!!!!!! ##############################################