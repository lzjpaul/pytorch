###RegLSTM, blocks=2, lr=0.3
## ncrf1
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.00001 -batchsize 100 -regmethod 0 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-71
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.00001 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-72
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.00001 -batchsize 100 -regmethod 2 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-73
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.00001 -batchsize 100 -regmethod 3 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-74
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.00001 -batchsize 100 -regmethod 4 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-75
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-76
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.0001 -batchsize 100 -regmethod 0 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-77
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.0001 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-78
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.0001 -batchsize 100 -regmethod 2 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-79
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.0001 -batchsize 100 -regmethod 3 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-80
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.0001 -batchsize 100 -regmethod 4 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-81
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python train_lstm_main_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_x_seq_word2vec200_window50.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_train_valid_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/movie_review_test_y_seq.csv -seqnum 25 -modelname reglstm -blocks 2 -lr 0.3 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-22-embedding-one-82