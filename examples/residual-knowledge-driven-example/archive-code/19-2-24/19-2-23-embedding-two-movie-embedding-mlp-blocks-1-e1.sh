## ncre1
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 0 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-23
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-24
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 2 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-25
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 3 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-26
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 4 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-27
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.1 -batchsize 100 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-28
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 0 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-29
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-30
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 2 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-31
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 3 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-32
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 4 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-33
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 1.0 -batchsize 100 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-2-23-embedding-two-movie-embedding-mlp-34