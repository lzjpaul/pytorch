###Regmlp, blocks=1
## ncrg1
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 10.0 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-1
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 1.0 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-2
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.5 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-3
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.2 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-4
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.1 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-5
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.05 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-6
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.01 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-7
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.001 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.001 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-8
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 10.0 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-11
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 1.0 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-12
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.5 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-13
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.2 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-14
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.1 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-15
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.05 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-16
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.01 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-17
CUDA_VISIBLE_DEVICES=1 /home/zhaojing/anaconda3-cuda-10/bin/python mlp_residual_hook_resreg_real.py -traindatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabel /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir /hdd1/zhaojing/res-regularization/Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -decay 0.0001 -reglambda 0.01 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --priorbeta 0.001 | tee -a 19-2-27-gen-prob-prior-one-movie-embedding-mlp-18