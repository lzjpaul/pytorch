18-10-10:
(1) softmax loss
(2) out += residual, no relu after

19-1-22:
(1) save the using |r| version, no baysian approach

19-2-7:
(1) gen_prob: need to divided by (trainnum * labelnum)

19-2-10:
(1) train_hook.py: adding model.py to train.py + adding hook
(2) res_regularizer.py: calculating correlation for different time steps

19-2-11:
(1) every layer has its own correlation_moving_average!!
(2) using main() to organize the code

19-2-12:
(1) mlp_residual_hook_resreg_real.py and train_main_hook_resreg_real.py:
design for healthcare and sentiment analysis
(2) train_lstm_main_hook_resreg_real.py: lstm model is ready

19-2-13:
(1) res_regularizer.py: divide the model paramter into 4 parts

19-2-21:
(1) read in embedding

19-2-22:
(1) for some time steps, if samples are all zeros, then judge whether it is nan or not

19-2-24:
dataset
(1) (using) formal_test_y_seq..: count
(2) (using) formal_test_y_seq_sparse.npz: sparse format
(3) (using) correct_formal_train_x_seq_embedding200_window50_mlp..: correct: mlp is averaged among embeddings
(4) (not using) movie_review_test_x_seq_word2vec200_window50: not correct, because not averaged correctly
