### panda3-0
CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_resreg_real_wlm_wd0001.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname dropoutreglstm -blocks 1 -lr 1.0 -batchsize 100 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-74.log
CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_resreg_real_wlm_wd0001.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname dropoutreglstm -blocks 1 -lr 1.0 -batchsize 100 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-75.log
CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_resreg_real_wlm_wd0001.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname dropoutreglstm -blocks 1 -lr 1.0 -batchsize 100 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-76.log
CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_resreg_real_wlm_wd0001_tune_maxnorm_14.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname reglstm -blocks 1 -lr 1.0 -batchsize 100 -regmethod 8 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-71.log
CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_resreg_real_wlm_wd0001_tune_corr_reg_lstm_mimic_iii_2345_2.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname reglstm -blocks 1 -lr 1.0 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 | tee -a /hdd2/zhaojing/res-regularization/20-8-28/20-8-28-fourth-run-average/8-28-fourth-run-average-80.log
############################################ 20-8-29-end !!!!!!!!!!!!!! ##############################################