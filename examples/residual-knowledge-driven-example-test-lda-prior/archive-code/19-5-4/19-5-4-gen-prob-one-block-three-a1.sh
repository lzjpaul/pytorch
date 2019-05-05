# running folder: /home/zhaojing/residual-knowledge-driven/pytorch/examples/residual-knowledge-driven-example-test-lda-prior
# running machine: ncra-ncrh
# ncra1
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd000000100001_lambda1_beta_1.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname mlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-10.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd000000100001_lambda1_beta_1.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname mlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-11.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd000000100001_lambda1_beta_1.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname mlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-12.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd000000100001_lambda1_beta_1.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname mlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-13.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd000000100001_lambda1_beta_1.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname mlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 1 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-14.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd0001000001_lambda00001001_beta1.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname regmlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 5 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-15.log
# ncra1-1
CUDA_VISIBLE_DEVICES=1 python mlp_residual_tune_wd0001000001_lambda00001001_beta8_2.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname regmlp -blocks 1 -lr 0.3 -batchsize 100 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first | tee -a 19-5-4-results/19-5-4-gen-prob-prior-three-17.log