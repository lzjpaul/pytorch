# CORR-Reg

This implements the CORR-Reg regularization method described at

Adaptive Knowledge Driven Regularization for Deep Neural Networks

## Requirements. 
[PyTorch] (https://github.com/pytorch/pytorch)

[torchvision] (https://github.com/pytorch/vision)


## Usage

All the models have five scripts, correspond to L1-Reg, L2-Reg, Maxnorm, Dropout and CORR-Reg.

MNIST-MLP

```
CUDA_VISIBLE_DEVICES=0 python mnist_mlp_l1_reg_main.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 7 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0
CUDA_VISIBLE_DEVICES=0 python mnist_mlp_l2_reg_main.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname mlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0
CUDA_VISIBLE_DEVICES=0 python mnist_mlp_maxnorm_main.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 8 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0
CUDA_VISIBLE_DEVICES=0 python mnist_mlp_dropout_main.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname dropoutregmlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 --dropout 0.1
CUDA_VISIBLE_DEVICES=0 python mnist_mlp_corr_reg_main.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0
```

MIMIC-III-MLP

```
CUDA_VISIBLE_DEVICES=0 python mimic_iii_mlp_l1_reg_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname regmlp -blocks 1 -lr 0.3 -batchsize 128 -regmethod 7 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first
CUDA_VISIBLE_DEVICES=0 python mimic_iii_mlp_l2_reg_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname mlp -blocks 1 -lr 0.3 -batchsize 128 -regmethod 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first
CUDA_VISIBLE_DEVICES=0 python mimic_iii_mlp_maxnorm_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname regmlp -blocks 1 -lr 0.3 -batchsize 128 -regmethod 8 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first
CUDA_VISIBLE_DEVICES=0 python mimic_iii_mlp_dropout_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname dropoutregmlp -blocks 1 -lr 0.3 -batchsize 128 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --dropout 0.3
CUDA_VISIBLE_DEVICES=0 python mimic_iii_mlp_corr_reg_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname regmlp -blocks 1 -lr 0.3 -batchsize 128 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first
```

Sentence-Polarity-MLP

```
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_mlp_l1_reg_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 7 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_mlp_l2_reg_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname mlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_mlp_maxnorm_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 8 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_mlp_dropout_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname dropoutregmlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --dropout 0.1
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_mlp_corr_reg_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50_mlp.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50_mlp.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50_mlp.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50_mlp.csv -seqnum 25 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 128 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first
```

MIMIC-III-LSTM

```
CUDA_VISIBLE_DEVICES=0 python mimic_iii_lstm_l1_reg_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname reglstm -blocks 1 -lr 1.0 -batchsize 128 -regmethod 7 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128
CUDA_VISIBLE_DEVICES=0 python mimic_iii_lstm_l2_reg_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname lstm -blocks 1 -lr 1.0 -batchsize 128 -regmethod 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128
CUDA_VISIBLE_DEVICES=0 python mimic_iii_lstm_maxnorm_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname reglstm -blocks 1 -lr 1.0 -batchsize 128 -regmethod 8 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128
CUDA_VISIBLE_DEVICES=0 python mimic_iii_lstm_dropout_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname dropoutreglstm -blocks 1 -lr 1.0 -batchsize 128 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 --dropout 0.1
CUDA_VISIBLE_DEVICES=0 python mimic_iii_lstm_corr_reg_main.py -traindatadir MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname reglstm -blocks 1 -lr 1.0 -batchsize 128 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128
```

Sentence-Polarity-LSTM

```
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_lstm_l1_reg_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname reglstm -blocks 1 -lr 0.3 -batchsize 128 -regmethod 7 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_lstm_l2_reg_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname lstm -blocks 1 -lr 0.3 -batchsize 128 -regmethod 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_lstm_maxnorm_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname reglstm -blocks 1 -lr 0.3 -batchsize 128 -regmethod 8 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_lstm_dropout_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname dropoutreglstm -blocks 1 -lr 0.3 -batchsize 128 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128 --dropout 0.3
CUDA_VISIBLE_DEVICES=0 python sentence_polarity_lstm_corr_reg_main.py -traindatadir Movie_Review/correct_movie_review_train_x_seq_word2vec200_window50.csv -trainlabeldir Movie_Review/correct_movie_review_train_y_seq_word2vec200_window50.csv -testdatadir Movie_Review/correct_movie_review_test_x_seq_word2vec200_window50.csv -testlabeldir Movie_Review/correct_movie_review_test_y_seq_word2vec200_window50.csv -seqnum 25 -modelname reglstm -blocks 1 -lr 0.3 -batchsize 128 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gpuid 0 --batch_first --nhid 128
```

MNIST-AE

```
CUDA_VISIBLE_DEVICES=0 python mnist_ae_l1_reg_main.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 7 --dropout 0.5
CUDA_VISIBLE_DEVICES=0 python mnist_ae_l2_reg_main.py -modelname autoenc -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5
CUDA_VISIBLE_DEVICES=0 python mnist_ae_maxnorm_main.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 8 --dropout 0.5
CUDA_VISIBLE_DEVICES=0 python mnist_ae_dropout_main.py -modelname dropoutregautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 9 --dropout 0.1
CUDA_VISIBLE_DEVICES=0 python mnist_ae_corr_reg_main.py -modelname regautoenc -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5
```

MNIST-LeNet

```
CUDA_VISIBLE_DEVICES=0 python mnist_lenet_l1_reg_main.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 7 --dropout 0.5
CUDA_VISIBLE_DEVICES=0 python mnist_lenet_l2_reg_main.py -modelname lenet -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5
CUDA_VISIBLE_DEVICES=0 python mnist_lenet_maxnorm_main.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 8 --dropout 0.5
CUDA_VISIBLE_DEVICES=0 python mnist_lenet_dropout_main.py -modelname dropoutreglenet -firstepochs 0 -considerlabelnum 1 -regmethod 9 --dropout 0.5
CUDA_VISIBLE_DEVICES=0 python mnist_lenet_corr_reg_main.py -modelname reglenet -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5
```

CIFAR-10-VGG

```
CUDA_VISIBLE_DEVICES=0 python cifar_10_vgg_l1_reg_main.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 7 --dropout 0.5 --save-dir=save_vgg16
CUDA_VISIBLE_DEVICES=0 python cifar_10_vgg_l2_reg_main.py -modelname vgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 100 --dropout 0.5 --save-dir=save_vgg16
CUDA_VISIBLE_DEVICES=0 python cifar_10_vgg_maxnorm_main.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 8 --dropout 0.5 --save-dir=save_vgg16
CUDA_VISIBLE_DEVICES=0 python cifar_10_vgg_dropout_main.py -modelname dropoutregvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 9 --dropout 0.3 --save-dir=save_vgg16
CUDA_VISIBLE_DEVICES=0 python cifar_10_vgg_corr_reg_main.py -modelname regvgg16_bn -firstepochs 0 -considerlabelnum 1 -regmethod 6 --dropout 0.5 --save-dir=save_vgg16
```
