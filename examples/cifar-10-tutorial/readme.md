(1) vim ~/.bashrc to check cuda version
cat /usr/local/cuda/version.txt
(2) install
(3) update
https://ptorch.com/news/37.html

conda config --add channels soumith

conda update pytorch torchvision

1-5:
(1) linear layer uses singa initialization
(2) the linear layer weight shape is [out_feature, in_feature]!!

1-6:
(1) the loss function and the backward grad !!! is /(batch_size * label_class)
(2) RNN uses GPU: model, inputs and targets

1-15:
(1) lda_regularizer.py & mimic_mlp_lda.py: LDA_regularization / (label_dim * sample_num) 
(2) the wight shape is inverse to the singa

1-16:
(1) test batchsize needs to be the same as train batchsize

1-17:
(1) adding weight decay manually!!
(2) deal with the scenario that the batchsize can not be divided wholy (init_hidden and inputs can not use batchsize)
(3) init_state() -- for tesing also!

yk:
(1) batchsize: 10
(2) Adam

1-19:
(1) change back to SGD
