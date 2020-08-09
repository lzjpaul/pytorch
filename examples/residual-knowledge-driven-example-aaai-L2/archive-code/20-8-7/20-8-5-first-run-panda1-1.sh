###### MLP-MNIST-1, panda1-1
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real_mnist_wd0001_tune_maxnorm.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 8 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-63.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real_mnist_wd0001.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname dropoutregmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 --dropout 0.1 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-72.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real_mnist_wd0001.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname dropoutregmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 --dropout 0.2 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-73.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real_mnist_wd0001.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname dropoutregmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 --dropout 0.3 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-74.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real_mnist_wd0001.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname dropoutregmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 --dropout 0.5 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-75.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real_mnist_wd0001.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname dropoutregmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 --dropout 0.7 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-76.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real_mnist_wd0001.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname dropoutregmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 9 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 --dropout 0.9 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-77.log
CUDA_VISIBLE_DEVICES=1 python mlp_residual_hook_resreg_real_mnist_wd0001_tune_corr_reg_1.py -traindatadir MNIST -trainlabeldir MNIST -testdatadir MNIST -testlabeldir MNIST -seqnum 1 -modelname regmlp -blocks 1 -lr 0.01 -batchsize 65 -regmethod 6 -firstepochs 0 -considerlabelnum 1 -maxepoch 200 -gpuid 0 | tee -a /hdd2/zhaojing/res-regularization/20-8-5/20-8-5-first-run/20-8-5-first-run-37.log
