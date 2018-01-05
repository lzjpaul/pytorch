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
