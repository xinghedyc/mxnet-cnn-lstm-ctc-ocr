# mxnet-cnn-lstm-ctc-ocr
  This repo contains code written by MXNet for ocr tasks, which uses an cnn-lstm-ctc architecture to do text recognition. 

  In addition buctketing module is used in the code to handle variable length of input images. 
# network
  The network in this preject is based on An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. B. Shi， X. Bai， C. Yao .*TPAMI*

paper https://arxiv.org/abs/1507.05717
original code written by torch https://github.com/bgshih/crnn

the main difference is that I use resnet as the cnn part in the architecture.
# prerequisites
1. you should follow steps in official mxnet example/warp-ctc https://github.com/dmlc/mxnet/tree/master/example/warpctc
   to make sure you install warp-ctc correctly and recompile mxnet with warp-ctc plug-in
2. download ICDAR2013 cropped word dataset http://rrc.cvc.uab.es/?ch=2&com=downloads and put it in the right fold 
   which should be consistant with the path in text_deep_ocr_bucketing.py
```
  path='crop_icdar2013_train.lst'
  path_test='crop_icdar2013_val.lst'
  data_root='/cache/icdar2013_word'
  test_root='/cache/icdar2013_word'
```
# train the model
  run this in your terminal:
```
python text_deep_ocr_bucketing.py
```
# results 
  if you use the default setting your should reach obout 40% accuracy in val set.
  To improve performance in future work:
  1. use synthetic 90k to pretrain the model and finetune on ICDAR2013
  2. use data argumentation
