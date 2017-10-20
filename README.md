# adversarial-deep-structural-networks
Adversarial Deep Structural Networks for Mammographic Mass Segmentation https://arxiv.org/abs/1612.05970

crfcnn_combine.py is the executive file.

modelname = 'cnn' #'crfcomb'

modelname = 'cnn' #is for FCN model. 

modelname = 'cnnat' #is for advesarial FCN model. 

modelname = 'crf' #is for joint FCN-CRF model. 

modelname = 'crfat' #is for adversarial FCN-CRF model. 

modelname = 'cnncomb' #is for multi-FCN model.

modelname = 'cnncombat' #is for adversarial multi-FCN model.

modelname = 'crfcomb' #is for joint multi-FCN-CRF model.

modelname = 'crfcombat' #is for adversarual multi-FCN-CRF model.

crfcnn_combine_ddsm.py is the main file for ddsm.

.m files are for reproducing miccai 15 Deep Learning and Structured Prediction for the Segmentation of Mass in Mammograms.

The inbreast dataset can be downloaded from https://drive.google.com/a/uci.edu/file/d/0B-7-8LLwONIZM1djY2pRLWNUemc/view?usp=sharing

The DDSM-BCRP dataset can be downloaded from https://drive.google.com/a/uci.edu/file/d/0B-7-8LLwONIZU0l2N3hXdU96Y2M/view?usp=sharing

Please cite our paper as Zhu, Wentao, and Xiaohui Xie. "Adversarial deep structural networks for mammographic mass segmentation." arXiv preprint arXiv:1612.05970 (2016).

If you have any questions, please contact with me wentaozhu1991@gmail.com.

Supplement code and data in https://drive.google.com/file/d/0B5Hl9mO74DHvUEowa1hyWmVsMmc/view?usp=sharing . Maybe it is helpful for you to reproduce the results.
