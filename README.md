# adversarial-deep-structural-networks
Adversarial Deep Structural Networks for Mammographic Mass Segmentation https://arxiv.org/abs/1612.05970

crfcnn_combine.py is the executive file.
modelname = 'cnn' #'crfcomb'
'cnn' is for FCN model. 
'cnnat' is for advesarial FCN model. 
'crf' is for FCN-CRF model. 
'crfat' is for adversarial FCN-CRF model. 
'cnncomb' is for multi-FCN model.
'cnncombat' is for adversarial multi-FCN model.
'crfcomb' is for multi-FCN-CRF model.
'crfcombat' is for adversarual multi-FCN-CRF model.

crfcnn_combine_ddsm.py is the main file for ddsm.

.m files are for reproducing miccai 15 Deep Learning and Structured Prediction for the Segmentation of Mass in Mammograms.

The inbreast dataset can be downloaded from https://drive.google.com/a/uci.edu/file/d/0B-7-8LLwONIZM1djY2pRLWNUemc/view?usp=sharing

The DDSM-BCRP dataset can be downloaded from https://drive.google.com/a/uci.edu/file/d/0B-7-8LLwONIZU0l2N3hXdU96Y2M/view?usp=sharing

Please cite our paper as Zhu, Wentao, and Xiaohui Xie. "Adversarial deep structural networks for mammographic mass segmentation." arXiv preprint arXiv:1612.05970 (2016).
