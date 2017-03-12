function [ dice ] = dice( label, prediction )
%DICE Summary of this function goes here
%   label is the true label of size batchsize*40*40, prediction is of the
%   same size as label.
TP = sum(label(prediction==1)==1);
FP = sum(label(prediction==1)==0);
FN = sum(label(prediction==0)==1);
dice = 2*TP*1.0 / (2*TP+FP+FN);
end