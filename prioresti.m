function [ priormat ] = prioresti(trainroipath, testroipath, postfix)
%PRIORESTI Summary of this function goes here
%   Detailed explanation goes here
filedir = dir([trainroipath '*' postfix]);
priormat = zeros(40,40);
if length(filedir) ~= 58
    display(trainroipath);
end
for i = 1 : length(filedir)
    im = load([trainroipath filedir(i).name]);
    im = im.im;
    priormat = priormat + im;
end
priormat = double(priormat) / length(filedir); %58;
imwrite(priormat, 'prior.jpeg');
TP = 0; FN = 0; FP = 0; TN = 0;
for i = 1 : length(filedir)
    im = load([trainroipath filedir(i).name]);
    im = im.im;
    TP = TP + sum(im(priormat>0.5)==1);
    FP = FP + sum(im(priormat>0.5)==0);
    FN = FN + sum(im(priormat<0.5)==1);
    TN = TN + sum(im(priormat<0.5)==0);
end
trainacc = double(TP+TN) / (58*40*40);
traindi = double(2*TP) / (2*TP+FP+FN);
display(['train dice' num2str(traindi) 'train acc' num2str(trainacc)]);
TP = 0; FN = 0; FP = 0; TN = 0;
filedir = dir([testroipath '*' postfix]);
if length(filedir) ~= 58
    display(testroipath);
end
for i = 1 : length(filedir)
    im = load([testroipath filedir(i).name]);
    im = im.im;
    TP = TP + sum(im(priormat>0.5)==1);
    FP = FP + sum(im(priormat>0.5)==0);
    FN = FN + sum(im(priormat<0.5)==1);
    TN = TN + sum(im(priormat<0.5)==0);
end
testacc = double(TP+TN) / (58*40*40);
testdi = double(2*TP) / (2*TP+FP+FN);
display(['train dice' num2str(testdi) 'test acc' num2str(testacc)]);
flag = 1;
end