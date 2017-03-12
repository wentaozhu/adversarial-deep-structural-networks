clear all; close all; clc;
cnnmodel = load('.\crf_cnn_adv\1dilbig\0.00370001.01dilbiglrfsave5cnnnorm.mat');
cnnatmodel = load('.\crf_cnn_adv\1dilbig\0.00370001.01dilbiglrfsave5cnnatnorm.mat');
%crfatmodel = load('.\crf_cnn_adv\1dilbig\cnncrfattestpred.mat');
trainfname = cnnatmodel.testfname;
trainlabel = cnnatmodel.testlabel; % 58 40 40
cnncrfattestpred = load('.\crf_cnn_adv\1dilbig\cnncrfattestpred.mat');
cnncrfattestpred = cnncrfattestpred.predlabel;

miccaitestfname = load('inbreasttestfilename.txt');
miccaitruelabel = load('testtruelabel.mat');
miccaitruelabel = miccaitruelabel.testlabels; % 1 58 cell
miccaipredlabel = load('testpredlabel.mat'); 
miccaipredy = miccaipredlabel.testypred_; % 58 1 cell

truelabel = zeros(58*40*40, 1);
miccaipredlabel = zeros(58*40*40, 1);
cnncrfatpredlabel = zeros(58*40*40, 1);

indexmap = zeros(58,1);
for i = 1 : 58
    for j = 1 : 58
        if miccaitestfname(j,:) == str2double(trainfname(i,1:length('300115071')))
            indexmap(i,1) = j;
        end
    end
end

for i = 1 : 58
    if miccaitestfname(indexmap(i,1),:) ~= str2double(trainfname(i,1:length('300115071')))
        miccaitestfname(indexmap(i,1),:)
        str2double(trainfname(i,1:length('300115071')))
    end
    for j = 1 : 40
        for k = 1 : 40
            if miccaitruelabel{1,indexmap(i,1)}(j,k) ~= trainlabel(i,j,k)
                indexmap(i,1)
                i
                j
                k
                miccaitruelabel{1,indexmap(i,1)}(j,k)
                trainlabel(i,j,k)
            end
            truelabel((i-1)*40*40+(j-1)*40+k, 1) = trainlabel(i,j,k);
            cnncrfatpredlabel((i-1)*40*40+(j-1)*40+k, 1) = cnncrfattestpred(i,j,k);
            miccaipredlabel((i-1)*40*40+(j-1)*40+k, 1) = miccaipredy{indexmap(i,1),1}(j,k);
        end
    end
end

c01 = 0; c10 = 0; c00 = 0; c11 = 0;
for i = 1 : 92800
    if truelabel(i,1) ~= miccaipredlabel(i,1)-1
        if truelabel(i,1) == cnncrfatpredlabel(i,1)
            c01 = c01 + 1;
        else
            c11 = c11 + 1;
        end
    elseif truelabel(i,1) == cnncrfatpredlabel(i,1)
        c00 = c00 + 1;
    end
    if truelabel(i,1) ~= cnncrfatpredlabel(i,1)
        if truelabel(i,1) == miccaipredlabel(i,1)-1
            c10 = c10 + 1;
        end
    end
end
k = min(c01, c10);
p = 0;
evalin(symengine, 'binomial(60, 30) / 2^60')
for j = 0 : k
    p = p + nchoosek(c01+c10,j);
end
p = p*1.0 / (2 ^ (c01+c10-1))