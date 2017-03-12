clear all; close all; clc;
%% image path and annotations
%%% separate into training test, training for cross validation parameter
%%% selection and model learning. Do not convert to jpeg
impath = '.\inbreast\im\';  %%% jpg not alignment
annotpath = '.\inbreast\xml\';
massimpath = '.\inbreast\massim\';
massimmirrorpath = '.\inbreast\massimmirror\';
% [flag, nummass] = fetchmassim(impath, annotpath, massimpath);
% mkdir([massimpath 'im+mask\']);
% movefile([massimpath '*im+mask.jpeg'], [massimpath 'im+mask\']);
% mkdir([massimpath 'mass\']);
% movefile([massimpath '*mass.jpeg'], [massimpath 'mass\']);
% mkdir([massimpath 'massori\']);
% movefile([massimpath '*massori.jpeg'], [massimpath 'massori\']);
% mirror(impath, massimpath, massimmirrorpath);
% mkdir([massimmirrorpath 'mass\']);
% movefile([massimmirrorpath '*mass.jpeg'], [massimmirrorpath 'mass\']);
% mkdir([massimmirrorpath 'im+mask\']);
% movefile([massimmirrorpath '*im+mask.jpeg'], [massimmirrorpath 'im+mask\']);
%%% manually separate train/test based on nummass by case independent
traintxt = '.\inbreast\train.txt';
testtxt = '.\inbreast\test.txt';
trainpath = '.\inbreast\train\'; %_line\';
testpath = '.\inbreast\test\'; %_line\';
%% Use the annotation, generate the roi, 2 scale than the original annotation, same center
%%% The scale is related to user setting
%%% In IJPRAI15, they generated from the annotated bounding box (BB) of 
%%% each mass, by expanding the BB by 20%
%%% Here we use the same setting as ICIP15, two times, that is, width and
%%% height is sqrt(2) times of original.
scale = 1%2%1%.2; %1
% prepmaskim(traintxt, impath, annotpath, trainpath);
% prepmaskim(testtxt, impath, annotpath, testpath);
% roiextract(trainpath, scale);
% roiextract(testpath, scale);
%% resize roi 40 * 40, use ball image enhance techique to process the roi
roiheight = 40;
roiwidth = 40;
trainroipath = '.\inbreast\trainroi1dilbig\'; %'.\inbreast\trainroi1dil\'; %'.\BCRP\bcrpseg\dataset\train1\';%'.\inbreast\trainroi1dil\'; %_line1nn\'; %_line
testroipath = '.\inbreast\trainroi1dilbig\'; %'.\inbreast\testroi1dil\'; %'.\inbreast\testroi1\'; %2 _line1nn\';
% roiresize(trainpath, trainroipath, roiheight, roiwidth);
% roiresize(testpath, testroipath, roiheight, roiwidth);
% ballenhance(trainroipath);
% ballenhance(testroipath);
%% CNN potential function
%%% 5*5 maxpooling 5*5 maxpooling 588 40*40, more than one cnn will
%%% overfitting, the number of filters 6, 12??
di = zeros(500,3,3);
for i = 70 %70 %0 : 10 : 4990
    %load(['.\inbreast\' num2str(i) 'savenorm.mat']);
%load('0.005cnnpredenhancematconvbias1.2norevrevnorm.mat'); %0.0025cnnpredenhancematconv1norevrev_linennnorm.mat');%'0.005cnnpredenhancematconvbias1.2norevrevnorm.mat');
load('dilbigcnnpredfc.mat');
cnntrainpath = '.\inbreast\cnnout\train\';
cnntestpath = '.\inbreast\cnnout\test\';
[cnntrainprob, cnntrainf] = convertpred2im(trainfname, train, cnntrainpath);
[cnntestprob, cnntestf] = convertpred2im(testfname, test, cnntestpath);
% figure
% plot(trainloss, 'r');
% hold on
% plot(testloss, 'b');
% figure
% plot(trainacc, 'r');
% hold on
% plot(testacc, 'b');
% figure
% plot(traindi, 'r');
% hold on
% plot(testdi, 'b');
%% One DBN potential function
%%% LRF 3*3, first use cd pretraining, then bp fine tune. Three hidden
%%% layers with 50 nodes in each layer.
load('dbn33dilbig.mat'); %dbn33line.mat'); %dbn33new
dbn33trainpath = '.\inbreast\dbn33out\train\';
dbn33testpath = '.\inbreast\dbn33out\test\';
[dbn33trainprob, dbn33trainf] = convertdbnpred2im(trainfname, trainlabel, dbn33trainpath);
[dbn33testprob, dbn33testf] = convertdbnpred2im(testfname, testlabel, dbn33testpath);
% figure, plot(testdils(4:4:400), 'r');
% hold on
% plot(traindils(2:4:400), 'b');
% figure, plot(testaccls(3:4:400), 'r');
% hold on
% plot(trainaccls(1:4:400), 'b');
%% Second DBN potential function
%%% LRF 5*5 the same as before
load('dbn55dilbig'); %'dbn55line.mat');
dbn55trainpath = '.\inbreast\dbn55out\train\';
dbn55testpath = '.\inbreast\dbn55out\test\';
[dbn55trainprob, dbn55trainf] = convertdbnpred2im(trainfname, trainlabel, dbn55trainpath);
[dbn55testprob, dbn55testf] = convertdbnpred2im(testfname, testlabel, dbn55testpath);
% figure, plot(testdils(4:4:320), 'r');
% hold on
% plot(traindils(2:4:320), 'b');
% figure, plot(testaccls(3:4:320), 'r');
% hold on
% plot(trainaccls(1:4:320), 'b');
%% GMM potential function
%%% prior p() = 0.5, pixel wise GMM, EM estimate the mu, sigma
%%% The number of components?? cross validation???
load('gmmline.mat');%line.mat');
gmmtrainpath = '.\inbreast\gmmout\train\';
gmmtestpath = '.\inbreast\gmmout\test\';
[gmmtrainprob, gmmtrainf] = convertgmmpred2im(trainfname, trainpredlabel, trainpredprob, gmmtrainpath);
[gmmtestprob, gmmtestf] = convertgmmpred2im(testfname, testpredlabel, testpredprob, gmmtestpath);
%% Shape location size potential function
%%% Just empiracal estimation on the training roi
priorprob = prioresti(trainroipath, testroipath, 'massgt.mat');
save('priorprob.mat', 'priorprob');
%% label transition penalty potential function
%%% 1 - delta(y(i)-y(j))

%% contrast penalty
%%% (1-delta(y(i)-y(j))) C(x(i)-x(j))

%% structure svm train and inference
addpath('.\svm-struct-matlab-master\');
%di(i/10+1,:,:) = testcrf(cnntrainf, cnntrainprob, dbn33trainf,...
%    dbn33trainprob, dbn55trainf, dbn55trainprob, gmmtrainf, gmmtrainprob, ...
%    priorprob, cnntestf, cnntestprob, dbn33testf, dbn33testprob, ...
%    dbn55testf, dbn55testprob, gmmtestf, gmmtestprob); %test_ssvm
di=testcrf(cnntrainf, cnntrainprob, dbn33trainf,...
    dbn33trainprob, dbn55trainf, dbn55trainprob, gmmtrainf, gmmtrainprob, ...
    priorprob, cnntestf, cnntestprob, dbn33testf, dbn33testprob, ...
    dbn55testf, dbn55testprob, gmmtestf, gmmtestprob); %test_ssvm
end