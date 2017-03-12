function [ trainprob, trainf ] = convertdbnpred2im( trainfname, trainlabel, dbn33trainpath )
%CONVERTDBNPRED2IM Summary of this function goes here
%   Detailed explanation goes here
trainlabel = exp(trainlabel) ./ ((exp(trainlabel(:,1))+exp(trainlabel(:,2)))*ones(1,2));
ims = trainlabel(:,2);% > trainlabel(:,1);
trainprob = zeros(58, 40, 40);
imscount = 1;
for i = 1 : size(trainfname, 1)
    fname = trainfname(i,:);
    targetim = zeros(40,40);
    for j = 1 : 40
        for k = 1 : 40
            targetim(j,k) = ims(imscount);
            trainprob(i,j,k) = ims(imscount);
            imscount = imscount + 1;
        end
    end
    imwrite(targetim, [dbn33trainpath fname '.jpeg']);
end
flag = 1;
trainf = trainfname;
end