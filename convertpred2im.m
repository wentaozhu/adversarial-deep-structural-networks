function [ trainprob, trainf ] = convertpred2im( trainfname, train, cnntrainpath )
%CONVERTPRED2IM Summary of this function goes here
%   Detailed explanation goes here
ims = train(:,:,:,2);% > train(:,:,:,1);
ims = reshape(ims, [size(ims,1), 40, 40]);
trainprob = ims;
for i = 1 : size(trainfname, 1)
    fname = trainfname(i,:);
    im = reshape(ims(i,:,:,:), [40, 40]);
    imwrite(im, [cnntrainpath fname '.jpeg']);
end
flag = 1;
trainf = trainfname;
end

