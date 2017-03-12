function [gmmprob, trainf] = convertgmmpred2im(trainfname, trainpredlabel, trainpredprob, gmmtrainpath )
%CONVERTGMMPRED2IM Summary of this function goes here
%   Detailed explanation goes here
%%% check
predlabel = trainpredprob(:,2) > trainpredprob(:,1);
error = sum(trainpredlabel' ~= predlabel);
if error > 0
    display(gmmtrainpath);
end
gmmprob = zeros(58, 40, 40);
ims = trainpredprob(:,2);% > trainlabel(:,1);
imscount = 1;
for i = 1 : size(trainfname, 1)
    fname = trainfname(i,:);
    targetim = zeros(40,40);
    for j = 1 : 40
        for k = 1 : 40
            targetim(j,k) = ims(imscount);
            gmmprob(i,j,k) = ims(imscount);
            imscount = imscount + 1;
        end
    end
    imwrite(targetim, [gmmtrainpath fname '.jpeg']);
end
flag = 1;
trainf = trainfname;
end

