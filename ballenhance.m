function [ flag ] = ballenhance( impath )
%BALLENHANCE Summary of this function goes here
%   Detailed explanation goes here
filedir = dir([impath '*roi.mat']);
for i = 1 : length(filedir)
    fn = filedir(i).name;
    im = load([impath fn]);
    im = im.im;
    im = ballimenhance(im);
    save([impath fn(1:end-4) 'enhance.mat'], 'im');
    imwrite(im, [impath fn(1:end-4) 'enhance.jpeg']);
%     im = load([impath fn(1:end-8) '.mat']);
%     im = im.dcmim;
%     im = ballimenhance(im);
%     save([impath fn(1:end-8) 'enhance.mat'], 'im');
%     im = uint8(im * 255);
%     imwrite(im, [impath fn(1:end-8) 'enhance.jpeg']);
end
flag = 1;
mkdir([impath 'roienhance\']);
movefile([impath '*roienhance.jpeg'], [impath 'roienhance\']);
end

