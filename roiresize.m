function [ flag ] = roiresize( trainpath, trainroipath, roiheight, roiwidth )
%ROIRESIZE Summary of this function goes here
%   Detailed explanation goes here
filedir = dir([trainpath '*roi.mat']);
for i = 1 : length(filedir)
    fn = filedir(i).name;
    im = load([trainpath fn]);
    im = im.roiim;
    im = imresize(im, [roiheight, roiwidth]);
    save([trainroipath fn], 'im');
    im = double(im);
    im = im - min(im(:));
    im = im ./ max(im(:));
    imwrite(im,[trainroipath fn(1:end-3) 'jpeg']);
    im = load([trainpath fn(1:end-7) 'massgt.mat']);
    im = im.massgt;
    im = imresize(im, [roiheight, roiwidth]);
    im = imbinarize(im);
    save([trainroipath fn(1:end-7) 'massgt.mat'], 'im');
    imwrite(im,[trainroipath fn(1:end-7) 'massgt.jpeg']);
end
flag = 1;
mkdir([trainroipath 'massgt\']);
movefile([trainroipath '*massgt.jpeg'], [trainroipath 'massgt\']);
mkdir([trainroipath 'roi\']);
movefile([trainroipath '*roi.jpeg'], [trainroipath, 'roi\']);
end