function [ flag ] = binarizelabel( path )
%BINARIZELABEL Summary of this function goes here
%   Detailed explanation goes here
filedir = dir([path '*massgt.jpeg']);
for i = 1 : length(filedir)
    fn = filedir(i).name;
    im = imread([path fn]);
    imwrite(im, [path fn(1:end-5) 'old.jpeg']);
    im = imbinarize(im);
    imwrite(im, [path fn]);
end
flag = 1;
end

