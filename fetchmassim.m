function [ flag, nummass ] = fetchmassim( impath, annotpath, massimpath )
%FETCHMASSIM Summary of this function goes here
%   Detailed explanation goes here
filedir = dir([annotpath '*.xml']);
nummass = {};
num_mass_count = 0;
for i = 1 : length(filedir)
    im = imread([impath filedir(i).name(1:end-3) 'jpeg']);
    [mask, maxpoint, minpoint, exception, num_mass] = readxml([annotpath filedir(i).name],...
        size(im,1), size(im,2));
    num_mass_count = num_mass_count + num_mass;
%     nummass = nummass + num_mass;
    if exception == 1
        display(filedir(i).name);
    end
    if num_mass > 0
        nummass = [nummass ; {filedir(i).name, num_mass_count}];
        imwrite(mask, [massimpath filedir(i).name(1:end-4) 'massori.jpeg']);
        save([massimpath filedir(i).name(1:end-4) 'massori.mat'], 'mask');
        mask = imdilate(mask, strel('disk',10));
        imwrite(mask, [massimpath filedir(i).name(1:end-4) 'mass.jpeg']);
        save([massimpath filedir(i).name(1:end-4) 'mass.mat'], 'mask');
        imwrite(im, [massimpath filedir(i).name(1:end-4) '.jpeg']);
        im(mask==255) = 255;
        imwrite(im, [massimpath filedir(i).name(1:end-4) 'im+mask.jpeg']);
    end
end
flag = 1;
end