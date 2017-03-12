function [ flag ] = mirror( impath, massimpath, massimmirrorpath )
%MIRROR Summary of this function goes here
%   Detailed explanation goes here
filedir = dir([massimpath '*mass.mat']);
for i = 1 : length(filedir)
    filename = filedir(i).name(1:end-8);
    dcmname = dir([impath filename '*.dcm']);
    if length(dcmname) ~= 1
        display(filename);
    end
    if ~isempty(strfind(dcmname(1).name, '_R_'))
        im = imread([massimpath filename '.jpeg']);
        maskorimat = load([massimpath filename 'massori.mat']);
        maskorimat = maskorimat.mask;
        dcmim = dicomread([impath dcmname(1).name]);
        if size(im,1) ~= size(dcmim,1) || size(im,2) ~= size(dcmim,2)
            display(filename);
        end
        im1 = im;
        maskorimat1 = maskorimat;
        dcmim1 = dcmim;
        for j = 1 : size(im,2)
            im(:,j) = im1(:, size(im,2) + 1 - j);
            maskorimat(:,j) = maskorimat1(:,size(im,2)+1-j);
            dcmim(:,j) = dcmim1(:,size(im,2)+1-j);
        end
        imwrite(im, [massimmirrorpath filename '.jpeg']);
        mask = imdilate(maskorimat, strel('disk',10));
        imwrite(mask, [massimmirrorpath filename 'mass.jpeg']);
        save([massimmirrorpath filename 'mass.mat'], 'mask');
        im(mask==255) = 255;
        imwrite(im, [massimmirrorpath filename 'im+mask.jpeg']);
        save([massimmirrorpath filename 'massori.mat'], 'maskorimat');
        dicomwrite(dcmim, [massimmirrorpath filename '.dcm']);
    else
        copyfile([massimpath filename '.jpeg'], massimmirrorpath);
        copyfile([massimpath filename 'mass.mat'], massimmirrorpath);
        copyfile([massimpath filename 'massori.mat'], massimmirrorpath);
%         copyfile([massimpath filename 'im+mask.jpeg'], massimmirrorpath);
        copyfile([impath dcmname(1).name], massimmirrorpath);
    end
    flag = 1;
end