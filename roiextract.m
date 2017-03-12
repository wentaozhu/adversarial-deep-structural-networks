function [ flag ] = roiextract(trainpath, scale )
%ROIEXTRACT Summary of this function goes here
%   Detailed explanation goes here
filedir = dir([trainpath '*massori.mat']);
for i = 1 : length(filedir)
    fn = filedir(i).name(1:end-11);
    im = imread([trainpath 'im+mask\' fn 'im+mask.jpeg']);
    mask = load([trainpath fn 'massori.mat']);
    mask = mask.tmptmp;
    [minx, miny, maxx, maxy] = rectbox(mask);  %%% x is col, y is row
%     maskroi = mask(miny:maxy, minx:maxx);
%     imwrite(maskroi, [trainpath fn 'maskroi.jpeg']);
    mask = uint8(zeros(size(im,1), size(im,2)));
    centerx = (minx+maxx) / 2;
    centery = (miny+maxy) / 2;
    xr = maxx - centerx;
    yr = maxy - centery;
    xr = xr * scale;
    yr = yr * scale;
    minx = int32(centerx-xr);
    miny = int32(centery-yr);
    maxx = int32(centerx+xr);
    maxy = int32(centery+yr);
    if minx <= 0
        minx = 1;
        maxx = int32(2 * xr + 1);
    end
    if miny <= 0
        miny = 1;
        maxy = int32(2 * yr + 1);
    end
    if maxx > size(im,2)
        maxx = size(im,2);
        minx = int32(size(im,2) - 2 * xr);
    end
    if maxy > size(im,1)
        maxy = size(im,1);
        miny = int32(size(im,1) - 2 * yr);
    end
    massgt = load([trainpath fn 'mass.mat']);
    massgt = massgt.tmptmp;
    massgt = massgt(miny:maxy, minx:maxx);
    imwrite(massgt, [trainpath fn 'massgt.jpeg']);
    save([trainpath fn 'massgt.mat'], 'massgt');
    mask(miny, minx:maxx) = 255;
    mask(maxy, minx:maxx) = 255;
    mask(miny:maxy, minx) = 255;
    mask(miny:maxy, maxx) = 255;
    mask = imdilate(mask, strel('disk',10));
    im(mask==255) = 255;
    imwrite(im, [trainpath fn 'im+mask+roi.jpeg']);
    dcmim = dicomread([trainpath fn(1:end-1) '.dcm']);
    roiim = dcmim(miny:maxy, minx:maxx);
    save([trainpath fn 'roi.mat'], 'roiim');
    im = imread([trainpath fn(1:end-1) '.jpeg']);
    roiim = im(miny:maxy, minx:maxx);
    imwrite(roiim, [trainpath fn 'roi.jpeg']);
end
mkdir([trainpath 'im+mask+roi\']);
movefile([trainpath '*im+mask+roi.jpeg'], [trainpath 'im+mask+roi\']);
mkdir([trainpath 'massgt\']);
movefile([trainpath '*massgt.jpeg'], [trainpath 'massgt\']);
mkdir([trainpath 'roi\']);
movefile([trainpath '*roi.jpeg'], [trainpath 'roi\']);
flag = 1;
end