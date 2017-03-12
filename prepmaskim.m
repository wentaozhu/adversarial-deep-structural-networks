function [ flag ] = prepmaskim( traintxt, impath, annotpath, trainpath )
%ROIEXTRACT Summary of this function goes here
%   Detailed explanation goes here
fid = fopen(traintxt, 'r');
filename = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);
fid = fopen([trainpath 'dilate.txt'], 'r');
filename = filename{1,1};
for i = 1 : length(filename)
    fn = filename{i,1};
    im = imread([impath fn '.jpeg']);
    [mask, maxpoint, minpoint, exception, nummass] = readxmlmasswise([annotpath fn '.xml'], ...
        size(im,1), size(im,2), 'linear'); %readxmlmasswise linear
    dcmname = dir([impath fn '*.dcm']);
    if length(dcmname) ~= 1
        display(fn);
    end
    dcmim = dicomread([impath dcmname(1).name]);
    if ~isempty(strfind(dcmname(1).name, '_R_'))
        im1 = im;
        dcmim1 = dcmim;
        mask1 = mask;
        for j = 1 : size(im,2)
            im(:,j) = im1(:, size(im,2)+1-j);
            dcmim(:,j) = dcmim1(:, size(im,2)+1-j);
        end
        for k = 1 : nummass
            for j = 1 : size(im,2)
                mask{k}(:, j) = mask1{k}(:, size(im,2)+1-j);
            end
        end
    end
    imwrite(im, [trainpath fn '.jpeg']);
    dicomwrite(dcmim, [trainpath fn '.dcm']);
    im1 = im;
    for k = 1 : nummass
        line = fgetl(fid);
        [str1, str2] = strtok(line, ' ');
        if str2num(str1) ~= str2num(fn)*10 + k
            display(fn);
        end

%         str2 = '5';  %%% write into file
%         fprintf(fid, '%s\n', [num2str(fn) num2str(k) ' ' str2]);
        masksss = mask{k}(:,:); %imdilate(mask{k}(:,:), strel('disk',str2num(str2)));
        im1(masksss==255) = 255;
        imwrite(im1, [trainpath fn num2str(k) 'im+mask.jpeg']);
%         se=strel('disk',5);A=imclose(mask{k}(:,:),se);
%         mask{k}(:,:) = bwfill(mask{k}(:,:), 'hole');
        tmptmp = imfill(masksss, 'hole'); 
        imwrite(tmptmp, [trainpath fn num2str(k) 'mass.jpeg']);
        save([trainpath fn num2str(k) 'mass.mat'], 'tmptmp');
        tmptmp = masksss;%mask{k}(:,:);
        save([trainpath fn num2str(k) 'massori.mat'], 'tmptmp');
    end
end
% fclose(fid);
flag = 1;
mkdir([trainpath 'mass\']);
movefile([trainpath '*mass.jpeg'], [trainpath 'mass\']);
mkdir([trainpath 'im+mask\']);
movefile([trainpath '*im+mask.jpeg'], [trainpath 'im+mask\']);
end