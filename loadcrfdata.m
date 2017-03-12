function [patterns, labels] = loadcrfdata(path, postfix, cnnflist, cnnprob, ...
dbn33flist, dbn33prob, dbn55flist, dbn55prob, gmmflist, gmmprob, priorprob)
%LOADSSVMDATA Summary of this function goes here
%   Detailed explanation goes here
if sum(cnnprob(:)<0) > 0 || sum(dbn33prob(:)<0) > 0 || sum(dbn55prob(:)<0) > 0 ...
        || sum(gmmprob(:)<0)>0 || sum(priorprob(:)<0) > 0
    display(path);
end
patterns = {} ;
labels = {} ;
filedir = dir([path postfix]);
if length(filedir) ~= 58
  display(path);
end
for i=1:length(filedir)
  fname = filedir(i).name;
  data = load([path fname]);
  data = data.im;
  cnnindex = findstr(cnnflist, fname);
  dbn33index = findstr(dbn33flist, fname);
  dbn55index = findstr(dbn55flist, fname);
  gmmindex = findstr(gmmflist, fname);
  cnndata0 = -log(1-cnnprob(cnnindex, :, :)+1e-6);
  cnndata0 = squeeze(cnndata0);
  cnndata = -log(cnnprob(cnnindex, :, :)+1e-6);
  cnndata = squeeze(cnndata);
  dbn33data0 = -log(1-dbn33prob(dbn33index, :, :)+1e-6);
  dbn33data0 = squeeze(dbn33data0);
  dbn33data = -log(dbn33prob(dbn33index, :, :)+1e-6);
  dbn33data = squeeze(dbn33data);
  dbn55data0 = -log(1-dbn55prob(dbn55index, :, :)+1e-6);
  dbn55data0 = squeeze(dbn55data0);
  dbn55data = -log(dbn55prob(dbn55index, :, :)+1e-6);
  dbn55data = squeeze(dbn55data);
  gmmdata0 = -log(1-gmmprob(gmmindex, :, :)+1e-6);
  gmmdata0 = squeeze(gmmdata0);
  gmmdata = -log(gmmprob(gmmindex, :, :)+1e-6);
  gmmdata = squeeze(gmmdata);
  priordata0 = -log(1-priorprob+1e-6);
  priordata = -log(priorprob+1e-6);
  alldata = ones(40, 40, 7);
  alldata(:,:,1) = data;
  %alldata(:,:,1,2) = data;
  alldata(:,:,2) = ones(40,40);
  alldata(:,:,3) = cnndata;
  alldata(:,:,4) = dbn33data;
  alldata(:,:,5) = dbn55data;
  alldata(:,:,6) = gmmdata;
  alldata(:,:,7) = priordata;
  %patterns{i} = alldata;
  labelarr    = load([path fname(1:end-length(postfix)+1) 'massgt.mat']); % 40 * 40 * 1
  labelarr = labelarr.im;
  %labels{i}   = zeros(40, 40, 2);
  %labels{i}(:,:,1) = 1 - labelarr;
  %labelarr(labelarr==0) = -1;
  x = zeros(40*40, 7);
  y = zeros(40*40,1);
  for j = 1 : 40
      for k = 1 : 40
          y((j-1)*40+k) = labelarr(j,k);
%          x((j-1)*40+k, :) = alldata(j,k, :);
      end
  end
  %if sum(y==0) > 1
  %    display(path);
  %end
  patterns{i} = alldata;
  labels{i} = labelarr; %[-y  y];
  %size(labels{i})
end
end

function [row] = findstr(strarr, str)
  row = 0;
  for i = 1 : size(strarr,1)
    if strcmp(strarr(i,:), str) == 1
      row = i;
      break;
    end
  end
end
