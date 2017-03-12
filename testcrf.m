function di = testcrf(cnntrainf, cnntrainprob, dbn33trainf,...
    dbn33trainprob, dbn55trainf, dbn55trainprob, gmmtrainf, gmmtrainprob, ...
    priorprob, cnntestf, cnntestprob, dbn33testf, dbn33testprob, ...
    dbn55testf, dbn55testprob, gmmtestf, gmmtestprob)
addpath(genpath('.\svm-struct-matlab-master\JGMT4\JustinsGraphicalModelsToolboxPublic'));
trainpath = '.\inbreast\trainroi1dilbig\';
  testpath = '.\inbreast\testroi1dilbig\';
  [trainpatterns, trainlabels] = loadcrfdata(trainpath, '*roienhance.mat', ...
cnntrainf, cnntrainprob, dbn33trainf, dbn33trainprob, dbn55trainf, dbn55trainprob,...
gmmtrainf, gmmtrainprob, priorprob);
  [testpatterns, testlabels] = loadcrfdata(testpath, '*roienhance.mat', ...
cnntestf, cnntestprob, dbn33testf, dbn33testprob, dbn55testf, dbn55testprob,...
gmmtestf, gmmtestprob, priorprob);
nvals  = 2;  % label number
rho    = .5; % (1 = loopy belief propagation) (.5 = tree-reweighted belief propagation)
% labels are listed as an array of integers 0-7 with negative for unlabeled in a text file
% must convert to our representation of 1-8 with 0 for unlabeled

N = 116;
ims    = cell(N,1);
labels = cell(N,1);

fprintf('loading data and computing feature maps...\n');
for n=1:58
    % load data
    lab = trainlabels{n};
    ims{n}  = reshape(trainpatterns{n}(:,:,1), [40,40]);
    labels0{n} = max(0,lab+1);
    % compute features
    feats{n}  = trainpatterns{n}(:,:,3:7);%featurize_im(ims{n},feat_params);
    % reshape features
    [ly lx lz] = size(feats{n});
    feats{n} = reshape(feats{n},ly*lx,lz);
end 
for n=59:116
    % load data
    lab = testlabels{n-58};
    ims{n}  = reshape(testpatterns{n-58}(:,:,1), [40,40]);
    labels0{n} = max(0,lab+1);
    % compute features
    feats{n}  = testpatterns{n-58}(:,:,3:7);%featurize_im(ims{n},feat_params);
    % reshape features
    [ly lx lz] = size(feats{n});
    feats{n} = reshape(feats{n},ly*lx,lz);
end 
labels = labels0;
% the images come in slightly different sizes, so we need to make many models
% use a "hashing" strategy to not rebuild.  Start with empty giant array
model_hash = repmat({[]},1000,1000);
fprintf('building models...\n')
for n=1:N
    [ly lx lz] = size(ims{n});
    if isempty(model_hash{ly,lx});
        model_hash{ly,lx} = gridmodel(ly,lx,nvals);
    end
end
models = cell(N,1);
for n=1:N
    [ly lx lz] = size(ims{n});
    models{n} = model_hash{ly,lx};
end

fprintf('computing edge features...\n')
edge_params = {{'const'},{'diffthresh'}};%,{'diffthresh'}}; %{'diffthresh'}}; % ,{'pairtypes'}
efeats = cell(N,1);
for n=1:N
    efeats{n} = edgeify_im(ims{n},edge_params,models{n}.pairs,models{n}.pairtype);
end

fprintf('splitting data into a training and a test set...\n')
% split everything into a training and test set

k = 1;
[who_train who_test] = kfold_sets(N,2,k);

ims_train     = ims(who_train);
feats_train   = feats(who_train);
efeats_train  = efeats(who_train);
labels_train  = labels(who_train);
labels0_train = labels0(who_train);
models_train  = models(who_train);

ims_test     = ims(who_test);
feats_test   = feats(who_test);
efeats_test  = efeats(who_test);
labels_test  = labels(who_test);
labels0_test = labels0(who_test);
models_test  = models(who_test);


% visualization function
function viz(b_i)
% here, b_i is a cell array of size nvals x nvars
M = 5;
for n=1:M
    [ly lx lz] = size(ims_train{n});
    subplot(3,M,n    ); miximshow(reshape(b_i{n}',ly,lx,nvals),nvals);
    subplot(3,M,n+  M); imshow(ims_train{n})
    subplot(3,M,n+2*M); miximshow(reshape(labels_train{n},ly,lx),nvals);

end
xlabel('top: marginals  middle: input  bottom: labels')
drawnow
end
    
fprintf('training the model (this is slow!)...\n')
loss_spec = 'trunc_cl_trwpll_5';
crf_type  = 'linear_linear';
di = zeros(10,13);
testypred_ = cell(length(feats_test),1);
bestdi = 0;
for iter = 100%10:1:50 %1000 %: 100 : 1000
    for c = -10:1:2 %20 : 0.5 : 5
%options.viz         = @viz;
options.print_times = 0; % since this is so slow, print stuff to screen
options.gradual     = 1; % use gradual fitting
options.maxiter     = iter; % for minFunc
options.rho         = rho;
options.reg         = 2^c;
options.opt_display = 0;
p = train_crf(feats_train,efeats_train,labels_train,models_train,loss_spec,crf_type,options);
save p p
%p = [];
%load p

% use this to train using all data 
%p = train_crf(feats,efeats,labels,models,loss_spec,crf_type,options);


fprintf('get the marginals for test images...\n');
close all
TP = 0; FP = 0; FN = 0;
for n=1:length(feats_test)
    [b_i b_ij] = eval_crf(p,feats_test{n},efeats_test{n},models_test{n},loss_spec,crf_type,rho);
    
    [ly lx lz] = size(labels_test{n});
    [~,x_pred] = max(b_i,[],1);
    x_pred = reshape(x_pred,ly,lx);
    testypred_{n} = x_pred;
    [ly lx lz] = size(labels0_test{n});
    x       = labels0_test{n};
    % upsample predicted images to full resolution
    %x_pred  = imresize(x_pred,size(x),'nearest');
    E(n)   = sum(x_pred(x(:)>0)~=x(x(:)>0));
    T(n)   = sum(x(:)>0);
    TP = TP + sum(x_pred(x(:)==2)==2);
    FP = FP + sum(x_pred(x(:)==1)==2);
    FN = FN + sum(x_pred(x(:)==2)==1);
%     [ly lx lz] = size(ims_test{n});
%     subplot(2,3,1)
%     miximshow(reshape(b_i',ly,lx,nvals),nvals);
%     subplot(2,3,2)
%     imshow(ims_test{n})
%     subplot(2,3,3)
%     miximshow(reshape(labels_test{n},ly,lx),nvals);
%     
%     [ly lx lz] = size(labels0_test{n});
%     subplot(2,3,4)
%     miximshow(reshape(x_pred,ly,lx),nvals);
%     subplot(2,3,5)
%     imshow(ims_test{n})
%     subplot(2,3,6)
%     miximshow(reshape(labels0_test{n},ly,lx),nvals);
%     drawnow
end
fprintf('total pixelwise error on test data: %f \n', sum(E)/sum(T))
fprintf('dice %f\n', (2*TP*1.0/(2*TP+FP+FN)));
index = int32((iter)/10);
di(index,(c+11)) = (2*TP*1.0/(2*TP+FP+FN));
if di(index, (c+11)) > bestdi
    bestdi = di(index, (c+11));
    save('testpredlabel.mat', 'testypred_');
end
    end
end
max(di(:))
end