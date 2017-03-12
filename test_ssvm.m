function [w] = test_ssvm(cnntrainf, cnntrainprob, dbn33trainf,...
    dbn33trainprob, dbn55trainf, dbn55trainprob, gmmtrainf, gmmtrainprob, ...
    priorprob, cnntestf, cnntestprob, dbn33testf, dbn33testprob, ...
    dbn55testf, dbn55testprob, gmmtestf, gmmtestprob)
% TEST_SVM_STRUCT_LEARN
%   A demo function for SVM_STRUCT_LEARN(). It shows how to use
%   SVM-struct to learn a standard linear SVM.
  addpath('.\svm-struct-matlab-master\all-mias\GCMex\');
  addpath('D:\git\medicalimage\segmentation\svm-struct-matlab-master\GCMex\');
  randn('state',0) ;
  rand('state',0) ;

  % ------------------------------------------------------------------
  %                                                      Generate data
  % ------------------------------------------------------------------
  trainpath = '.\inbreast\trainroi1.2\';
  testpath = '.\inbreast\testroi1.2\';
  [trainpatterns, trainlabels] = loadssvmdata(trainpath, '*roienhance.mat', ...
cnntrainf, cnntrainprob, dbn33trainf, dbn33trainprob, dbn55trainf, dbn55trainprob,...
gmmtrainf, gmmtrainprob, priorprob);
  [testpatterns, testlabels] = loadssvmdata(testpath, '*roienhance.mat', ...
cnntestf, cnntestprob, dbn33testf, dbn33testprob, dbn55testf, dbn55testprob,...
gmmtestf, gmmtestprob, priorprob);
  % ------------------------------------------------------------------
  %                                                    Run SVM struct
  % ------------------------------------------------------------------

  parm.patterns = trainpatterns ; % 1*58 cell, each cell is 1600*7 double matrix (pixel, 1, cnn, dbn33, dbn55, gmm, prior)
  parm.labels = trainlabels ; % 1*58 cell, each cell is 1600*1 double matrix 0-1 valed
  parm.lossFn = @lossCB ;
  parm.constraintFn  = @constraintCB ;
  parm.featureFn = @featureCB ;
  parm.dimension = 7 ; % 7 unary and 2 pair wise
  parm.verbose = 0; %1 ;
  trainaccarr=zeros(26,1); traindiarr=zeros(26,1); testaccarr=zeros(26,1); testdiarr=zeros(26,1);
 for c=-20:1:5
    model = svm_struct_learn([' -c ' num2str(2^c) ' -o 2 -v 0 '], parm) ; % -c 1.0 -o 1
    %%% test
    trainacc = 0; traindi = 0; pred = zeros(58*1600,1); truelabel = zeros(58*1600,1);
    for i = 1:58
      ypred = constraintCB(parm, model, trainpatterns{i}, trainlabels{i});  %%% label is useless
      pred((i-1)*1600+1:i*1600,1) = ypred;
      truelabel((i-1)*1600+1:i*1600,1) = trainlabels{i};
    end
    trainacc = double(sum(pred(truelabel==1)==1) + sum(pred(truelabel==0)==0))/(58*1600);
    TP = sum(pred(truelabel==1)==1);
    FP = sum(pred(truelabel==0)==1);
    FN = sum(pred(truelabel==1)==0);
    traindi = 2*TP*1.0 / (2*TP+FP+FN);
    testacc = 0; testdi = 0; pred = zeros(58*1600,1); truelabel = zeros(58*1600,1);
    for i = 1:58
      ypred = constraintCB(parm, model, testpatterns{i}, testlabels{i});  %%% label is useless
      pred((i-1)*1600+1:i*1600,1) = ypred;
      truelabel((i-1)*1600+1:i*1600,1) = testlabels{i};
    end
    testacc = (sum(pred(truelabel==1)==1) + sum(pred(truelabel==0)==0))/58*1600;
    TP = sum(pred(truelabel==1)==1);
    FP = sum(pred(truelabel==0)==1);
    FN = sum(pred(truelabel==1)==0);
    testdi = 2*TP*1.0 / (2*TP+FP+FN);
    index = int32((c+21));
    trainaccarr(index,1) = trainacc; traindiarr(index,1) = traindi;
    testaccarr(index,1) = testacc; testdiarr(index,1)=testdi;
    fprintf('testdi %f\n', testdi);
  end
  w = model.w ;
  flag=1;
end

% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------

function delta = lossCB(param, y, ybar)
%   TP = sum(y(ybar==1)==1);
%   FP = sum(y(ybar==0)==1);
%   FN = sum(y(ybar==1)==0);
%   delta = (2*TP+FP+FN) - 2*TP*1.0;
  delta = sum(double(int32(y) ~= int32(ybar))) ;
  if param.verbose
      fprintf('delta = %f\n', delta);
    %fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
  end
end

function psi = featureCB(param, x, y) % param
  y2d = zeros(40,40);  % convert vector y to matrix
  data = zeros(40,40,1); % convert the first dim x into matrix pixel values
  for i = 1 : 40
      for j = 1 : 40
          y2d(i,j) = y((i-1)*40+j);
%           data(i,j,:) = x((i-1)*40+j,1);
      end
  end
  data = x(:,:,1,1);
  %data = x(:,:,1);
  data = squeeze(data);
  unary = zeros(5,1); pieceloss = zeros(2,1);
  for i = 1 : 40
      for j = 1 : 40
          unary = unary + reshape(x(i,j,3:end,-y2d(i,j)+2), [5,1]);
          if i ~= 40 && y2d(i,j)~=y2d(i+1,j)
              pieceloss = pieceloss + [1 ; exp(-(data(i,j)-data(i+1,j))^2)];
          end
          if j ~= 40 && y2d(i,j)~=y2d(i,j+1)
              pieceloss = pieceloss + [1 ; exp(-(data(i,j)-data(i,j+1))^2)];
          end
      end
  end
  psi = sparse([unary ; pieceloss]);
end

function yhat = constraintCB(param, model, x, y) %param
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
  w = (model.w+1e-6);
  y2d = zeros(40,40);  % convert vector y to matrix
  data = zeros(40,40,1); % convert the first dim x into matrix pixel values
  for i = 1 : 40
      for j = 1 : 40
          y2d(i,j) = y((i-1)*40+j);
%           data(i,j,:) = x((i-1)*40+j,1);
      end
  end
  data = x(:,:,1,1);
  %data = x(:,:,1);
  data = squeeze(data);
  priorprob = load('priorprob.mat');
  priorprob = priorprob.priorprob;
  initialim = (priorprob>=0.5); % GCMEX initial use the average image
  segclass = zeros(1,40*40);  % initial class label
  unary = zeros(2,1600); pairloss = zeros(1600,1600); unaryarr= zeros(40,40,2);
  vc = zeros(40,40); hc= zeros(40,40);
  for i = 1 : 40
      for j = 1 : 40
          segclass(1, (i-1)*40+j) = initialim(i,j);
          unary(1,(i-1)*40+j) = reshape(x(i,j,3:end,2), [5,1])'*w(1:5,1);
          unary(2,(i-1)*40+j) = reshape(x(i,j,3:end,1), [5,1])'*w(1:5,1);
          unaryarr(i,j,1) = reshape(x(i,j,3:end,2), [5,1])'*w(1:5,1);
          unaryarr(i,j,2) = reshape(x(i,j,3:end,1), [5,1])'*w(1:5,1);
          if i ~= 40
              vc(i,j) = w(6,1)+w(7,1)*exp(-(data(i,j)-data(i+1,j))^2);
              pairloss((i-1)*40+j,(i)*40+j) = ...
                  w(6,1)+w(7,1)*exp(-(data(i,j)-data(i+1,j))^2);
              pairloss((i)*40+j, (i-1)*40+j) = ...
                  w(6,1)+w(7,1)*exp(-(data(i,j)-data(i+1,j))^2);
          end
          if j~= 40
              hc(i,j) = w(6,1)+w(7,1)*exp(-(data(i,j)-data(i,j+1))^2);
              pairloss((i-1)*40+j,(i-1)*40+j+1) = ...
                  w(6,1)+w(7,1)*exp(-(data(i,j)-data(i,j+1))^2);
              pairloss((i-1)*40+j+1,(i-1)*40+j) = ...
                  w(6,1)+w(7,1)*exp(-(data(i,j)-data(i,j+1))^2);
          end
      end
  end
  %yhat = unary(1,:)' <= unary(2,:)';
  %return;
%   minvalue = min(min(pairloss));
%   if minvalue < 0
%       pairloss(pairloss~=0) = pairloss(pairloss~=0) + abs(minvalue);
%   end
  minvalue = min(min(hc));
  if min(min(vc)) < minvalue
      minvalue = min(min(vc));
  end
  labelcost = [0,1;1,0];
  if minvalue < 0
      unaryarr = unaryarr - minvalue;
      vc = vc - minvalue;
      hc = hc - minvalue;
      labelcost = labelcost * (-minvalue);
  end
      
  pairloss = sparse(pairloss);
  [gch] = GraphCut('open', unaryarr, labelcost, single(vc), single(hc)); 
  [gch, L] = GraphCut('expand', gch ); %//optimize and get the labeling L
  yhat = zeros(1600,1);
  for i = 1 : 40
      for j = 1 : 40
          yhat((i-1)*40+j) = L(i,j);
      end
  end
  gch=GraphCut('close',gch); %// clean up the mess
  %[yhat E Eafter] = GCMex(segclass, single(unary), pairloss, single(labelcost),0);
end