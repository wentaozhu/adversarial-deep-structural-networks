import tensorflow as tf
import scipy
import numpy as np 
import scipy.io as sio
import os
import scipy.misc
rng = np.random.RandomState(1234)
tf.set_random_seed(1)
boxheight = 40
boxwidth = 40
batchsize = 29
postfix='roienhance.mat' # 'roienhance.mat' 'roienhance.jpeg'
traindatapath = '../inbreast/trainroi1.2/'
trainlabelpath = '../inbreast/trainroi1.2/'
testdatapath = '../inbreast/testroi1.2/'
testlabelpath = '../inbreast/testroi1.2/'

def meanstd(x, meanx, stdx):
  x -= meanx
  x /= stdx
  return x

def fetchdatalabel(path, postfix='roienhance.mat', flag='train'):  # 'enhance.mat' 'roienhance.jpeg'
  data = np.zeros((58, 40, 40))
  label = np.zeros((58, 40, 40))
  if flag == 'train':
    data = np.zeros((58*4, 40, 40))
    label = np.zeros((58*4, 40, 40))
  datacount = 0
  fname = []
  for file in os.listdir(path):
    if file.endswith(postfix):
      if postfix[-4:] == '.mat':
        im = sio.loadmat(path+file)
        im = im['im']
      elif postfix[-5:] == '.jpeg':
        im = scipy.misc.imread(path+file)
        im = im*1.0 / 255.0
      imlabel = sio.loadmat(path+file[:-len(postfix)]+'massgt.mat')
      imlabel = imlabel['im']
      data[datacount, :, :] = im
      label[datacount, :, :] = imlabel
      datacount += 1
      if flag == 'train':
        data[datacount, :, :] = im[:, ::-1]
        label[datacount, :, :] = imlabel[:, ::-1]
        data[datacount+1, :, :] = im[::-1, :]
        label[datacount+1, :, :] = imlabel[::-1, :]
        im1 = im[::-1, :]  # vertical flip, then horizontal flip
        imlabel1 = imlabel[::-1, :]
        data[datacount+2, :, :] = im1[:, ::-1]
        label[datacount+2, :, :] = imlabel1[:, ::-1] 
        datacount += 3
      fname.append(file)
  if flag == 'train': assert(datacount==58*4)
  else: assert(datacount==58)
  return data , label, fname

def dice(label, pred):
  TP = np.sum(pred[label==1]==1)
  FP = np.sum(label[pred==1]==0)
  FN = np.sum(label[pred==0]==1)
  return 2*TP*1. / (FP+FN+2*TP)

def dice_tf(label, pred):
  TP = tf.reduce_sum(tf.mul(pred, label))
  FP = tf.reduce_sum(tf.mul(pred, 1-label))
  FN = tf.reduce_sum(tf.mul(1-pred, label))
  return tf.truediv(2*TP,  FP+FN+2*TP)

def convlayer(x, w, b, flag='stride'):
  assert(flag=='stride' or flag=='maxpool')
  strides = [1,1,1,1]
  if flag == 'stride':
    strides = [1,2,2,1]
  hconv1 = tf.nn.conv2d(x, w, strides=strides, padding='VALID')
  hconv1bias = tf.nn.bias_add(hconv1, b)
  hconv1tan = tf.nn.tanh(hconv1bias)
  if flag == 'maxpool':
    hconv1pool = tf.nn.max_pool(hconv1tan, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    return hconv1pool
  return hconv1

def crfrnn(ux, wsmooth, wcontra, k1, k2, trainiter=5, testiter=10, wunary=None): 
  '''Here we use the pot model and only consider grid model. The feature used is only pixel x
  ux is (boxheight * boxwidth) * 2 (here we only have 2 labels). wsmooth, wcontra is 2*1 matrix.
  k1, k2 are filters of sizes (boxheight * boxwidth) * (boxheight*boxwidth)'''
  if wunary is None:
    q = ux # ux has been softmaxed tf.exp(ux - tf.reduce_max(ux, reduction_indices=2, keep_dims=True))
  else:
    ux = tf.reshape(ux, [-1, boxheight, boxwidth, 8])
    wunary = tf.reshape(wunary, [8,])
    h = tf.mul(ux, wunary)
    h = tf.reshape(h, [-1, boxheight, boxwidth, 4, 2])
    hsum = tf.reduce_sum(h, reduction_indices=3)
    hconv4bias = tf.reshape(hsum, [-1,2])
    hconv4soft = tf.nn.softmax(hconv4bias)
    hconv4clip = tf.clip_by_value(hconv4soft, 1e-6, 1.)
    q = (tf.reshape(hconv4clip, [-1, boxheight, boxwidth, 2]))
    #e_x = tf.exp(hsum - tf.reduce_max(hsum, reduction_indices=2, keep_dims=True))
  #q = e_x #/ (tf.clip_by_value(tf.reduce_sum(e_x, reduction_indices=2, keep_dims=True), 1e-6,1.))  # batchsize*(boxheight*boxwidth)*2
  k1 = tf.reshape(k1, [-1, boxheight*boxwidth, boxheight*boxwidth, 1])
  k1 = tf.concat(3, [k1, k1])
  k2 = tf.reshape(k2, [-1, boxheight*boxwidth, boxheight*boxwidth, 1])
  k2 = tf.concat(3, [k2, k2])
  wsmooth = tf.diag(tf.squeeze(wsmooth))
  wcontra = tf.diag(tf.squeeze(wcontra))
  #wsmoothaug = tf.reshape(wsmooth, [1,1,2])
  #wsmoothaug = tf.tile(wsmoothaug, [batchsize, boxheight*boxwidth, 1])
  #wcontraaug = tf.reshape(wcontra, [1,1,2])
  #wcontraaug = tf.tile(wcontraaug, [batchsize, boxheight*boxwidth, 1])
  for epoch in range(testiter):
    q = tf.reshape(q, [-1, boxheight*boxwidth, 1, 2]) 
    q = tf.tile(q, [1,1,boxheight*boxwidth,1])
    q1 = tf.reduce_sum(tf.mul(k1, q), reduction_indices=2)  # for pairwise potential 1, batchsize*(boxheight*boxwidth)*2
    q2 = tf.reduce_sum(tf.mul(k2, q), reduction_indices=2)  # for pairwise potential 2, batchsize*(boxheight*boxwidth)*2
    q1 = tf.reshape(q1, [-1,2])
    q2 = tf.reshape(q2, [-1,2])
    #qpre = tf.mul(wsmoothaug, q1) + tf.mul(wcontraaug, q2)
    qpre = tf.matmul(q1, wsmooth) + tf.matmul(q2, wcontra)
    #qpre = tf.reshape(qpre, [-1,2])
    qhat = tf.matmul(qpre, tf.constant(np.array([[0,1],[1,0]]).astype('float32')))
    qhat = tf.reshape(qhat, [-1, boxheight, boxwidth, 2])
    if wunary is None:
      qu = ux - qhat
    else:
      ux = tf.reshape(ux, [-1, boxheight, boxwidth,8])
      #wunary = tf.reshape(wunary, [8,])
      h = tf.mul(ux, wunary)
      h = tf.reshape(h, [-1, boxheight, boxwidth, 4, 2])
      hsum = tf.reduce_sum(h, reduction_indices=3)
      qu = hsum - qhat
    #e_xx = tf.exp(qu - tf.reduce_max(qu, reduction_indices=2, keep_dims=True))
    #q = e_xx / (tf.clip_by_value(tf.reduce_sum(e_xx, reduction_indices=2, keep_dims=True), 1e-6, 1.))
    qubias = tf.reshape(qu, [-1,2])
    qusoft = tf.nn.softmax(qubias)
    quclip = tf.clip_by_value(qusoft, 1e-6, 1.)
    q = (tf.reshape(quclip, [-1, boxheight, boxwidth, 2]))
    if epoch+1 == trainiter:
      trainq = q
  return trainq, q

def calfilter(X):
  '''X is nbatch*boxheight*boxwidth image. k1 and k2 is the nbatch*(boxheight*boxwidth)*(boxheight*boxwidth)
  filters. Here we only consider 4 neigbor regeion.'''
  k1 = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[1], X.shape[2]))
  k2 = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[1], X.shape[2]))
  for i in range(X.shape[1]):
    for j in range(X.shape[2]):
      if i != 0:
        k1[:,i,j,i-1,j] = 1
        k2[:,i,j,i-1,j] = np.exp(-(X[:,i,j]-X[:,i-1,j])**2)
      if i != X.shape[1]-1:
        k1[:,i,j,i+1,j] = 1
        k2[:,i,j,i+1,j] = np.exp(-(X[:,i,j]-X[:,i+1,j])**2)
      if j != 0:
        k1[:,i,j,i,j-1] = 1
        k2[:,i,j,i,j-1] = np.exp(-(X[:,i,j]-X[:,i,j-1])**2)
      if j != X.shape[2]-1:
        k1[:,i,j,i,j+1] = 1
        k2[:,i,j,i,j+1] = np.exp(-(X[:,i,j]-X[:,i,j+1])**2)
  k1 = k1.reshape((X.shape[0], X.shape[1]*X.shape[2], X.shape[1]*X.shape[2]))
  k2 = k2.reshape((X.shape[0], X.shape[1]*X.shape[2], X.shape[1]*X.shape[2]))
  return k1, k2

def init(nhid1=6, nhid2=12, nhid3=588, lrf1=5, lrf2=5, lrf3=7):  #nhid1=6, nhid2=12, nhid3=588, lrf1=5, lrf2=5, lrf3=7
  paras = {'wconv1': tf.Variable(tf.random_normal([lrf1, lrf1, 1, nhid1])),
           'wconv2': tf.Variable(tf.random_normal([lrf2, lrf2, nhid1, nhid2])),
           'wconv3': tf.Variable(tf.random_normal([lrf3, lrf3, nhid2, nhid3])),
           'wconv4': tf.Variable(tf.random_normal([boxheight, boxwidth, 2, nhid3])),
           'bconv1': tf.Variable(tf.random_normal([nhid1])),
           'bconv2': tf.Variable(tf.random_normal([nhid2])),
           'bconv3': tf.Variable(tf.random_normal([nhid3])),
           'bconv4': tf.Variable(tf.random_normal([boxheight*boxwidth*2])),
           'wsmooth': tf.Variable(tf.random_normal([2,1])),
           'wcontra': tf.Variable(tf.random_normal([2,1]))}
  return paras
def initcomb(nhid221=37, nhid222=12, nhid223=355, lrf221=2, lrf222=2, lrf223=9,
      nhid331=16, nhid332=13, nhid333=415, lrf331=3, lrf332=3, lrf333=8,
      nhid441=9, nhid442=12, nhid443=588, lrf441=4, lrf442=4, lrf443=7,
      nhid551=6, nhid552=12, nhid553=588, lrf551=5, lrf552=5, lrf553=7, fusion=None):
  paras = {'wconv221': tf.Variable(tf.random_normal([lrf221, lrf221, 1, nhid221])),
           'wconv222': tf.Variable(tf.random_normal([lrf222, lrf222, nhid221, nhid222])),
           'wconv223': tf.Variable(tf.random_normal([lrf223, lrf223, nhid222, nhid223])),
           'wconv224': tf.Variable(tf.random_normal([boxheight, boxwidth, 2, nhid223])),
           'bconv221': tf.Variable(tf.random_normal([nhid221])),
           'bconv222': tf.Variable(tf.random_normal([nhid222])),
           'bconv223': tf.Variable(tf.random_normal([nhid223])),
           'bconv224': tf.Variable(tf.random_normal([boxheight*boxwidth*2])),
           'wconv331': tf.Variable(tf.random_normal([lrf331, lrf331, 1, nhid331])),
           'wconv332': tf.Variable(tf.random_normal([lrf332, lrf332, nhid331, nhid332])),
           'wconv333': tf.Variable(tf.random_normal([lrf333, lrf333, nhid332, nhid333])),
           'wconv334': tf.Variable(tf.random_normal([boxheight, boxwidth, 2, nhid333])),
           'bconv331': tf.Variable(tf.random_normal([nhid331])),
           'bconv332': tf.Variable(tf.random_normal([nhid332])),
           'bconv333': tf.Variable(tf.random_normal([nhid333])),
           'bconv334': tf.Variable(tf.random_normal([boxheight*boxwidth*2])),
           'wconv441': tf.Variable(tf.random_normal([lrf441, lrf441, 1, nhid441])),
           'wconv442': tf.Variable(tf.random_normal([lrf442, lrf442, nhid441, nhid442])),
           'wconv443': tf.Variable(tf.random_normal([lrf443, lrf443, nhid442, nhid443])),
           'wconv444': tf.Variable(tf.random_normal([boxheight, boxwidth, 2, nhid443])),
           'bconv441': tf.Variable(tf.random_normal([nhid441])),
           'bconv442': tf.Variable(tf.random_normal([nhid442])),
           'bconv443': tf.Variable(tf.random_normal([nhid443])),
           'bconv444': tf.Variable(tf.random_normal([boxheight*boxwidth*2])),
           'wconv551': tf.Variable(tf.random_normal([lrf551, lrf551, 1, nhid551])),
           'wconv552': tf.Variable(tf.random_normal([lrf552, lrf552, nhid551, nhid552])),
           'wconv553': tf.Variable(tf.random_normal([lrf553, lrf553, nhid552, nhid553])),
           'wconv554': tf.Variable(tf.random_normal([boxheight, boxwidth, 2, nhid553])),
           'bconv551': tf.Variable(tf.random_normal([nhid551])),
           'bconv552': tf.Variable(tf.random_normal([nhid552])),
           'bconv553': tf.Variable(tf.random_normal([nhid553])),
           'bconv554': tf.Variable(tf.random_normal([boxheight*boxwidth*2])),
           'wunary': tf.Variable(tf.random_normal([4,2]))}
  if fusion=='late':
    paras['wsmooth22'] = tf.Variable(tf.random_normal([2,1]))
    paras['wcontra22'] = tf.Variable(tf.random_normal([2,1]))
    paras['wsmooth33'] = tf.Variable(tf.random_normal([2,1]))
    paras['wcontra33'] = tf.Variable(tf.random_normal([2,1]))
    paras['wsmooth44'] = tf.Variable(tf.random_normal([2,1]))
    paras['wcontra44'] = tf.Variable(tf.random_normal([2,1]))
    paras['wsmooth55'] = tf.Variable(tf.random_normal([2,1]))
    paras['wcontra55'] = tf.Variable(tf.random_normal([2,1]))
  else:
    paras['wsmooth'] = tf.Variable(tf.random_normal([2,1]))
    paras['wcontra'] = tf.Variable(tf.random_normal([2,1]))
  return paras

def buildmodel(X, paras):
  hconv1 = convlayer(X, paras['wconv1'], paras['bconv1'], flag='maxpool')
  hconv2 = convlayer(hconv1, paras['wconv2'], paras['bconv2'], flag='maxpool')
  
  hconv3 = tf.nn.conv2d(hconv2, paras['wconv3'], strides=[1,1,1,1], padding='VALID')
  hconv3bias = tf.nn.bias_add(hconv3, paras['bconv3'])
  hconv3tan = tf.nn.tanh(hconv3bias)

  hconv4 = tf.nn.conv2d_transpose(hconv3tan, paras['wconv4'], [batchsize,boxheight,boxwidth,2], 
                                  strides=[1,1,1,1], padding='VALID')
  hconv4 = tf.reshape(hconv4, [-1,boxheight*boxwidth*2])
  hconv4bias = tf.nn.bias_add(hconv4, paras['bconv4'])
  hconv4bias = tf.reshape(hconv4bias, [-1, boxheight, boxwidth, 2])
  hconv4bias = tf.reshape(hconv4bias, [-1,2])
  hconv4soft = tf.nn.softmax(hconv4bias)
  hconv4clip = tf.clip_by_value(hconv4soft, 1e-6, 1.)
  hconv4clip = (tf.reshape(hconv4clip, [-1, boxheight, boxwidth, 2]))
  return hconv4clip

def cnnmodel(X, Y, paras, flag='single'):
  assert(flag=='single' or flag=='combine')
  X = tf.reshape(X, shape=[-1, boxheight, boxwidth, 1])
  yreshape = tf.reshape(Y, [-1, boxheight, boxwidth, 1])
  yonehot = tf.concat(3, [1-yreshape, yreshape])
  if flag == 'combine':
    hconv4clip = buildcombmodel(X, paras)
  else: hconv4clip = buildmodel(X, paras)
  #hconv4log = -tf.log(hconv4clip)
  #q_train, q_test = crfrnn(hconv4log, paras['wsmooth'], paras['wcontra'], k1, k2, trainiter=5, testiter=10)
  #q_train = tf.reshape(q_train, [-1, boxheight, boxwidth, 2])
  q_train = -tf.log(hconv4clip)
  trainenergy = tf.reduce_sum((q_train)*yonehot, reduction_indices=3)
  #trainenergy = tf.reduce_prod(trainenergy, reduction_indices=[1,2])
  trainenergy = tf.reduce_mean(trainenergy, [0,1,2])
  q_test = hconv4clip
  #q_test = crfrnn(hconv4, paras['wsmooth'], paras['wcontra'], k1, k2, iter=5)
  q_test = tf.reshape(q_test, [-1, boxheight, boxwidth, 2])
  testenergy = tf.reduce_sum(tf.mul(q_test, yonehot), reduction_indices=3)
  #testenergy = tf.reduce_prod(testenergy, reduction_indices=[1,2])
  testenergy = tf.reduce_mean(testenergy, [0,1,2])
  predarg = tf.argmax(q_test, 3)
  yint64 = tf.to_int64(Y)
  acc = tf.equal(yint64, predarg)
  acc = tf.to_float(acc)
  accuracy = tf.reduce_mean(acc, [0,1,2])
  di = dice_tf(tf.reshape(yint64, [-1,]), tf.reshape(predarg, [-1,]))
  return trainenergy, accuracy, di, testenergy, q_test

def model(X, Y, k1, k2, paras, flag='single', fusion=None):
  assert(flag=='single' or flag=='combine')
  X = tf.reshape(X, shape=[-1, boxheight, boxwidth, 1])
  yreshape = tf.reshape(Y, [-1, boxheight, boxwidth, 1])
  yonehot = tf.concat(3, [1-yreshape, yreshape])
  if flag == 'combine':
    hconv4clip = buildcombmodel(X, paras, fusion=False)
    if fusion=='late':
      h22, h33, h44, h55 = tf.split(3, 4, hconv4clip)
      h22 = tf.squeeze(h22)
      h33 = tf.squeeze(h33)
      h44 = tf.squeeze(h44)
      h55 = tf.squeeze(h55)
      q_train22, q_test22=crfrnn(h22, paras['wsmooth22'], paras['wcontra22'], k1, k2)
      q_train33, q_test33=crfrnn(h33, paras['wsmooth33'], paras['wcontra33'], k1, k2)
      q_train44, q_test44=crfrnn(h44, paras['wsmooth44'], paras['wcontra44'], k1, k2)
      q_train55, q_test55=crfrnn(h55, paras['wsmooth55'], paras['wcontra55'], k1, k2)
      q_train22 = tf.reshape(q_train22, [-1,boxheight,boxwidth,1,2])
      q_train33 = tf.reshape(q_train33, [-1,boxheight,boxwidth,1,2])
      q_train44 = tf.reshape(q_train44, [-1,boxheight,boxwidth,1,2])
      q_train55 = tf.reshape(q_train55, [-1,boxheight,boxwidth,1,2])
      q_trainconc = tf.concat(3, [q_train22, q_train33, q_train44, q_train55])
      q_trainconc = tf.reshape(q_trainconc, [-1,boxheight,boxwidth,8])
      w_unary = tf.reshape(paras['wunary'], [8])
      q_trainw = tf.mul(q_trainconc, w_unary)
      q_trainw = tf.reshape(q_trainw, [-1,boxheight,boxwidth,4,2])
      q_trainsum = tf.reduce_sum(q_trainw, 3)
      q_trainsum = tf.reshape(q_trainsum, [-1,2])
      q_trainsoft = tf.nn.softmax(q_trainsum)
      q_trainclip = tf.clip_by_value(q_trainsoft, 1e-6, 1.)
      q_train = (tf.reshape(q_trainclip, [-1, boxheight, boxwidth, 2]))

      q_test22 = tf.reshape(q_test22, [-1,boxheight,boxwidth,1,2])
      q_test33 = tf.reshape(q_test33, [-1,boxheight,boxwidth,1,2])
      q_test44 = tf.reshape(q_test44, [-1,boxheight,boxwidth,1,2])
      q_test55 = tf.reshape(q_test55, [-1,boxheight,boxwidth,1,2])
      q_testconc = tf.concat(3, [q_test22, q_test33, q_test44, q_test55])
      q_testconc = tf.reshape(q_testconc, [-1,boxheight,boxwidth,8])
      q_testw = tf.mul(q_testconc, w_unary)
      q_testw = tf.reshape(q_testw, [-1,boxheight,boxwidth,4,2])
      q_testsum = tf.reduce_sum(q_testw, 3)
      q_testsum = tf.reshape(q_testsum, [-1,2])
      q_testsoft = tf.nn.softmax(q_testsum)
      q_testclip = tf.clip_by_value(q_testsoft, 1e-6, 1.)
      q_test = (tf.reshape(q_testclip, [-1, boxheight, boxwidth, 2]))
    else:
      q_train, q_test = crfrnn(hconv4clip, paras['wsmooth'], paras['wcontra'], k1, k2, 
        trainiter=5, testiter=10, wunary=paras['wunary'])
  else: 
    hconv4clip = buildmodel(X, paras)
    q_train, q_test = crfrnn(hconv4clip, paras['wsmooth'], paras['wcontra'], k1, k2, 
      trainiter=5, testiter=10)
  #hconv4log = -tf.log(hconv4clip)
  #q_train = tf.reshape(q_train, [-1, boxheight, boxwidth, 2])
  #q_train = -tf.log(hconv4clip)
  q_trainclip = tf.clip_by_value(q_train, 1e-6, 1.)
  trainenergy = tf.reduce_sum(-tf.log(q_trainclip)*yonehot, reduction_indices=3)
  #trainenergy = tf.reduce_prod(trainenergy, reduction_indices=[1,2])
  trainenergy = tf.reduce_mean(trainenergy, [0,1,2])
  
  #q_test = hconv4clip
  #q_test = crfrnn(hconv4, paras['wsmooth'], paras['wcontra'], k1, k2, iter=5)
  q_test = tf.reshape(q_test, [-1, boxheight, boxwidth, 2])
  testenergy = tf.reduce_sum(tf.mul(q_test, yonehot), reduction_indices=3)
  #testenergy = tf.reduce_prod(testenergy, reduction_indices=[1,2])
  testenergy = tf.reduce_mean(testenergy, [0,1,2])
  predarg = tf.argmax(q_test, 3)
  yint64 = tf.to_int64(Y)
  acc = tf.equal(yint64, predarg)
  acc = tf.to_float(acc)
  accuracy = tf.reduce_mean(acc, [0,1,2])
  di = dice_tf(tf.reshape(yint64, [-1,]), tf.reshape(predarg, [-1,]))
  return trainenergy, accuracy, di, testenergy, q_test

def crfatmodel(X, Y, k1, k2, paras, epsilon, flag='single', fusion=None):
  assert(flag=='single' or flag=='combine')
  energy, accuracy, di, testenergy, qtest = model(X, Y, k1, k2, paras, flag, fusion=fusion)
  gradx = tf.stop_gradient(tf.gradients(energy, X, aggregation_method=2))
  gradx = gradx / (1e-6 + tf.reduce_max(tf.abs(gradx), reduction_indices=[1,2], keep_dims=True))
  gradx = gradx / tf.sqrt(1e-12 + tf.reduce_sum(gradx**2, reduction_indices=[1,2], keep_dims=True))
  radv = epsilon * gradx
  advenergy, _, _, _, _ = model(X+radv, Y, k1, k2, paras, flag)
  energy = energy + advenergy
  return energy, accuracy, di, testenergy, qtest, advenergy

def cnnatmodel(X, Y, paras, epsilon, flag='single'):
  assert(flag=='single' or flag=='combine')
  energy, accuracy, di, testenergy, qtest = cnnmodel(X, Y, paras, flag=flag)
  gradx = tf.stop_gradient(tf.gradients(energy, X, aggregation_method=2))
  gradx = gradx / (1e-6 + tf.reduce_max(tf.abs(gradx), reduction_indices=[1,2], keep_dims=True))
  gradx = gradx / tf.sqrt(1e-12 + tf.reduce_sum(gradx**2, reduction_indices=[1,2], keep_dims=True))
  radv = epsilon * gradx
  advenergy, _, _, _, _ = cnnmodel(X+radv, Y, paras, flag)
  energy = energy + advenergy
  return energy, accuracy, di, testenergy, qtest, advenergy

def buildcombmodel(X, paras, fusion=True):
  hconv221 = convlayer(X, paras['wconv221'], paras['bconv221'], flag='maxpool')
  hconv222 = convlayer(hconv221, paras['wconv222'], paras['bconv222'], flag='maxpool')  
  hconv223 = tf.nn.conv2d(hconv222, paras['wconv223'], strides=[1,1,1,1], padding='VALID')
  hconv223bias = tf.nn.bias_add(hconv223, paras['bconv223'])
  hconv223tan = tf.nn.tanh(hconv223bias)
  hconv224 = tf.nn.conv2d_transpose(hconv223tan, paras['wconv224'], [batchsize,boxheight,boxwidth,2], 
                                  strides=[1,1,1,1], padding='VALID')
  hconv224 = tf.reshape(hconv224, [-1,boxheight*boxwidth*2])
  hconv224bias = tf.nn.bias_add(hconv224, paras['bconv224'])
  hconv224bias = tf.reshape(hconv224bias, [-1, boxheight, boxwidth, 2])
  hconv224bias = tf.reshape(hconv224bias, [-1,2])
  hconv224soft = tf.nn.softmax(hconv224bias)
  hconv224clip = tf.clip_by_value(hconv224soft, 1e-6, 1.)
  hconv224clip = (tf.reshape(hconv224clip, [-1, boxheight, boxwidth, 1, 2]))

  hconv331 = convlayer(X, paras['wconv331'], paras['bconv331'], flag='maxpool')
  hconv332 = convlayer(hconv331, paras['wconv332'], paras['bconv332'], flag='maxpool')
  hconv333 = tf.nn.conv2d(hconv332, paras['wconv333'], strides=[1,1,1,1], padding='VALID')
  hconv333bias = tf.nn.bias_add(hconv333, paras['bconv333'])
  hconv333tan = tf.nn.tanh(hconv333bias)
  hconv334 = tf.nn.conv2d_transpose(hconv333tan, paras['wconv334'], [batchsize,boxheight,boxwidth,2], 
                                  strides=[1,1,1,1], padding='VALID')
  hconv334 = tf.reshape(hconv334, [-1,boxheight*boxwidth*2])
  hconv334bias = tf.nn.bias_add(hconv334, paras['bconv334'])
  hconv334bias = tf.reshape(hconv334bias, [-1, boxheight, boxwidth, 2])
  hconv334bias = tf.reshape(hconv334bias, [-1,2])
  hconv334soft = tf.nn.softmax(hconv334bias)
  hconv334clip = tf.clip_by_value(hconv334soft, 1e-6, 1.)
  hconv334clip = (tf.reshape(hconv334clip, [-1, boxheight, boxwidth, 1, 2]))

  hconv441 = convlayer(X, paras['wconv441'], paras['bconv441'], flag='maxpool')
  hconv442 = convlayer(hconv441, paras['wconv442'], paras['bconv442'], flag='maxpool')
  hconv443 = tf.nn.conv2d(hconv442, paras['wconv443'], strides=[1,1,1,1], padding='VALID')
  hconv443bias = tf.nn.bias_add(hconv443, paras['bconv443'])
  hconv443tan = tf.nn.tanh(hconv443bias)
  hconv444 = tf.nn.conv2d_transpose(hconv443tan, paras['wconv444'], [batchsize,boxheight,boxwidth,2], 
                                  strides=[1,1,1,1], padding='VALID')
  hconv444 = tf.reshape(hconv444, [-1,boxheight*boxwidth*2])
  hconv444bias = tf.nn.bias_add(hconv444, paras['bconv444'])
  hconv444bias = tf.reshape(hconv444bias, [-1, boxheight, boxwidth, 2])
  hconv444bias = tf.reshape(hconv444bias, [-1,2])
  hconv444soft = tf.nn.softmax(hconv444bias)
  hconv444clip = tf.clip_by_value(hconv444soft, 1e-6, 1.)
  hconv444clip = (tf.reshape(hconv444clip, [-1, boxheight, boxwidth, 1, 2]))

  hconv551 = convlayer(X, paras['wconv551'], paras['bconv551'], flag='maxpool')
  hconv552 = convlayer(hconv551, paras['wconv552'], paras['bconv552'], flag='maxpool')
  hconv553 = tf.nn.conv2d(hconv552, paras['wconv553'], strides=[1,1,1,1], padding='VALID')
  hconv553bias = tf.nn.bias_add(hconv553, paras['bconv553'])
  hconv553tan = tf.nn.tanh(hconv553bias)
  hconv554 = tf.nn.conv2d_transpose(hconv553tan, paras['wconv554'], [batchsize,boxheight,boxwidth,2], 
                                  strides=[1,1,1,1], padding='VALID')
  hconv554 = tf.reshape(hconv554, [-1,boxheight*boxwidth*2])
  hconv554bias = tf.nn.bias_add(hconv554, paras['bconv554'])
  hconv554bias = tf.reshape(hconv554bias, [-1, boxheight, boxwidth, 2])
  hconv554bias = tf.reshape(hconv554bias, [-1,2])
  hconv554soft = tf.nn.softmax(hconv554bias)
  hconv554clip = tf.clip_by_value(hconv554soft, 1e-6, 1.)
  hconv554clip = (tf.reshape(hconv554clip, [-1, boxheight, boxwidth, 1, 2]))

  hconc = tf.concat(3, [hconv224clip, hconv334clip, hconv444clip, hconv554clip])
  if fusion:
    hconc = tf.reshape(hconc, [-1, 8])
    wunary = tf.reshape(paras['wunary'], [8,])
    h = hconc * wunary
    h = tf.reshape(h, [-1, boxheight, boxwidth, 4, 2])
    hsum = tf.reduce_sum(h, reduction_indices=3)
    hsum = tf.reshape(hsum, [-1,2])
    hsumsoft = tf.nn.softmax(hsum)
    hsumclip = tf.clip_by_value(hsumsoft, 1e-6, 1.)
    hsumclip = (tf.reshape(hsumclip, [-1, boxheight, boxwidth, 2]))
    return hsumclip
  else:
    return hconc
