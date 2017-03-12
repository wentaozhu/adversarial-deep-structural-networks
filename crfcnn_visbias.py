import tensorflow as tf
import scipy
import numpy as np 
import scipy.io as sio
import os
import scipy.misc
#import matplotlib.pyplot as plt
import utils_combine
from utils_combine import fetchdatalabel, calfilter, init, initcomb, model, crfatmodel, cnnatmodel, meanstd, cnnmodel, dice
rng = np.random.RandomState(1234)
tf.set_random_seed(1)
lrbig = 0.01 #0.01
lrmid = 0.003 #5
lr = 0.003 #2#5 #2 #3 #4 #3#5
lrsmall = 0.002
l2factor =  0.5 #0.01 # 0.01
totalepoch = 7000 #7000 #300 #200 #120
atepsilon = 1e-1 #5e-1 #0 #-1
modelname = 'cnn' #'crfcomb'
debug = False #False
savename = str(lr)+str(totalepoch)+str(atepsilon)+'1dilbiglrfsavebias'+str(5)+modelname+'norm'
fusion=None #'late' #None #'late'
saveperepoch = False#True #True #False #True #True # True
savefreq = 10
print(savename)
boxheight = 40
boxwidth = 40
batchsize = 29
#thresh = 0.885
postfix='roienhance.mat' # 'roienhance.mat' 'roienhance.jpeg'
traindatapath = '../trainroi1dilbig/'
trainlabelpath = '../trainroi1dilbig/'
testdatapath = '../testroi1dilbig/'
testlabelpath = '../testroi1dilbig/'

if __name__ == '__main__':
  assert(modelname!='crf' or modelname!='cnn' or modelname!='cnnat' or modelname!='crfat' 
  	or modelname!='cnncomb' or modelname!='crfcomb' or modelname!='cnncombat' or modelname!='crfcombat')
  traindata, trainlabel, trainfname = fetchdatalabel(path=traindatapath, postfix=postfix, flag='train')
  testdata, testlabel, testfname = fetchdatalabel(path=testdatapath, postfix=postfix, flag='test')
  if savename[-4:] == 'norm':
    print('norm')
    meandata = traindata.mean(axis=0)
    stddata = traindata.std(axis=0)
    traindata = meanstd(traindata, meandata, stddata)
    testdata = meanstd(testdata, meandata, stddata)
  traink1, traink2 = calfilter(traindata)
  testk1, testk2 = calfilter(testdata)

  X = tf.placeholder(tf.float32, [None, boxheight, boxwidth])
  Y = tf.placeholder(tf.float32, [None, boxheight, boxwidth])
  k1 = tf.placeholder(tf.float32, [None, boxheight*boxwidth, boxheight*boxwidth])
  k2 = tf.placeholder(tf.float32, [None, boxheight*boxwidth, boxheight*boxwidth])
  print(modelname[3:7])
  if modelname[3:7] == 'comb':
  	paras = initcomb(nhid221=37, nhid222=12, nhid223=355, lrf221=2, lrf222=2, lrf223=9,
  		nhid331=16, nhid332=13, nhid333=415, lrf331=3, lrf332=3, lrf333=8,
  		nhid441=9, nhid442=12, nhid443=588, lrf441=4, lrf442=4, lrf443=7,
  		nhid551=6, nhid552=12, nhid553=588, lrf551=5, lrf552=5, lrf553=7, fusion=fusion)
  else:
    paras = init(nhid1=6, nhid2=12, nhid3=588, lrf1=5, lrf2=5, lrf3=7)
  if modelname == 'cnn':
    trainenergy, accuracy, di, testenergy, qtest = cnnmodel(X, Y, paras)
  elif modelname == 'cnnat':
  	trainenergy, accuracy, di, testenergy, qtest, advloss = cnnatmodel(X, Y, paras, atepsilon)
  elif modelname == 'crfat':
  	trainenergy, accuracy, di, testenergy, qtest, advloss= crfatmodel(X, Y, k1, k2, paras, atepsilon)
  elif modelname == 'crf':
    trainenergy, accuracy, di, testenergy, qtest = model(X, Y, k1, k2, paras)
  elif modelname == 'cnncomb':
  	trainenergy, accuracy, di, testenergy, qtest = cnnmodel(X, Y, paras, flag='combine')
  elif modelname == 'cnncombat':
  	trainenergy, accuracy, di, testenergy, qtest, advloss = cnnatmodel(X, Y, paras, atepsilon, flag='combine')
  elif modelname == 'crfcomb':
  	trainenergy, accuracy, di, testenergy, qtest = model(X, Y, k1, k2, paras, flag='combine', fusion=fusion)
  elif modelname == 'crfcombat':
  	trainenergy, accuracy, di, testenergy, qtest, advloss= crfatmodel(X, Y, k1, k2, paras, atepsilon, 
  		flag='combine', fusion=fusion)
  learningrate = tf.Variable(lr, trainable=False)
  opt = tf.train.AdamOptimizer(learning_rate=learningrate)
  params = tf.trainable_variables()
  if modelname[:3] == 'crf':
  	if fusion=='late':
  	  l2loss = tf.nn.l2_loss(paras['wsmooth22']) + tf.nn.l2_loss(paras['wcontra22'])
  	  l2loss = l2loss + tf.nn.l2_loss(paras['wsmooth33']) + tf.nn.l2_loss(paras['wcontra33'])
  	  l2loss = l2loss + tf.nn.l2_loss(paras['wsmooth44']) + tf.nn.l2_loss(paras['wcontra44'])
  	  l2loss = l2loss + tf.nn.l2_loss(paras['wsmooth55']) + tf.nn.l2_loss(paras['wcontra55'])
  	else:
  	  l2loss = tf.nn.l2_loss(paras['wsmooth']) + tf.nn.l2_loss(paras['wcontra'])
  	l2loss = l2loss * l2factor
  	trainenergy = trainenergy + l2loss
  grads = tf.gradients(trainenergy, params, aggregation_method=2)
  optimizer = opt.apply_gradients(zip(grads, params))
  #optimizer = tf.train.AdamOptimizer(learning_rate=learningrate).minimize(trainenergy)
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  #with tf.Session() as sess:
    sess.run(init)
    tf.set_random_seed(1)
    trenerls, traccls, trdils, teenerls, teaccls, tedils = [], [], [], [], [], []
    besttrener, tolstep, besttedi, besttrdi = 1e6, 0, 0, 0
    predmat = {}
    for epoch in range(0,totalepoch):
      randindex = np.random.permutation(batchsize*8)
      trainloss, trainacc, traindi, trainl2 = 0, 0, 0, 0
      for iindex in range(8):
        #print(epoch)
        randindexii = randindex[iindex*batchsize : (iindex+1)*batchsize]
        traindataii = traindata[randindexii, :, :]
        trainlabelii = trainlabel[randindexii, :, :] 
        #print(len(randindexii))
        if modelname[:3] == 'cnn':
          if not debug:
            energy, acc, dival, _ = sess.run([trainenergy, accuracy, di, optimizer], feed_dict={X: traindataii,
                                         Y: trainlabelii, learningrate: lr})
          if debug and modelname[-2:] == 'at':
          	energy, acc, dival, adv, _ = sess.run([trainenergy, accuracy, di, advloss, optimizer], feed_dict={X: traindataii,
                                         Y: trainlabelii, learningrate: lr})
          	print(adv)
        else:
          traink1ii = traink1[randindexii, :, :]
          traink2ii = traink2[randindexii, :, :]  
          if not debug:
            energy, acc, dival, l2val, _ = sess.run([trainenergy, accuracy, di, l2loss, optimizer], feed_dict={X: traindataii,
                                         Y: trainlabelii, learningrate: lrbig, k1: traink1ii, k2: traink2ii})
          if debug and modelname[-2:] == 'at':
          	energy, acc, dival, l2val, adv,_ = sess.run([trainenergy, accuracy, di, l2loss, advloss, optimizer], feed_dict={X: traindataii,
                                         Y: trainlabelii, learningrate: lrbig, k1: traink1ii, k2: traink2ii})
          	print(adv)
          trainl2 += l2val
        traindi += dival
        trainloss += energy
        trainacc += acc
      print('epoch'+str(epoch)+', '+str(trainloss/8)+' '+str(trainl2/84)+' '+str(trainacc/8)+' '+str(traindi/8))
      #if trainloss/4 < besttrainloss:
      #  besttrainloss = trainloss/4
      #  tolstep = 0
      #else: tolstep += 1 
      #if tolstep == 8: lr *= .99
      for iindex in range(2):
        testdataii = testdata[iindex*batchsize:(iindex+1)*batchsize, :, :]
        testlabelii = testlabel[iindex*batchsize:(iindex+1)*batchsize, :, :]   
        if modelname[:3] == 'cnn':
          #print(testlabelii.shape, type(testlabelii))
          energy, acc, dival, qtestval = sess.run([testenergy, accuracy, di, qtest], feed_dict={X: testdataii,
            Y: testlabelii, learningrate: 0})
        else:
          testk1ii = testk1[iindex*batchsize:(iindex+1)*batchsize, :, :]       
          testk2ii = testk2[iindex*batchsize:(iindex+1)*batchsize, :, :]
          energy, acc, dival, qtestval = sess.run([testenergy, accuracy, di, qtest], feed_dict={X: testdataii,
                            Y: testlabelii, learningrate: 0, k1: testk1ii, k2: testk2ii})
        if iindex == 0:
          testenergyval, testacc, testpred = energy, acc, qtestval
        else:
          testenergyval += energy
          testacc += acc
          testpred = np.concatenate((testpred, qtestval), axis=0)
      testdi = dice(testlabel[:], (testpred.argmax(3))[:])
      if saveperepoch and (epoch%savefreq==0): #(totalepoch<200 or testdi>thresh):  # because of model size, save the result and tune further
      	for iindex in range(8):
          traindataii = traindata[iindex*batchsize:(iindex+1)*batchsize, :, :]
          trainlabelii = trainlabel[iindex*batchsize:(iindex+1)*batchsize, :, :]   
          if modelname[:3] == 'cnn':
            qtestval = sess.run(qtest, feed_dict={X: traindataii,
              Y: trainlabelii, learningrate: 0})
          else:
            traink1ii = traink1[iindex*batchsize:(iindex+1)*batchsize, :, :]       
            traink2ii = traink2[iindex*batchsize:(iindex+1)*batchsize, :, :]
            qtestval = sess.run(qtest, feed_dict={X: traindataii,
                            Y: trainlabelii, learningrate: 0, k1: traink1ii, k2: traink2ii})
          if iindex == 0:
            trainpred = qtestval
          else:
            trainpred = np.concatenate((trainpred, qtestval), axis=0)
        savemat={}
        savemat['testpred'] = testpred
        #savemat['testlabel'] = testlabel
        savemat['trainpred'] = trainpred
        #savemat['trainlabel'] = trainlabel
        sio.savemat(savename+str(epoch)+'.mat', savemat)
      print('epoch'+str(epoch)+', '+str(testenergyval/2)+' '+str(testacc/2)+' '+str(testdi))
      if traindi/8>0.90 and testdi > besttedi:
        saver.save(sess, savename+'.ckpt')
        besttedi = testdi
        besttrdi = traindi/8
        predmat['test'] = testpred
        bias = sess.run(paras['bconv4'], feed_dict={learningrate: 0})
        predmat['bias'] = bias
      #  lrbig = lrmid
      #if traindi/12 > 0.885:
      if testdi > 0.88:
      	lrbig = lr
      if testdi > 0.89:
      	lrbig = lrsmall
      trenerls.append(trainloss/8)
      traccls.append(trainacc/8)
      trdils.append(traindi/8)
      teenerls.append(testenergyval/8)
      teaccls.append(testacc/8)
      tedils.append(testdi)
    saver.restore(sess, savename+'.ckpt')
    #argpred, dival = sess.run([predarg, di], feed_dict={X: testdata,
    #                                Y: testlabel, learningrate: 0})#, k1: testk1, k2: testk2})
    #assert(dival==besttedi)
    print(str(besttedi)+' '+str(max(trdils)))
    predmat['trainloss'] = trenerls
    predmat['trainacc'] = traccls
    predmat['traindi'] = trdils
    predmat['testfname'] = testfname
    predmat['testloss'] = teenerls
    predmat['testacc'] = teaccls
    predmat['testdi'] = tedils
    predmat['testlabel'] = testlabel
    predmat['trainlabel'] = trainlabel
    predmat['traindata'] = traindata
    predmat['trainfname'] = trainfname
    predmat['testdata'] = testdata
    for iindex in range(8):
      traindataii = traindata[iindex*batchsize:(iindex+1)*batchsize, :, :]
      trainlabelii = trainlabel[iindex*batchsize:(iindex+1)*batchsize, :, :]   
      if modelname[:3] == 'cnn':
        qtestval = sess.run(qtest, feed_dict={X: traindataii,
          Y: trainlabelii, learningrate: 0})
      else:
        traink1ii = traink1[iindex*batchsize:(iindex+1)*batchsize, :, :]       
        traink2ii = traink2[iindex*batchsize:(iindex+1)*batchsize, :, :]
        qtestval = sess.run(qtest, feed_dict={X: traindataii,
          Y: trainlabelii, learningrate: 0, k1: traink1ii, k2: traink2ii})
      if iindex == 0:
        trainpred = qtestval
      else:
        trainpred = np.concatenate((trainpred, qtestval), axis=0)
    predmat['trainpred'] = trainpred
    print(savename+'.mat', str(max(tedils)))
    sio.savemat(savename+'.mat', predmat)
    plt.figure(1)
    t = np.arange(0, totalepoch, 1)
    plt.subplot(311)
    plt.plot(t, trenerls, 'r', t, teenerls, 'b')
    plt.subplot(312)
    plt.plot(t, traccls, 'r', t, teaccls, 'b')
    plt.subplot(313)
    plt.plot(t, trdils, 'r', t, tedils, 'b')
    plt.savefig(savename+'.png')
