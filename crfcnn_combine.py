import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy
import numpy as np 
import scipy.io as sio
import os
import scipy.misc
import matplotlib.pyplot as plt
import utils_combine
from utils_combine import fetchdatalabel, calfilter, init, initcomb, model, crfatmodel, cnnatmodel, meanstd, cnnmodel, dice
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-lr", "--learningrate", default=0.003, help="Learning Rate", type=float)
parser.add_argument("-l2", "--l2factor", default=0.5, help="L2 Factor", type=float)
parser.add_argument("-n", "--n_epoch", default=100, help="Number of epochs", type=int)
parser.add_argument("-at", "--at_epsilon", default=1e-1, help="Epsilon for Adversarial Training", type=float)
parser.add_argument("-md", "--modelname", default='cnn', help="Model Name")
parser.add_argument("-norm", "--normalize", action="store_true", help="normalize the training data.")
parser.add_argument("-deb", "--debug", action="store_true", help="Debug mode")


rng = np.random.RandomState(1234)
tf.set_random_seed(1)

args = parser.parse_args()

lrbig = 0.003 #0.01
lrmid = 0.003 #5
lr = args.learningrate #2#5 #2 #3 #4 #3#5
lrsmall = 0.002
l2factor =  args.l2factor #0.01 # 0.01
totalepoch = args.n_epoch #5000 #7000 #7000 #300 #200 #120
atepsilon = args.at_epsilon #5e-1 #0 #-1
modelname = args.modelname #'crfcomb'
savename = str(lr)+'_'+str(totalepoch)+'_'+str(atepsilon)+'_2lrfsave_'+str(5)+'_' + modelname + ('_norm' if args.normalize else '') + '_valpara335'
fusion=None #'late' #None #'late'
saveperepoch = False #True #False #True #True # True
savefreq = 10
print(savename)
boxheight = 40
boxwidth = 40
batchsize = 29
#thresh = 0.885
postfix='roienhance.mat' # 'roienhance.mat' 'roienhance.jpeg'
traindatapath = '../inbreast/trainroi1dilbig/'
trainlabelpath = '../inbreast/trainroi1dilbig/'
testdatapath = '../inbreast/testroi1dilbig/'
testlabelpath = '../inbreast/testroi1dilbig/'
if __name__ == '__main__':
  assert(modelname!='crf' or modelname!='cnn' or modelname!='cnnat' or modelname!='crfat' 
  	or modelname!='cnncomb' or modelname!='crfcomb' or modelname!='cnncombat' or modelname!='crfcombat')
  traindata, trainlabel, trainfname = fetchdatalabel(path=traindatapath, postfix=postfix, flag='train')
  testdata, testlabel, testfname = fetchdatalabel(path=testdatapath, postfix=postfix, flag='test')
  if args.normalize:
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
    paras = init(nhid1=44, nhid2=16, nhid3=335, lrf1=2, lrf2=2, lrf3=9)  #  nhid441=9, nhid442=12, nhid443=588, lrf441=4, lrf442=4, lrf443=7,
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
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    tf.set_random_seed(1)
    trenerls, traccls, trdils, teenerls, teaccls, tedils = [], [], [], [], [], []
    besttrener, tolstep, besttedi, besttrdi = 1e6, 0, 0, 0
    predmat = {}

    ### Epoch Looping Starts ###
    for epoch in range(0,totalepoch):
      
      """ Train Starts Here """
      randindex = np.random.permutation(batchsize*8)
      trainloss, trainacc, traindi, trainl2 = 0, 0, 0, 0
      for iindex in range(8):
        #print(epoch)
        randindexii = randindex[iindex*batchsize : (iindex+1)*batchsize]
        traindataii = traindata[randindexii, :, :]
        trainlabelii = trainlabel[randindexii, :, :] 
        #print(len(randindexii))
        if modelname[:3] == 'cnn': # FCN without CRF
          if not args.debug:
            energy, acc, dival, _ = sess.run([trainenergy, accuracy, di, optimizer], feed_dict={X: traindataii,
                                         Y: trainlabelii, learningrate: lr})
          elif modelname[-2:] == 'at':
          	energy, acc, dival, adv, _ = sess.run([trainenergy, accuracy, di, advloss, optimizer], feed_dict={X: traindataii,
                                         Y: trainlabelii, learningrate: lr})
          	print(adv)
        elif modelname[:3] == 'crf':  # FCN with CRF
          traink1ii = traink1[randindexii, :, :]
          traink2ii = traink2[randindexii, :, :]  
          if not args.debug:
            energy, acc, dival, l2val, _ = sess.run([trainenergy, accuracy, di, l2loss, optimizer], feed_dict={X: traindataii,
                                         Y: trainlabelii, learningrate: lrbig, k1: traink1ii, k2: traink2ii})
          elif modelname[-2:] == 'at':
          	energy, acc, dival, l2val, adv,_ = sess.run([trainenergy, accuracy, di, l2loss, advloss, optimizer], feed_dict={X: traindataii,
                                         Y: trainlabelii, learningrate: lrbig, k1: traink1ii, k2: traink2ii})
          	print(adv)
          trainl2 += l2val
        else:
          print("Invalid model name")
          exit(1)
        traindi += dival
        trainloss += energy
        trainacc += acc
        print(acc)
      print('epoch'+str(epoch)+', train-loss: '+str(trainloss/8)+' train-l2-penalty: '+str(trainl2/84)+' train-accuracy: '+str(trainacc/8)+' train-dice: '+str(traindi/8))
      #if trainloss/4 < besttrainloss:
      #  besttrainloss = trainloss/4
      #  tolstep = 0
      #else: tolstep += 1 
      #if tolstep == 8: lr *= .99
      """ Train Ends Here """

      """ Test Starts Here """
      for iindex in range(2):
        testdataii = testdata[iindex*batchsize:(iindex+1)*batchsize, :, :]
        testlabelii = testlabel[iindex*batchsize:(iindex+1)*batchsize, :, :]   
        if modelname[:3] == 'cnn':  # FCN without CRF
          #print(testlabelii.shape, type(testlabelii))
          energy, acc, dival, qtestval = sess.run([trainenergy, accuracy, di, qtest], feed_dict={X: testdataii,
            Y: testlabelii, learningrate: 0})
        elif modelname[:3] == 'crf':   # FCN with CRF
          testk1ii = testk1[iindex*batchsize:(iindex+1)*batchsize, :, :]       
          testk2ii = testk2[iindex*batchsize:(iindex+1)*batchsize, :, :]
          energy, acc, dival, qtestval = sess.run([trainenergy, accuracy, di, qtest], feed_dict={X: testdataii,
                            Y: testlabelii, learningrate: 0, k1: testk1ii, k2: testk2ii})
        else:
          print("Invalid model name detected in testing")
          exit(1)
        if iindex == 0:
          testenergyval, testacc, testpred = energy, acc, qtestval
        else:
          testenergyval += energy
          testacc += acc
          testpred = np.concatenate((testpred, qtestval), axis=0)
      testdi = dice(testlabel[:], (testpred.argmax(3))[:])
      print('epoch'+str(epoch)+', test energy: '+str(testenergyval/2)+' test accuracy: '+str(testacc/2)+' test dice: '+str(testdi))
      """ Test Ends Here """

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

      if testdi > besttedi:
        saver.save(sess, savename+'.ckpt')
        besttedi = testdi
        besttrdi = traindi/8
        predmat['test'] = testpred
      #  lrbig = lrmid
      #if traindi/12 > 0.885:
      if testdi > 0.88:
      	lrbig = lr
      if testdi > 0.89:
      	lrbig = lrsmall
      trenerls.append(trainloss/8)
      traccls.append(trainacc/8)
      trdils.append(traindi/8)
      teenerls.append(testenergyval/2)
      teaccls.append(testacc/2)
      tedils.append(testdi)
    ### Epoch Looping Ends ###
    print(str(besttedi)+' '+str(max(trdils)))

    ### Save .mat File ###
    saver.restore(sess, savename+'.ckpt')
    #argpred, dival = sess.run([predarg, di], feed_dict={X: testdata,
    #                                Y: testlabel, learningrate: 0})#, k1: testk1, k2: testk2})
    #assert(dival==besttedi)
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
    ### End Saving .mat File###


    ### Plot ###
    plt.figure(1)
    t = np.arange(0, totalepoch, 1)
    sub1 = plt.subplot(311)
    sub1.set_title('Energy')
    sub1.plot(t, trenerls, 'r', t, teenerls, 'b')
    sub2 = plt.subplot(312)
    sub2.set_title('Accuracy')
    sub2.plot(t, traccls, 'r', t, teaccls, 'b')
    sub3 = plt.subplot(313)
    sub3.set_title('Dice Score')
    sub3.plot(t, trdils, 'r', t, tedils, 'b')
    from matplotlib.patches import Patch
    legends = [Patch(facecolor='r', label='Train'), Patch(facecolor='b', label='Test')]
    plt.legend(handles=legends)
    plt.savefig(savename+'.png')
    ### End Plot ###