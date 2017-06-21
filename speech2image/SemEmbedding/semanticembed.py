import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from timeit import timeit
#from IPython.core.debugger import Tracer
from scipy.io import wavfile
#from scnn_test import scnn_test


# Spectrogram CNN used for semantic embedding task
# More info see Harwath & Glass 2016 paper
# Written by Liming Wang, Apr. 17, 2017
# Modification:
#   Apr. 17: change the output of the scnn function to return all the network parameters,
#   so we do not need to train the network every time before testing

# Use off-the-shelf package for mel frequency spectrogram (not MFCC) for now, may write one myself at some point

####################################
# Set Execution Control Parameters #
####################################
#LD_DATA = False
DEV_TEST = True
EVAL_TEST = True
PRINT_MODEL = False
PRINT_HIDDEN_OUT = False
PRINT_ACT = False
PRINT_SIM = True
SAVE_META = False
SAVE_NPZ = True
RAND_TRAIN = True
RAND_BATCH = False
RAND_COST = False
DEBUG = True
USE_MOMENT = False


# Copyright (c) 2006 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

"""Compute MFCC coefficients.
    
    This module provides functions for computing MFCC (mel-frequency
    cepstral coefficients) as used in the Sphinx speech recognition
    system.
    """

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
__version__ = "$Revision: 6390 $"

import numpy, numpy.fft

def mel(f):
    return 2595. * numpy.log10(1. + f / 700.)

def melinv(m):
    return 700. * (numpy.power(10., m / 2595.) - 1.)

class MFCC(object):
    def __init__(self, nfilt=40, ncep=13,
                 lowerf=133.3333, upperf=6855.4976, alpha=0.97,
                 samprate=16000, frate=160, wlen=0.0256,
                 nfft=512):
        # Store parameters
        self.lowerf = lowerf
        self.upperf = upperf
        self.nfft = nfft
        self.ncep = ncep
        self.nfilt = nfilt
        self.frate = frate
        self.fshift = float(samprate) / frate
        
        # Build Hamming window
        self.wlen = int(wlen * samprate)
        self.win = numpy.hamming(self.wlen)
        
        # Prior sample for pre-emphasis
        self.prior = 0
        self.alpha = alpha
        
        # Build mel filter matrix
        self.filters = numpy.zeros((nfft/2+1,nfilt), 'd')
        dfreq = float(samprate) / nfft
        if upperf > samprate/2:
            raise(Exception,
                  "Upper frequency %f exceeds Nyquist %f" % (upperf, samprate/2))
        melmax = mel(upperf)
        melmin = mel(lowerf)
        dmelbw = (melmax - melmin) / (nfilt + 1)
        # Filter edges, in Hz
        filt_edge = melinv(melmin + dmelbw * numpy.arange(nfilt + 2, dtype='d'))
        
        for whichfilt in range(0, nfilt):
            # Filter triangles, in DFT points
            leftfr = round(filt_edge[whichfilt] / dfreq)
            centerfr = round(filt_edge[whichfilt + 1] / dfreq)
            rightfr = round(filt_edge[whichfilt + 2] / dfreq)
            # For some reason this is calculated in Hz, though I think
            # it doesn't really matter
            fwidth = (rightfr - leftfr) * dfreq
            height = 2. / fwidth
            
            if centerfr != leftfr:
                leftslope = height / (centerfr - leftfr)
            else:
                leftslope = 0
            freq = leftfr + 1
            while freq < centerfr:
                self.filters[int(freq),int(whichfilt)] = (freq - leftfr) * leftslope
                freq = freq + 1
            if freq == centerfr: # This is always true
                self.filters[int(freq),int(whichfilt)] = height
                freq = freq + 1
            if centerfr != rightfr:
                rightslope = height / (centerfr - rightfr)
            while freq < rightfr:
                self.filters[int(freq),int(whichfilt)] = (freq - rightfr) * rightslope
                freq = freq + 1
                #             print("Filter %d: left %d=%f center %d=%f right %d=%f width %d" %
                #                   (whichfilt,
                #                   leftfr, leftfr*dfreq,
                #                   centerfr, centerfr*dfreq,
                #                   rightfr, rightfr*dfreq,
                #                   freq - leftfr))
                #             print self.filters[leftfr:rightfr,whichfilt]
                # Build DCT matrix
                self.s2dct = s2dctmat(nfilt, ncep, 1./nfilt)
                self.dct = dctmat(nfilt, ncep, numpy.pi/nfilt)

    def sig2s2mfc(self, sig):
        nfr = int(len(sig) / self.fshift + 1)
        mfcc = numpy.zeros((nfr, self.ncep), 'd')
        fr = 0
        while fr < nfr:
            start = round(fr * self.fshift)
            end = min(len(sig), start + self.wlen)
            frame = sig[start:end]
            if len(frame) < self.wlen:
                frame = numpy.resize(frame,self.wlen)
                frame[self.wlen:] = 0
            mfcc[fr] = self.frame2s2mfc(frame)
            fr = fr + 1
        return mfcc

    def sig2logspec(self, sig):
        nfr = int(len(sig) / self.fshift + 1)
        mfcc = numpy.zeros((nfr, self.nfilt), 'd')
        fr = 0
        while fr < nfr:
            start = round(fr * self.fshift)
            end = min(len(sig), start + self.wlen)
            frame = sig[start:end]
            if len(frame) < self.wlen:
                frame = numpy.resize(frame,self.wlen)
                frame[self.wlen:] = 0
            mfcc[fr] = self.frame2logspec(frame)
            fr = fr + 1
        return mfcc

    def pre_emphasis(self, frame):
        # FIXME: Do this with matrix multiplication
        outfr = numpy.empty(len(frame), 'd')
        outfr[0] = frame[0] - self.alpha * self.prior
        for i in range(1,len(frame)):
            outfr[i] = frame[i] - self.alpha * frame[i-1]
        self.prior = frame[-1]
        return outfr

    def frame2logspec(self, frame):
        frame = self.pre_emphasis(frame) * self.win
        fft = numpy.fft.rfft(frame, self.nfft)
        # Square of absolute value
        power = fft.real * fft.real + fft.imag * fft.imag
        return numpy.log(numpy.dot(power, self.filters).clip(1e-5,numpy.inf))
    
    def frame2s2mfc(self, frame):
        logspec = self.frame2logspec(frame)
        return numpy.dot(logspec, self.s2dct.T) / self.nfilt

def s2dctmat(nfilt,ncep,freqstep):
    """Return the 'legacy' not-quite-DCT matrix used by Sphinx"""
    melcos = numpy.empty((ncep, nfilt), 'double')
    for i in range(0,ncep):
        freq = numpy.pi * float(i) / nfilt
        melcos[i] = numpy.cos(freq * numpy.arange(0.5, float(nfilt)+0.5, 1.0, 'double'))
    melcos[:,0] = melcos[:,0] * 0.5
    return melcos

def logspec2s2mfc(logspec, ncep=13):
    """Convert log-power-spectrum bins to MFCC using the 'legacy'
        Sphinx transform"""
    nframes, nfilt = logspec.shape
    melcos = s2dctmat(nfilt, ncep, 1./nfilt)
    return numpy.dot(logspec, melcos.T) / nfilt

def dctmat(N,K,freqstep,orthogonalize=True):
    """Return the orthogonal DCT-II/DCT-III matrix of size NxK.
        For computing or inverting MFCCs, N is the number of
        log-power-spectrum bins while K is the number of cepstra."""
    cosmat = numpy.zeros((N, K), 'double')
    for n in range(0,N):
        for k in range(0, K):
            cosmat[n,k] = numpy.cos(freqstep * (n + 0.5) * k)
    if orthogonalize:
        cosmat[:,0] = cosmat[:,0] * 1./numpy.sqrt(2)
    return cosmat

def dct(input, K=13):
    """Convert log-power-spectrum to MFCC using the orthogonal DCT-II"""
    nframes, N = input.shape
    freqstep = numpy.pi / N
    cosmat = dctmat(N,K,freqstep)
    return numpy.dot(input, cosmat) * numpy.sqrt(2.0 / N)

def dct2(input, K=13):
    """Convert log-power-spectrum to MFCC using the normalized DCT-II"""
    nframes, N = input.shape
    freqstep = numpy.pi / N
    cosmat = dctmat(N,K,freqstep,False)
    return numpy.dot(input, cosmat) * (2.0 / N)

def idct(input, K=40):
    """Convert MFCC to log-power-spectrum using the orthogonal DCT-III"""
    nframes, N = input.shape
    freqstep = numpy.pi / K
    cosmat = dctmat(K,N,freqstep).T
    return numpy.dot(input, cosmat) * numpy.sqrt(2.0 / K)

def dct3(input, K=40):
    """Convert MFCC to log-power-spectrum using the unnormalized DCT-III"""
    nframes, N = input.shape
    freqstep = numpy.pi / K
    cosmat = dctmat(K,N,freqstep,False)
    cosmat[:,0] = cosmat[:,0] * 0.5
    return numpy.dot(input, cosmat.T)

def loaddata(ntr, ntx):
    mfcc = MFCC()
    dir_info = '../data/flickr_audio/'
    filename_info = 'wav2capt.txt'
    
    dir_sp = '../data/flickr_audio/wavs/'
    dir_penult = '../data/vgg_flickr8k_nnet_penults/'
    
    captions_tr = []
    im_tr = []
    captions_tx = []
    im_tx = []
    Leq = 1024
    with open(dir_info+filename_info, 'r') as f:
        for i in range(ntr):
            # Load the filenames of the files storing the audio captions and its corresponding vgg16 feature
            cur_info = f.readline()
            cur_info_parts = cur_info.rstrip().split()
            sp_name = cur_info_parts[0]
            caption_info = wavfile.read(dir_sp+sp_name)
            caption_time = caption_info[1]
            # Convert the audio into spectrogram
            caption = mfcc.sig2logspec(caption_time)
            # Transpose the caption data
            if caption.shape[0] > caption.shape[1]:
                caption = np.transpose(caption)
            
            # Equalize the length
            if caption.shape[1] < Leq:
                nframes = caption.shape[1]
                nmf = caption.shape[0]
                caption_new = np.zeros((nmf, Leq))
                caption_new[:, round((Leq-nframes)/2):round((Leq-nframes)/2)+nframes] = caption
            else:
                if caption.shape[1] > Leq:
                    nframes = caption.shape[1]
                    nmf = caption.shape[0]
                    caption_new = np.zeros((nmf, Leq))
                    caption_new = caption[:, round((nframes-Leq)/2):round((nframes-Leq)/2)+Leq]
            captions_tr.append(caption_new)
            # Remove the .jpg# at the end of the file to .npz format, which is used to store vgg16 feature
            im_name_raw = cur_info_parts[1]
            im_name_parts = im_name_raw.split('.')
            im_name = im_name_parts[0]
            # Load the softmax activations of the images, store them into an array
            data = np.load(dir_penult+im_name+'.npz')
            cur_penult = data['arr_0']
            im_tr.append(cur_penult)
            if i%10:
                print('Finish loading', 100*i/ntr, 'percent of training data')

        for j in range(ntx):
            # Load the image names and the image captions, break the captions into words and store in a list
            cur_info = f.readline()
            cur_info_parts = cur_info.rstrip().split()
            sp_name = cur_info_parts[0]
            caption_info = wavfile.read(dir_sp+sp_name)
            caption_time = caption_info[1]
            caption = mfcc.sig2logspec(caption_time)
            # Transpose the data
            if caption.shape[0] > caption.shape[1]:
                caption = np.transpose(caption)
            # Equalize the length
            if caption.shape[1] < Leq:
                nframes = caption.shape[1]
                nmf = caption.shape[0]
                caption_new = np.zeros((nmf, Leq))
                print('274:', nframes, (Leq-nframes)/2)
                caption_new[:, (Leq-nframes)/2:(Leq-nframes)/2+nframes] = caption
            else:
                if caption.shape[1] > Leq:
                    nframes = caption.shape[1]
                    nmf = caption.shape[0]
                    caption_new = np.zeros((nmf, Leq))
                    caption_new = caption[:, (nframes-Leq)/2:(nframes-Leq)/2+Leq]
            captions_tx.append(caption_new)
            # Remove the .jpg# at the end of the file to the format of vgg16 feature file
            im_name_raw = cur_info_parts[1]
            im_name_parts = im_name_raw.split('.')
            # Remove the caption number
            im_name = im_name_parts[0]
            # Load the softmax activations of the images, store them into an array
            data = np.load(dir_penult+im_name+'.npz')
            cur_penult = data['arr_0']
            im_tx.append(cur_penult)
            if j % 10:
                print('Finish loading', 100*j/ntx, 'percent of test data')

    captions_tr = np.array(captions_tr)
    captions_tx = np.array(captions_tx)
    im_tr = np.array(im_tr)
    im_tx = np.array(im_tx)
    np.savez('captions.npz', captions_tr, captions_tx)
    np.savez('images.npz', im_tr, im_tx)
    return captions_tr, captions_tx, im_tr, im_tx


#############
# Load data #
#############

Fs = 16000;
nlabel = 61;
#nphn = 61;
caption_name = 'captions.npz'
image_name = 'images.npz'

if len(sys.argv) >= 3:
    ntr = int(sys.argv[1])
    ntx = int(sys.argv[2])
    if len(sys.argv) >= 5:
        caption_name = sys.argv[3]
        image_name = sys.argv[4]

if os.path.isfile(caption_name):
    data_caption = np.load(caption_name)
    captions_tr = data_caption['arr_0'][0:ntr]
    # Use training set for test purpose
    #captions_tx = data_caption['arr_0'][0:200]
    captions_tx = data_caption['arr_0'][ntr:ntr+ntx]
    data_im = np.load(image_name)
    im_tr = data_im['arr_0'][0:ntr]
    # Use training set for test purpose
    #im_tx = data_im['arr_0'][0:200]
    im_tx = data_im['arr_0'][ntr:ntr+ntx]
    print('Line 333 existing data dimension: ', captions_tr.shape)

else:
    captions_tr, captions_tx, im_tr, im_tx = loaddata(ntr, ntx)
    print('Line 354 newly loaded data dimension: ', captions_tr.shape)
print('Line 335: finish loading data')
#nframes = captions_tr[0].shape[1]# number of frames in each frequency channel
#nmf = captions_tr[0].shape[0]
nframes = 1024
nmf = 40
npenult_vgg = 4096
nembed = 1024

# Load flikr8k audio data
X_tr = captions_tr
X_tx = captions_tx

# Load penultimate layer data of the vgg net
Z_tr_vgg = im_tr
Z_tx_vgg = im_tx

if not RAND_COST:
    learn_rate = 1e-5
else:
    learn_rate = 1e-5

###############################
# Build the computation graph #
###############################

def weight_variable(dims):
    if not RAND_COST:
        w = tf.Variable(tf.truncated_normal(dims, mean=0.0, stddev=0.01))
    else:
        w = tf.Variable(tf.truncated_normal(dims, mean=0.0, stddev=0.01))
    return w

def bias_variable(dims):
    if not RAND_COST:
        b = tf.Variable(tf.truncated_normal(dims, mean=0.01, stddev=0.01))
    else:
        b = tf.Variable(tf.truncated_normal(dims, mean=0.01, stddev=0.01))
    return b

# Build the computation graph
J = [nmf, 64, 512, 1024, nlabel]
N = [4, 24, 24]
D = [nframes, nframes/2, nframes/4, nframes/4]

with tf.device('/gpu:0'):
    # Assume 4 layers
    # Filter dimension: 1 x filter length x num input channels x num output channels
    # X_in, Y: input data and labels respectively
    # J: channel dimension of the architecture of TDNN
    # N: filter order in each channel
    # D: dimension of input node at each layer
    # More see Harwath et al. 2015 & 2016
    # Created by Liming Wang on April. 4th, 2017
    
    # Mean subtraction with mean spectrogram estimated over the batch
    X_in = tf.placeholder(tf.float32, shape=[None, 1, nframes, nmf])
    X_mean = tf.reduce_mean(X_in)
    X_zm = X_in - X_mean
    w_in = weight_variable([1, N[0]+1, J[0], J[1]])
    b_in = bias_variable([J[1]])
    
    w_hidden1 = weight_variable([1, N[1]+1, J[1], J[2]])
    b_hidden1 = bias_variable([J[2]])
    
    w_hidden2 = weight_variable([1, N[2]+1, J[2], J[3]])
    b_hidden2 = bias_variable([J[3]])
    
    # Initialize softmax layer
    w_out = weight_variable([J[3], J[4]])
    b_out = bias_variable([J[4]])
    
    a1_conv = tf.nn.conv2d(X_zm, w_in, strides=[1, 1, 1, 1], padding='SAME') + b_in
    # Max pooling with vertical stride 1 and horizontal stride 2
    a1_pool = tf.nn.max_pool(a1_conv, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='SAME')
    h1 = tf.nn.relu(a1_pool)
    
    a2_conv = tf.nn.conv2d(h1, w_hidden1, strides=[1, 1, 1, 1], padding='SAME') + b_hidden1
    a2_pool = tf.nn.max_pool(a2_conv, ksize=[1, 1, 4, 1], strides=[1, 1, 2, 1], padding='SAME')
    h2 = tf.nn.relu(a2_pool)
    
    a3_conv = tf.nn.conv2d(h2, w_hidden2, strides=[1, 1, 1, 1], padding='SAME') + b_hidden2
    h3 = tf.nn.relu(a3_conv)
    
    # Penultimate layer
    h4 = tf.nn.max_pool(h3, ksize=[1, 1, D[3], 1], strides=[1, 1, 1, 1], padding='VALID')
    h4_re = tf.reshape(h4, [-1, J[3]])
    # L2 normalization
    h4_ren = tf.nn.l2_normalize(h4_re, dim=1)
    Z_embed_sp = h4_ren
    a_out = tf.matmul(h4_ren, w_out) + b_out
    Y_pred = tf.nn.softmax(a_out)
    
    pmtrs = [w_in, b_in, w_hidden1, b_hidden1, w_hidden2, b_hidden2, w_out, b_out]

    w_embed = weight_variable([npenult_vgg, nembed])
    b_embed = bias_variable([nembed])

    # Map the penultimate layer of the VGG to embedding space
    Z_penult_vgg = tf.placeholder(tf.float32, shape=[None, npenult_vgg])
    Z_embed_vgg = tf.matmul(Z_penult_vgg, w_embed) + b_embed

    # Compute the similarity scores
    s_a = tf.matmul(Z_embed_sp, tf.transpose(Z_embed_vgg))
    s = tf.nn.relu(s_a)
    s_p = tf.diag_part(s)
    # Maximum margin cost function
    if not RAND_COST:
        cost = tf.reduce_sum(tf.nn.relu(s-s_p+1)+tf.nn.relu(tf.transpose(s)-s_p+1))#*(1-tf.cast(tf.equal(s, s_p), tf.float32))))
    else:
        s_p_diag = tf.diag(s)
        # fake image similarity scores
        #s_i = tf.reduce_max(s-s_p_diag, reduction_indices=1)
        #s_c = tf.reduce_max(tf.transpose(s-s_p_diag), reduction_indices=1)
        #cost = tf.reduce_sum(tf.nn.relu(s_c) + tf.nn.relu(s_i), reduction_indices=0)
        cost = tf.reduce_sum(tf.nn.relu(tf.reduce_min(s-s_p+1, reduction_indices=1)) + tf.nn.relu(tf.reduce_min(tf.transpose(s)-s_p+1, reduction_indices=1))) + tf.reduce_sum(tf.nn.relu(tf.reduce_max(s-s_p+1, reduction_indices=1)) + tf.nn.relu(tf.reduce_max(tf.transpose(s)-s_p+1, reduction_indices=1)))

    ds = s-s_p

    #global_step = tf.Variable(0, trainable=False)
    if not USE_MOMENT:
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(cost)
    else:
        #cur_learn_rate = tf.train.exponential_decay(learn_rate, global_step, 5, 0.5, staircase=True)
        train_step = tf.train.MomentumOptimizer(learn_rate, momentum=0.5).minimize(cost)

config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)

init = tf.initialize_all_variables()
sess.run(init)

###############################################################
# Train the ANN model (TDNN) using minibatch gradient descent #
###############################################################

print('Start Training ...')
batch_size = 128;
nbatch = int(ntr/batch_size);
niter = 50;

tr_accuracy = np.zeros([niter,])
dev_accuracy = np.zeros([niter,])
for t in range(niter):
    randidx = np.argsort(np.random.normal(size=(ntr,)), 0)
    if not RAND_TRAIN:
        X_tr_rand = X_tr
        Z_tr_vgg_rand = Z_tr_vgg
    else:
        X_tr_rand = X_tr[randidx]
        Z_tr_vgg_rand = Z_tr_vgg[randidx]
    rand_b_idx = np.argsort(np.random.normal(size=[nbatch]))
    for i in range(nbatch):
        if RAND_BATCH:
            X_batch = X_tr_rand[rand_b_idx[i]*batch_size:(rand_b_idx[i]+1)*batch_size]

        X_batch = X_tr_rand[i*batch_size:(i+1)*batch_size]
        Z_batch = Z_tr_vgg_rand[i*batch_size:(i+1)*batch_size]

        # Recall the input for conv2d is of shape batch x input height x input width x # of channels
        X_batch_4d = np.reshape(X_batch, [batch_size, 1, nframes, nmf])
        # Test every nbatch batches
        sess.run(train_step, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
        if i % 10 == 0:
            print('Iteration', t, 'at batch', i)

            if PRINT_HIDDEN_OUT:
                _h4_ren = sess.run(h4_ren, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
                print('sp embedding layer', _h4_ren.shape)
                print(_h4_ren)
                _Z_embed_vgg = sess.run(Z_embed_vgg, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
                print('vgg embedding layer:', _Z_embed_vgg.shape)
                print(_Z_embed_vgg)

            # Evaluate the model with the top 10 image matching error rate
            similarity = sess.run(s, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            cur_cost = sess.run(cost, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
            # Find the indices of the images with the top 10 similarity score
            ntop = 10
            top_indices = np.zeros((ntop, batch_size))
            for k in range(ntop):
                cur_top_idx = np.argmax(similarity, axis=1)
                top_indices[k] = cur_top_idx
                # To leave out the top values that have been determined and the find the top values for the rest of the indices
                for l in range(batch_size):
                    similarity[l][cur_top_idx[l]] = -1
            # Find if the image with the matching index has the highest similarity score
            dev = abs(top_indices - np.linspace(0, batch_size-1, batch_size))
            min_dev = np.amin(dev, axis=0)
            print('current deviation from correct indices for training:', min_dev)
            # Count the number of correct matching by counting the number of 0s in dev
            tr_accuracy[t] = np.mean((min_dev==0))
            if PRINT_SIM:
                cur_sa = sess.run(s_a, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
                cur_ds = sess.run(ds, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
                print('Training raw similarity score is:\n', cur_sa)
                print('Training similarity score is:\n', similarity)

            print('Training cost:', cur_cost)
            print('Training accuracy is: ', tr_accuracy[t])

            if DEV_TEST:
                X_dev_4d = np.reshape(X_tx[0:batch_size], [batch_size, 1, nframes, nmf])
                #X_dev_4d = np.reshape(X_batch[batch_size*i:batch_size*(i+1)], [batch_size, 1, nframes, nmf])
                Z_dev = Z_tx_vgg[0:batch_size]
                #Z_dev = Z_batch[batch_size*i:batch_size*(i+1)]

                similarity_dev = sess.run(s, feed_dict={X_in:X_dev_4d, Z_penult_vgg:Z_dev})

                if PRINT_SIM:
                    cur_sa_dev = sess.run(s_a, feed_dict={X_in:X_dev_4d, Z_penult_vgg:Z_dev})
                    print('Development raw similarity score is:\n', cur_sa_dev)
                    print('Development similarity score is:\n', similarity_dev)

                cur_cost = sess.run(cost, feed_dict={X_in:X_dev_4d, Z_penult_vgg:Z_dev})
                # Find the indices of the images with the top 10 similarity score
                ntop = 10
                top_indices_dev = np.zeros((ntop, batch_size))
                for k in range(ntop):
                    cur_top_idx = np.argmax(similarity_dev, axis=1)
                    top_indices_dev[k] = cur_top_idx
                    # To leave out the top values that have been determined and the find the top values for the rest of the indices
                    for l in range(batch_size):
                        similarity_dev[l][cur_top_idx[l]] = -1
                # Find if the image with the matching index has the highest similarity score
                dev = abs(top_indices_dev - np.linspace(0, batch_size-1, batch_size))
                min_dev = np.amin(dev, axis=0)
                dev_accuracy[t] = np.mean((min_dev==0))
            
                print('Development accuracy is: ', dev_accuracy[t])

            #print('Top 10 indices is:\n', top10_indices)
            #print('\n')

# Save training accuracy
np.savetxt('train_accuracy_scnn.txt', tr_accuracy)
np.savetxt('dev_accuracy_scnn.txt', dev_accuracy)

##############
# Save Model #
##############
if SAVE_META:
    tf.add_to_collection('scnn_vars', w_in)
    tf.add_to_collection('scnn_vars', b_in)
    tf.add_to_collection('scnn_vars', w_hidden1)
    tf.add_to_collection('scnn_vars', b_hidden1)
    tf.add_to_collection('scnn_vars', w_hidden2)
    tf.add_to_collection('scnn_vars', b_hidden2)
    tf.add_to_collection('scnn_vars', w_out)
    tf.add_to_collection('scnn_vars', b_out)
    tf.add_to_collection('scnn_vars', s_a)
    tf.add_to_collection('scnn_vars', s)
    tf.add_to_collection('scnn_in', X_in)
    tf.add_to_collection('scnn_hidden', h1)
    tf.add_to_collection('scnn_hidden', h2)
    tf.add_to_collection('scnn_hidden', h3)
    tf.add_to_collection('scnn_hidden', h4)

    tf.add_to_collection('vgg_vars', w_embed)
    tf.add_to_collection('vgg_vars', b_embed)
    tf.add_to_collection('vgg_in', Z_penult_vgg)
    tf.add_to_collection('vgg_hidden', Z_embed_vgg)

    saver = tf.train.Saver()
    saver.save(sess, 'speech2image')

###########
# Testing #
###########
#scnn_test(X_batch, Z_batch, ntop)

#runtime = timeit() - begin_time
#print('Total runtime:', runtime)
# Save the parameters of the scnn
#X_tr_4d = X_tr.reshape([ntr, 1, nframes, nmf])
_w_in = sess.run(w_in, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
_b_in = sess.run(b_in, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
_w_hidden1 = sess.run(w_hidden1, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
_b_hidden1 = sess.run(b_hidden1, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
_w_hidden2 = sess.run(w_hidden2, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
_b_hidden2 = sess.run(b_hidden2, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
_w_out = sess.run(w_out, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
_b_out = sess.run(b_out, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})


if SAVE_NPZ:
    scnn_pmtrs = [_w_in, _b_in, _w_hidden1, _b_hidden1, _w_hidden2, _b_hidden2, _w_out, _b_out]
    _w_embed = sess.run(w_embed, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
    _b_embed = sess.run(b_embed, feed_dict={X_in:X_batch_4d, Z_penult_vgg:Z_batch})
    np.savez('scnn_pmtrs.npz', scnn_pmtrs)
    np.savez('vgg_pmtrs.npz', [_w_embed, _b_embed])

if EVAL_TEST:
    print('Start testing ...')
    # Test the ANN model (SCNN)
    #X_tx_4d = X_batch_4d
    X_tx_4d = np.reshape(captions_tx[0:ntx], [ntx, 1, nframes, nmf])
    #np.reshape(captions_tx[0:batch_size], [batch_size, 1, nframes, nmf])
    #Z_tx = Z_batch
    Z_tx = im_tx[0:ntx]
    #Z_tx = im_tx[0:batch_size]
    similarity = sess.run(s, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})


    if PRINT_MODEL:
        print('Input weight:', _w_in.shape)
        print(_w_in)
        print('Input bias', _b_in.shape)
        print(_b_in)
        print('First Hidden weight:', _w_hidden1.shape)
        print(_w_hidden1)
        print('First Hidden bias:', _b_hidden1.shape)
        print(_b_hidden1)
        print('Second Hidden weight:', _w_hidden2.shape)
        print(_w_hidden2)
        print('VGG embed weight:', _w_embed)
        print('VGG embed bias:', _b_embed)

    if PRINT_HIDDEN_OUT:
        _X_zm = sess.run(X_zm, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('Input after mean subtraction: ', _X_zm.shape)
        print(_X_zm[:, :, 200:800])
    
        _h1 = sess.run(h1, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('first hidden layer:', _h1.shape)
        print(_h1)
    
        _h2 = sess.run(h2, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('second hidden layer:', _h2.shape)
        print(_h2)

        _h3 = sess.run(h3, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('third hidden layer:', _h3.shape)
        print(_h3)

        _h4_re = sess.run(h4_re, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('fourth hidden layer:', _h4_re.shape)
        print(_h4_re)

        _h4_ren = sess.run(h4_ren, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('fourth hidden layer after normalization:', _h4_ren.shape)
        print(_h4_ren)

        _Z_embed_vgg = sess.run(Z_embed_vgg, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('vgg embedding layer:', _Z_embed_vgg.shape)
        print(_Z_embed_vgg)

    if PRINT_ACT:
        _a1_conv = sess.run(a1_conv, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('activation of first hidden layer:', _a1_conv)
        _a2_conv = sess.run(a2_conv, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('activation of second hidden layer:', _a2_conv)
        _a3_conv = sess.run(a3_conv, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('activation of third hidden layer:', _a3_conv)

    if PRINT_SIM:
        similarity_raw = sess.run(s_a, feed_dict={X_in:X_tx_4d, Z_penult_vgg:Z_tx})
        print('raw similarity:', similarity_raw)

    #test_accuracy = sess.run(accuracy, feed_dict={X_in:X_tx_4d, Z_in:Z_tx})
    ntop = 10
    top_indices = np.zeros((ntop, batch_size))
    for k in range(ntop):
        # Find the most similar image feature of the speech feature on the penultimate feature space
        cur_top_idx = np.argmax(similarity, axis=1)
        top_indices[k] = cur_top_idx
        # To leave out the top values that have been determined and the find the top values for the rest of the indices
        for l in range(ntx):
            similarity[l][cur_top_idx[l]] = -1

    # Find the image with the highest similarity score
    dev = abs(top_indices - np.linspace(0, batch_size-1, batch_size))
    min_dev = np.amin(dev, axis=0)

    # Count the number of correct matching by counting the number of 0s in dev
    test_accuracy = np.mean(min_dev == 0)
    print('Test accuracy is: ', str(test_accuracy))
