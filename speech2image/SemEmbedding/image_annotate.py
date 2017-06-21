from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
#from scnn_test import *

# This script retrieves the top n images with highest similarity scores for a given speech
def read_file_list(n):
    # Read a dict from index to filename for both image and caption
    file_info = '../data/flickr_audio/wav2capt.txt'
    files_sp = []
    files_im = []
    with open(file_info, 'r') as f:
        for i in range(n):
            files = f.readline()
            files_part = files.split()
            cur_sp = files_part[0]
            cur_im = files_part[1]
            #print(cur_sp, cur_im)
            files_sp.append(cur_sp)
            files_im.append(cur_im)
    #print(cur_im[125:129])
    return files_sp, files_im

def read_captions(captfiles):
    # Read a dict from image file to its text caption
    file_info = '../data/Flickr8k_text/Flickr8k.token.txt'
    text_capts = {}
    with open(file_info, 'r') as f:
        while len(f.readline()) > 0:
            files = f.readline()
            files_part = files.split()
            nparts = len(files_part)
            cur_sp_parts = files_part[0].split('#')
            cur_sp = cur_sp_parts[0]
            #print(len(cur_sp))
            textcap = ''
            for k in range(nparts-1):
                textcap = textcap+files_part[k+1]+' '
            #print(textcap)
            text_capts[cur_sp] = textcap
    return text_capts


def pil2arr(imfile):
    path = '../data/Flicker8k_Dataset/'
    # load the image and return
    im = Image.open(path+imfile)
    #im_data_seq = im.getdata()
    #print(np.array(im_data_seq).shape)
    #im_data_arr = np.array(im_data_seq, dtype=float).reshape(im.size[0], im.size[1], 3)
    im_data_arr = np.array(im)
    #print(im.size)
    #print(im_data_arr.shape)
    #im.show()
    return im_data_arr

def retrieve(captids):
    # Get the top n indices of image of the current speech. For now, the number of captions and images have to be
    # the same
    #scnn_test(captions, images, n)
    data = np.load('top_indices_ret.npz')
    top_ids = data['arr_0']
    [n, ndata] = top_ids.shape
    ncapt = captids.shape[0]
    files_sp, files_im = read_file_list(ndata)
    text_capts = read_captions(files_im)
    # Transpose to size n x ndata if the dimension of the list is fliped
    if n > ndata:
        top_ids = np.transpose(top_ids)
        [n, ndata] = top_ids.shape
    # Find the images for the caption and plot it
    for i in range(ncapt):
        cur_ims = []
        # If the image is correctly retrieved, show it and the rest associated with the queried caption
        for j in range(n):
            cur_im_idx = int(top_ids[j, captids[i]])
            cur_name_im = files_im[cur_im_idx]
            cur_name_sp = files_sp[i]
            #print('Line 217 the image', cur_name_im, 'is related to the caption', cur_name_sp)
            cur_im = pil2arr(cur_name_im)
        # Find the caption name
        right_im_name = files_im[i]
        caption = text_capts[right_im_name]
        # Plot the image
        plt.figure()
        nim = len(cur_ims)
        f,axarr = plt.subplots(1, nim)
        for m in range(nim):
            #print('Line 248 type of the image:', cur_ims[m].dtype)
            axarr[m].imshow(cur_ims[m], aspect=1)
            axarr[m].axis('off')
        plt.title(caption)
        plt.show()
        cur_name_parts = cur_name_im.split('.')
        tmp = cur_name_parts[0]
    #np.savez(tmp+'_top'+str(n)+'.npz', cur_ims)


'''
    cur_top_idx = np.argmax(similarity, axis=1)
    top_indices[i] = cur_top_idx
    # To leave out the top values that have been determined and find the top values for the rest of the indices
    similarity[cur_top_idx] = -1
    
    ## Plot the Mel frequency spectrogram of the speech along with the top images of the speech
    # Load data
    file_sp = 'captions.npz'#sys.argv[1]
    data = np.load(file_sp)
    captions = data['arr_0']
    
    file_im = 'images.npz'#sys.argv[2]
    #file_im = 'images.npz'
    data = np.load(file_im)
    images = data['arr_0']
    
    ntop = 5
    #int(sys.argv[3])
    #if len(sys.argv) > 4:
    #    nsel = int(sys.argv[4])
    #    captions = captions[0:nsel]
    #    images = images[0:nsel]
    captions = captions[0:200]
    images = images[0:200]
    
    retrieve(captions, images, ntop)
    
    '''
# Load top indices
#data = np.load('top_indices_im.npz')
#top_indices = np.transpose(data['arr_0'])

# Find the indices corresponding to captions correctly mapped to its image
captions_dat = np.load('captions.npz')
captions = captions_dat['arr_0']

good_ids = []
data = np.load('top_indices_ret.npz')
top_ids = data['arr_0']
ndata = top_ids.shape[1]
for i in range(ndata):
    if not np.amin(np.abs(top_ids[:, i]-i)):
        good_ids.append(i)

good_ids = np.array(good_ids)
retrieve(good_ids)

# Plot the MFCC of the good captions
#nplot = 10 #len(good_ids)
gd_mfcc = []
for l in range(good_ids):
    plt.figure()
    mfcc = captions[int(good_ids[l])]
    gd_mfcc.append(mfcc)

np.savez('gd_mfcc.npz', np.array(gd_mfcc))
#    plt.imshow(mfcc, cmap=plt.get_cmap('gray'), aspect='auto')

#ncor_capt = good_indices.shape[0]
#nplot = 10

'''for j in range(nplot):
    retrieve(top_indices[:,good_indices[j]])
    cur_capt = captions[good_indices[j]]
    # Plot the mel-freq spectrogram
    plt.figure()
    plt.imshow(cur_capt, intepolation='linear', aspect_ratio='auto')
    '''

'''
    # Image captioning
    data = np.load('top_indices_sp.npz')
    top_indices_sp = np.transpose(data['arr_0'])
    correct_indices = np.linspace(0, ncapt-1, ncapt)
    correct = (np.amin(np.abs(top_indices_sp-)) == 0)
    ncor = correct.shape[0]
    top_correct_indices = []
    
    for i in range(ncor):
    if correct[i] == 1:
    top_is.append(i)
    
    top_correct_indices = np.array(top_correct_indices)
    ncor_capt = .shape[0]
    nplot = 10
    
    for j in range(nplot):
    find_caption(top_indices[:,top_correct_indices[j]])
    cur_im = 
    plt.figure()
    plt.imshow(captions, intepolation='linear', aspect_ratio='auto')'''