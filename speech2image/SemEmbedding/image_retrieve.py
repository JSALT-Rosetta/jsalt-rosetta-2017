from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
#from scnn_test import *

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

def annotate(imids):
    # Get the top n indices of image of the current speech. For now, the number of captions and images have to be
    # the same
    #scnn_test(captions, images, n)
    data = np.load('top_indices_ann.npz')
    top_ids = data['arr_0']
    [n, ndata] = top_ids.shape
    nim = imids.shape[0]
    files_sp, files_im = read_file_list(ndata)
    text_capts = read_captions(files_im)
    # Transpose to size n x ndata if the dimension of the list is fliped  
    if n > ndata:
        top_ids = np.transpose(top_ids)
        [n, ndata] = top_ids.shape
    
    # Find the images for the caption and plot it
    for i in range(nim):
        #right = (np.amin(np.abs(top_ids[:, i]-i)) == 0)
        cur_capts = []
        # If the image is correctly retrieved, show it and the rest associated with the queried caption
        #if right:
        #print('Top indices', top_ids[:, imids[i]])
        for j in range(n):
            cur_im_idx = int(top_ids[j, imids[i]])
            cur_name_im = files_im[cur_im_idx]
            cur_name_sp = files_sp[i]
            #print('Line 217 the image', cur_name_im, 'is related to the caption', cur_name_sp)
            curcapt = text_capts[cur_name_im]
            cur_capts.append(curcapt)
            #cur_im = pil2arr(cur_name_im)
            # Merge the image side-by-side
            #np.concatenate((cur_ims, cur_im), axis=1)
            # Print the captions
            print(curcapt)
        print('\n')
        # Print the current image
        plt.figure()
        right_im_name = files_im[imids[i]]
        cur_im_arr = pil2arr(right_im_name)
        plt.imshow(cur_im_arr)
        plt.axis('off')
        plt.show()

good_ids = []
data = np.load('top_indices_ann.npz')
top_ids = data['arr_0']
ndata = top_ids.shape[1]
for i in range(ndata):
    if not np.amin(np.abs(top_ids[:, i]-i)):
        good_ids.append(i)

good_ids = np.array(good_ids)
nid = 10 # Only plot the first 10 correct retrieval
annotate(good_ids[0:nid])