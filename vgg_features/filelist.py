#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:43:34 2017
@author: danny
"""
import glob
import os
import argparse
# create a list of all the jpeg files and paths for the output files for 
# the extract function

parser = argparse.ArgumentParser(description='create file paths')
parser.add_argument('inputdir')
parser.add_argument('outdir')


args = parser.parse_args()

images = glob.glob(os.path.join(args.inputdir, '*jpeg' ))

for im in images:
    base = os.path.basename(im)
    name, ext = os.path.splitext(base)
    outname = os.path.join(args.outdir, name + '.fea')
    print(im, outname)
    
