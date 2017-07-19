#! /usr/bin/python
# -*- coding: utf-8 -*-
import json
import glob
from os import listdir
from os.path import isfile, join
from pprint import pprint
from utils import getLogger
from collections import OrderedDict
import pdb, os, glob, codecs, re, argparse, random, sys

logger = getLogger()

def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])

def get_speakers_files(file_name):
    f = codecs.open(file_name, "r", "utf-8")
    files = f.readlines()
    fileslist = [ f.strip() for f in files]
    zeroes = len(str(len(files)))
    files_dict = {}
    for file in fileslist:
        wav_file_name = file.split()[0]
        wav_file_name = wav_file_name.split('.')[0]
        speaker_name = str(file.split()[1])
        speaker_name = str(speaker_name).zfill(3)
        logger.debug("This is the speaker name: %s " %speaker_name)
        if speaker_name not in files_dict.keys():
            files_dict[speaker_name] = []
        idx = len(files_dict[speaker_name])
        counter = str(idx).zfill(zeroes)
        file_name = speaker_name+'_'+counter+'_'+wav_file_name
        speaker_list = files_dict[speaker_name]
        speaker_list.append(file_name)
        files_dict[speaker_name] = speaker_list
        logger.debug(files_dict[speaker_name])
    return files_dict

def get_phonemes(file_name, main_dir_align):
    remove_list = ['SIL', '+BREATH+', '+HUMAN+', '+LAUGH+', '+NOISE+', 'GARBAGE', '+FILLER+', '+FILLER2+']
    file_name = main_dir_align+'align_'+file_name+'.txt'
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
            phones = [p.split()[0] for p in lines if p.split()[0] not in remove_list]
        phones_string = ' '.join(phones)
    except:
        phones_string = ""
        logger.debug("The file %s does not have alignments." % file_name)
    return phones_string

def main():
    main_dir_wav = sys.argv[1]
    main_dir_align = sys.argv[2]
    data_dir = sys.argv[3]
    logger.debug("This is the wav directory: %s" % main_dir_wav)
    logger.debug("This is the json directory: %s" % main_dir_align)
    logger.debug("This is the data_word directory: %s" % data_dir)
    os.system('rm -rf '+data_dir+' && '+'mkdir -p '+data_dir)
    suffix='totaldata'

    units = set()

    os.system('mkdir -p '+data_dir+'/'+suffix)
    os.system('mkdir -p '+data_dir+'/'+'lang_phn')

    fid_txt = codecs.open(data_dir+'/'+suffix+'/text',"w","utf-8")
    fid_utt = codecs.open(data_dir+'/'+suffix+'/utt2spk',"w","utf-8")
    fid_wav = codecs.open(data_dir+'/'+suffix+'/wav.scp',"w","utf-8")
    fid_uni = codecs.open(data_dir+'/'+'lang_phn/units.txt',"w","utf-8")

    wav2spk_filename = main_dir_wav+'/wav2spk.txt'
    wav2spk_dict = get_speakers_files(wav2spk_filename)

    wav2spk_dict_sorted = OrderedDict(sorted(wav2spk_dict.items(), key=lambda k: k[0]))
    n_files = 37974
    logger.debug(wav2spk_dict_sorted)

    speakers = wav2spk_dict_sorted.keys()
#    speakers.sort()
    logger.debug(speakers)
    i = 0
    # Creating the utt2spk file and the wav.scp files
    for speaker in speakers:
        for utt_id_sorted in wav2spk_dict_sorted[speaker]:
            file_name = split_at(utt_id_sorted, '_', 2)[1]
            phones_list_string = get_phonemes(file_name, main_dir_align)
            if phones_list_string != "":
                fid_txt.write("%s %s\n" % (utt_id_sorted, phones_list_string))
                fid_utt.write("%s %s\n" % (utt_id_sorted, speaker))
                fid_wav.write("%s %s\n" % (utt_id_sorted, main_dir_wav+'wavs/'+file_name+'.wav'))
                phones_set = set(phones_list_string.split())
                units = units.union(phones_set)
            i += 1
            if i % 100 == 0:
                perc = i * 100.0 / n_files
                logger.info("Done file %d over %d total files, %f completed" % (i, n_files, perc))
    i = 1
    for unit in units:
        fid_uni.write('%s %s\n' % (unit, i))
        i += 1


if __name__ == '__main__':
    main()
