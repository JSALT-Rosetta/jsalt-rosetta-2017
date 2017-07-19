# -*- coding: utf-8 -*-
import string
import sys
import codecs
from unidecode import unidecode

basedir = '/home/fciannel/src/eesencgn/'
corpus_file_input = basedir+'data/totaldata/text'
corpus_file_output = basedir+'data/totaldata/textl'
lexicon_file = basedir+'resources/lexicon.txt'
fid_inp = codecs.open(corpus_file_input,"r","utf-8")
fid_out = codecs.open(corpus_file_output,"w","utf-8")
fid_lex = codecs.open(lexicon_file,"r","utf-8")
input_content = fid_inp.readlines()
lexicon = fid_lex.readlines()
utt_list = []
utt_key = []
lexicon = [ l.strip() for l in lexicon]
lexicon_dict = {key: value for (key, value) in [l.split('\t') for l in lexicon] }



def get_phonemes(w):
    try:
        wpp = lexicon_dict[w]
    except:
        wpp = ""
        if w in ['één', 'café', 'josé']:
            t = unidecode(w)
            t.encode("ascii")
#            if t == 'een':
            wpp = lexicon_dict[t]
        else:
            wpp = "@ @"
#            print("The word %s is not in the lexicon" % w)
    return wpp
        

def main():
    for l in input_content:
        l = l.strip()
        l_list = l.split()
        l_list_seq = l_list[1:]
        # utt_list.append(l_list[1:])
        # utt_key.append(l_list[0])
        phonemes = ""
        for w in l_list_seq:
            wp = get_phonemes(w)
            if wp != "":
                if phonemes == "":
                    phonemes = wp
                else:
                    phonemes += ' '+wp 
                
        fid_out.write('%s %s\n' % (l_list[0], phonemes))
    fid_out.close()
    fid_lex.close()
    fid_inp.close()




    # # This uses the 3-argument version of str.maketrans with arguments (x, y, z) where 'x' and 'y' must be equal-length strings and characters in 'x' are replaced by characters in 'y'. 'z' is a string (string.punctuation here) where each character in the string is mapped to None
    # remove_punct = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
    # translator = str.maketrans('', '', remove_punct)
    # for s in input_content:
    #     s = s.translate(translator)
    #     fid_out.write(s.lower())
"""
Create the languge model
~/team/tools/kenlm/build/bin/lmplz -o 3 <./corpus_parsed.txt >corpus_parsed.arpa
"""



if __name__ == '__main__':
    main()
