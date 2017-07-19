#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --output=log/slurm-%j.out
#SBATCH --export=ALL
#SBATCH --time="48:00:00"

uname -a
date

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

############## Variables from Odette Model #################################
minlmwt=1
maxlmwt=25
expname=$1
feats_nj=80
train_nj=80
decode_nj=30


datadir="data"
expdir="exp"
mfccdir="mfcc"

lmdir="${datadir}/lmdir"
langdir="${datadir}/lang"
localdir="${datadir}/local"
dictdir="${datadir}/local/dict"
resourcedir="resources"

# Acoustic model parameters

numLeavesTri1=7500
numGaussTri1=40000
numLeavesMLLT=7500
numGaussMLLT=40000
numLeavesSAT=7500
numGaussSAT=40000
numGaussUBM=800
numLeavesSGMM=10000
numGaussSGMM=20000
# ====================================================================


stage=0

. parse_options.sh

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation                                      "
  echo =====================================================================
  local/flickr_data_prep.py /pylon5/ci560op/fciannel/flickraudio/flickr_audio/ /pylon2/ci560op/odette/data/flickr/flickr_labels/ $datadir
  utils/utt2spk_to_spk2utt.pl ${datadir}/totaldata/utt2spk > ${datadir}/totaldata/spk2utt

for x in totaldata; do
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 ${datadir}/$x ${datadir}/train ${datadir}/val
done 


fi

if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  for set in train val; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 28 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done
fi

# if [ $stage -le 3 ]; then
#   echo =====================================================================
#   echo "                        Network Training                           "
#   echo =====================================================================
#   # Specify network structure and generate the network topology
#   lstm_layer_num=6     # number of LSTM layers
#   lstm_cell_dim=140    # number of memory cells in every LSTM layer
#   input_dim=80
#   proj_dim=80

#   dir=exp/train_phn_l${lstm_layer_num}_c${lstm_cell_dim}
#   mkdir -p $dir

#   utils/sym2int.pl -f 2- data/lang_phn/units.txt data/val/text   | gzip -c > $dir/labels.cv.gz
#   utils/sym2int.pl -f 2- data/lang_phn/units.txt data/train/text | gzip -c > $dir/labels.tr.gz

#   # Train the network with CTC. Refer to the script for details about the arguments
#   steps/train_ctc_tf.sh \
#     --nlayer $lstm_layer_num --nhidden $lstm_cell_dim --nproj $proj_dim --feat_proj $input_dim \
#     data/train data/val $dir

# fi

# #   echo =====================================================================
# #   echo "                            Decoding                               "
# #   echo =====================================================================
# #   # Config for the basic decoding: --beam 30.0 --max-active 5000 --acoustic-scales "0.7 0.8 0.9"
# # #  for lm_suffix in tgpr tg; do
# # #    steps/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 10 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 \
# # #      data/lang_phn_test_${lm_suffix} data/test_dev93 $dir/decode_dev93_${lm_suffix} || exit 1;
# # #    steps/decode_ctc_lat.sh --cmd "$decode_cmd" --nj 8 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 \
# # #      data/lang_phn_test_${lm_suffix} data/test_eval92 $dir/decode_eval92_${lm_suffix} || exit 1;
# # #  done
# # fi

# # date
