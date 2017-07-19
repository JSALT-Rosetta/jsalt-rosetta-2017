export EESEN_ROOT=/pylon2/ir3l68p/metze/eesen
export PATH=$PWD/utils/:$EESEN_ROOT/src/netbin:$EESEN_ROOT/src/featbin:$EESEN_ROOT/src/decoderbin:$EESEN_ROOT/src/fstbin:$EESEN_ROOT/tools/openfst/bin:$EESEN_ROOT/../kaldi/src/featbin:$PWD:$PATH

# export KALDI_ROOT=/home/fciannel/team/tools/kaldi-cuda
# [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
# export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
# [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
# . $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C


if [[ `uname -n` =~ br0* || `uname -n` =~ r0* || `uname -n` =~ gpu0* ]]; then
  # PSC Bridges
  module unload cuda/7.5
  module load atlas
  module load cuda/8.0
  module load gcc
  module load python3

  #d=`mktemp -d`
  #virtualenv $d
  #source ~/tf-daily-3/bin/activate
  if [[ `uname -n` =~ gpu0* ]]; then
    source ~/tf-1.2.1/bin/activate
  else
    source ~/tf-1.2.1-nogpu/bin/activate
  fi
  #pip install ~/tensorflow_gpu-1.head-cp35-cp35m-manylinux1_x86_64.whl
  #Module load anaconda/4.2.0-3.5.2
  #source activate /opt/packages/TensorFlow/anaconda/TensorFlowEnv_1.0
  #echo $PYTHONPATH
  #module load tensorflow/0.12.1
  export PYTHONPATH=$PYTHONPATH:/pylon2/ir3l68p/metze/eesen-tf-new/tf/ctc-train
  #export PYTHONPATH=$PYTHONPATH:/opt/packages/TensorFlow/anaconda/TensorFlowEnv_1.0/lib/python3.6/site-packages
  #module load tensorflow/0.12.1
  #echo $PYTHONPATH  
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pylon5/ci560op/fciannel/kaldi/src/lib:
  # export LD_LIBRARY_PATH=/home/metze/lib64
  export TMPDIR=$LOCAL
  #export TMPDIR=/pylon1/ir3l68p/metze
fi

[ -f ../../tools/env.sh ] && . ../../tools/env.sh

