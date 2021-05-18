#!/bin/bash

# input arguments
DATA="${1-PROTEINS}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
fold=${2-0}  # which fold as testing data
test_number=${3-0}  # if specified, use the last test_number graphs as test data

# general settings
GPU='0'  # select the GPU FP_LEN=0  # final dense layer's input dimension, decided by data
bsize=50  # batch size
dropout=0.0
LATENT=64
OUT=32
pool=3

# dataset-specific settings
case ${DATA} in
MUTAG)
  num_epochs=100
  learning_rate=0.0001
  type='default'
  class=2
  ;;
PTC)
  num_epochs=100
  learning_rate=0.0001
  type='default'
  class=2
  ;;
IMDBBINARY)
  num_epochs=150
  learning_rate=0.0001
  type='deg'
  class=2
  ;;
IMDBMULTI)
  num_epochs=100
  learning_rate=0.0005
  type='deg'
  class=3
  ;;
PROTEINS)
  num_epochs=80
  learning_rate=0.0001
  type='default'
  class=2
  ;;
*)
  num_epochs=500
  learning_rate=0.00001
  ;;
esac


if [ ${fold} == 0 ]; then
  rm ${DATA}_acc_result.txt
  touch ${DATA}_acc_result.txt
  echo "Running 10-fold cross validation"
  start=`date +%s`
  for i in $(seq 1 10)
  do
    python3 main.py \
        -cuda $GPU \
        -dataset $DATA \
        -fold $i \
        -learning_rate $learning_rate \
        -num_epochs $num_epochs \
        -latent_dim $LATENT \
        -out_dim $OUT \
        -batch_size $bsize \
        -dropout $dropout \
        -feature_type $type \
        -num_class $class \
        -num_pool $pool
  done
  stop=`date +%s`
  echo "End of cross-validation"
  echo "The total running time is $[stop - start] seconds."
  echo "The accuracy results for ${DATA} are as follows:"
  cat ${DATA}_acc_result.txt
else
  python3 main.py \
        -cuda $GPU \
      	-fold $fold \
	-dataset $DATA \
        -learning_rate $learning_rate \
        -num_epochs $num_epochs \
        -latent_dim $LATENT \
        -out_dim $OUT \
        -batch_size $bsize \
        -dropout $dropout  \
        -feature_type $type \
        -num_class $class \
        -num_pool $pool
      
fi
