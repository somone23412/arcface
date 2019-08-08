export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR="../datasets/faces_casia"

NETWORK=r34
JOB=casia-base
MODELDIR="../models/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
CUDA_VISIBLE_DEVICES='0,1,2,3' ../../../anaconda2/bin/python -u quyan_train.py \
--data-dir $DATA_DIR \
--network "$NETWORK" \
--loss-type 4 \
--lr-steps 60000,100000,140000 \
--prefix "$PREFIX" \
--per-batch-size 128 \
2>&1|tee "$LOGFILE"

