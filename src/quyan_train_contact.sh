export MXNET_CPU_WORKER_NTHREADS=12
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
# export MXNET_ENABLE_GPU_P2P=0

DATA_DIR="../datasets/faces_casia"

NETWORK=r34
JOB=casia-contact
MODELDIR="../models/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
CUDA_VISIBLE_DEVICES='6,7' ../../../anaconda2/bin/python -u quyan_train.py \
--data-dir $DATA_DIR \
--network "$NETWORK" \
--loss-type 4 \
--pretrained "../models/model-r34-casia-base/model,140" \
--lr 0.001 \
--lr-steps 40000,60000 \
--verbose 2000 \
--prefix "$PREFIX" \
--per-batch-size 128 \
2>&1|tee "$LOGFILE"


# --lr-steps 60000,100000,140000 \
# --lr-steps 100000,140000,160000 \