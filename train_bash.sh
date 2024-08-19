CONFIGS=${1}
GPUS=${2:-'1'}
PY_ARGS=${@:3}
PORT=$(( RANDOM % 100 + 12300 ))
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=${GPUS} --master_port=${PORT} scripts/train.py \
 ${CONFIGS} --no-validate --launcher pytorch ${PY_ARGS} --seed 42
