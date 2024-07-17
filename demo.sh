CONFIGS=${1}
GPUS=${2:-'1'}
PY_ARGS=${@:3}
PORT=$(( RANDOM % 100 + 12300 ))
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=${GPUS} --master_port=${PORT}  scripts/demo.py \
 ${CONFIGS} --launcher pytorch ${PY_ARGS}