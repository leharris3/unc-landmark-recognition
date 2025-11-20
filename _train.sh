LOGS_DIR=__exps__/__logs__
EXP_ROOT_DIR=__exps__/cifar10
EXP_NAME="resnet18-bs128-lr1e-5-CIFAR10-full-ft"

CONFIG_FP="configs/full-ft.json"

python train.py \
    --config $CONFIG_FP > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit