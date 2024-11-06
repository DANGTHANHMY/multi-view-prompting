set -ex

export CUDA_VISIBLE_DEVICES=0

cd multi-view-prompting/src

for SEED in 5 10 15 20 25
do
K=5
INFER_PATH=$K
CTRL_TOKEN=post
TASK=unified
OUT_DIR="../outputs/$TASK/top${K}_seed${SEED}"

mkdir -p $OUT_DIR


python main.py \
    --data_path "../data/" \
    --dataset seed$SEED \
    --model_name_or_path t5-base \
    --output_dir $OUT_DIR \
    --num_train_epochs 20 \
    --save_top_k 0 \
    --task $TASK \
    --top_k $K \
    --ctrl_token $CTRL_TOKEN \
    --multi_path \
    --beam_size 2 \
    --num_path $INFER_PATH \
    --seed $SEED \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lowercase \
    --sort_label \
    --data_ratio 1.0 \
    --check_val_every_n_epoch 10  \
    --agg_strategy vote \
    --eval_batch_size 32 \
    --constrained_decode \
    --multi_task \
    --do_train \
    > $OUT_DIR/train.log
    # batch_size: 16, eval_batch_size = 64
    # --load_ckpt_name "best.ckpt" \
    # --load_path_cache \
    # > $OUT_DIR/train.log 2>&1&
done
