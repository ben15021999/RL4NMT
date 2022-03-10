export CUDA_VISIBLE_DEVICES=0
binFile=./tensor2tensor/bin


PROBLEM=translate_envi_wmt15
MODEL=transformer
HPARAMS=envi_wmt15_transformer_rl_delta_setting
# HPARAMS=envi_wmt15_transformer_rl_delta_setting_random
# HPARAMS=envi_wmt15_transformer_rl_total_setting
# HPARAMS=envi_wmt15_transformer_rl_total_setting_random
# HPARAMS=envi_wmt15_transformer_rl_delta_setting_random_baseline
# HPARAMS=envi_wmt15_transformer_rl_delta_setting_random_mle

DATA_DIR=./transformer_data/envi

TRAIN_DIR=./model/${HPARAMS}
TMP_DIR=./t2t_datagen
mkdir -p  $TRAIN_DIR


${binFile}/t2t-trainer \
--t2t_usr_dir=./envi_wmt15 \
--tmp_dir=$TMP_DIR \
--data_dir=$DATA_DIR \
--problems=$PROBLEM \
--model=$MODEL \
--hparams_set=$HPARAMS \
--output_dir=$TRAIN_DIR \
--train_steps=3000000 \
--save_checkpoints_steps=500 \
--keep_checkpoint_max=5 \
--local_eval_frequency=10000000 \
--hparams='batch_size=64,learning_rate=0.0001' \
--eval_steps=3 \
--worker_gpu=1

