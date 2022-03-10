PROBLEM=translate_envi_wmt15 # We chose a problem translation English to French with 32.768 vocabulary
MODEL=transformer # Our model
HPARAMS=envi_wmt15_transformer_rl_delta_setting


DATA_DIR=./transformer_data/envi/ # This folder contain the data
TMP_DIR=./t2t_datagen/
TRAIN_DIR=./model/$HPARAMS # This folder contain the model
EXPORT_DIR=TRAIN_DIR # This folder contain the exported model for production
TRANSLATIONS_DIR=./translation/ # This folder contain  all translated sequence
EVENT_DIR=./event/ # Test the BLEU score
USR_DIR=./envi_wmt15/ # This folder contains our data that we want to add

DECODE_FILE=$DATA_DIR/decode_this.txt
REF_FILE=$DATA_DIR/ref.vn

BEAM_SIZE=10
ALPHA=1.1

./tensor2tensor/bin/t2t-decoder \
--t2t_usr_dir=$USR_DIR \
--data_dir=$DATA_DIR \
--problem=$PROBLEM \
--model=$MODEL \
--hparams_set=$HPARAMS \
--output_dir=$TRAIN_DIR \
--decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,batch_size=32" \
--decode_from_file=$DECODE_FILE \
--decode_to_file=$DATA_DIR/translation.en