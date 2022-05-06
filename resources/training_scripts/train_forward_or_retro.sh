THIS_DIR="/some/user/directory"

MODEL_TYPE="forward"  # or "retro"

# Where to get the preprocessed data from (see `preprocess_retro.sh` and `preprocess_forward.sh`)
PREPROCESSED_DATA_DIR="${THIS_DIR}/${MODEL_TYPE}_preprocessed_dir"

# Where to save the models
SAVE_MODEL_DIR="${THIS_DIR}/${MODEL_TYPE}_models_dir"
mkdir -p ${SAVE_MODEL_DIR}

SEED=42

WANDB_PROJECT_NAME="some_project_name"
WANDB_RUN_NAME="some_run_name"


onmt_train \
  -accum_count 4 \
  -adam_beta1 0.9 \
  -adam_beta2 0.998 \
  -batch_size 6144 \
  -batch_type tokens \
  -data ${PREPROCESSED_DATA_DIR}/preprocessed  \
  -decay_method noam \
  -decoder_type transformer \
  -dropout 0.1 \
  -encoder_type transformer \
  -global_attention general \
  -global_attention_function softmax \
  -gpu_ranks 0 \
  -heads 8 \
  -keep_checkpoint 20 \
  -label_smoothing 0.0 \
  -layers 4 \
  -learning_rate 2 \
  -max_generator_batches 32 \
  -max_grad_norm 0  \
  -normalization tokens \
  -optim adam \
  -param_init 0  \
  -param_init_glorot \
  -position_encoding \
  -report_every 1000 \
  -rnn_size 384 \
  -save_checkpoint_steps 5000 \
  -save_model ${SAVE_MODEL_DIR}/model  \
  -seed $SEED \
  -self_attn_type scaled-dot \
  -share_embeddings \
  -train_steps 500000 \
  -transformer_ff 2048 \
  -valid_batch_size 8 \
  -valid_steps 10000 \
  -wandb \
  -wandb_project_name ${WANDB_PROJECT_NAME} \
  -wandb_run_name ${WANDB_RUN_NAME} \
  -warmup_steps 8000 \
  -word_vec_size 384
