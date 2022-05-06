# In principle, the script is very similar to the one for training.
# The relevant changes are:
# - `reset_optim` set to none
# - `train_from` indicating which model to start from
# Probably, many of the options below can even be left out, 
# since they will be overwritten by the pre-trained model.


THIS_DIR="/some/user/directory"

MODEL_TYPE="forward"  # or "retro"

# Where to get the preprocessed data from (see `preprocess_retro.sh` and `preprocess_forward.sh`)
PREPROCESSED_DATA_DIR="${THIS_DIR}/${MODEL_TYPE}_preprocessed_dir"

SEED=42

WANDB_PROJECT_NAME="some_project_name"
WANDB_RUN_NAME="some_run_name_continued"

# Where the models from the initial run are
SAVE_MODEL_DIR="${THIS_DIR}/${MODEL_TYPE}_models_dir"
MODEL_TO_START_FROM=${SAVE_MODEL_DIR}/model_step_123456.pt

# Where to save the models
# Note: This can also simply be the same as $SAVE_MODEL_DIR
CONTINUE_SAVE_MODEL_DIR="${THIS_DIR}/${MODEL_TYPE}_models_dir_continued"
mkdir -p ${CONTINUE_SAVE_MODEL_DIR}


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
  -reset_optim none \
  -rnn_size 384 \
  -save_checkpoint_steps 5000 \
  -save_model ${CONTINUE_SAVE_MODEL_DIR}/model  \
  -seed $SEED \
  -self_attn_type scaled-dot \
  -share_embeddings \
  -train_from ${SAVE_MODEL_DIR}/model_step_260000.pt \
  -train_steps 500000 \
  -transformer_ff 2048 \
  -valid_batch_size 8 \
  -valid_steps 10000 \
  -wandb \
  -wandb_project_name ${WANDB_PROJECT_NAME} \
  -wandb_run_name ${WANDB_RUN_NAME} \
  -warmup_steps 8000 \
  -word_vec_size 384
