MODEL_TYPE="forward"  # or "retro"

# Which model to use for the predictions
THIS_DIR="/some/user/directory"
SAVE_MODEL_DIR="${THIS_DIR}/${MODEL_TYPE}_models_dir"
SAVED_MODEL="${SAVE_MODEL_DIR}/model_step_250000.pt"

# Where the data is located
DATA_DIR="/some/data/directory"
SPLIT="valid"

# Where to save the prediction
PREDICTION_FILE="/path/to/predictions"

# Common settings
BATCH_SIZE=64

# settings that are different for forward or retro
if [[ ${MODEL_TYPE} = "retro" ]]
then
  TOPN=10
  BEAM_SIZE=15
  MAX_LENGTH=300
  TARGET_FILE="$DATA_DIR/precursors-${SPLIT}.txt"
  TEST_FILE="$DATA_DIR/product-${SPLIT}.txt"
else
  TOPN=2
  BEAM_SIZE=10
  MAX_LENGTH=300
  TEST_FILE="$DATADIR/precursors-${SPLIT}.txt"
  TARGET_FILE="$DATADIR/product-${SPLIT}.txt"
fi

onmt_translate \
  -model ${SAVED_MODEL} \
  -src ${TEST_FILE} \
  -tgt ${TARGET_FILE} \
  -output ${PREDICTION_FILE} \
  -log_probs \
  -n_best ${TOPN} \
  -beam_size ${BEAM_SIZE} \
  -max_length ${MAX_LENGTH} \
  -batch_size ${BATCH_SIZE} \
  -gpu 0
