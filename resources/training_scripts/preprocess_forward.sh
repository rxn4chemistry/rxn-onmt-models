THIS_DIR="/some/user/directory"
DATA_DIR="/some/data/directory"

# Where the preprocessed data will be saved
PREPROCESSED_DATA_DIR="${THIS_DIR}/forward_preprocessed_dir"
mkdir -p ${PREPROCESSED_DATA_DIR}

onmt_preprocess \
  -train_src ${DATA_DIR}/precursors-train.txt \
  -train_tgt ${DATA_DIR}/product-train.txt \
  -valid_src ${DATA_DIR}/precursors-valid.txt \
  -valid_tgt ${DATA_DIR}/product-valid.txt \
  -save_data ${PREPROCESSED_DATA_DIR}/preprocessed \
  -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab
