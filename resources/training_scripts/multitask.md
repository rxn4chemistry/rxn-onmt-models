# Commands to use for multi-task


Preprocess:
```
onmt_preprocess \
  -train_src ${SUB1_DIR}/precursors-train.txt ${SUB2_DIR}/precursors-train.txt \
  -train_tgt ${SUB1_DIR}/product-train.txt ${SUB2_DIR}/product-train.txt \
  -train_ids sub1 sub2 \
  -valid_src ${SUB2_DIR}/precursors-valid.txt \
  -valid_tgt ${SUB2_DIR}/product-valid.txt \
  -save_data ${PREPROCESSED_DATA_DIR}/preprocessed \
  -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab
```


Train:
```
onmt_train \
  -accum_count 4 \
  -adam_beta1 0.9 \
  -adam_beta2 0.998 \
  -batch_size 6144 \
  -batch_type tokens \
  -data ${PREPROCESSED_DATA_DIR}/preprocessed  \
  -data_ids sub1 sub2 \
  -data_weights ${SUB1_WEIGHT} ${SUB2_WEIGHT} \
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
  -wandb_project_name client-ch1-common-training \
  -wandb_run_name ${RUN_NAME} \
  -warmup_steps 8000 \
  -word_vec_size 384
```

Continue training:
Necessary ones for continuing. 
Note (ava, May 11): I went through what happens in the opennmt code and the ones below really play a role.
```
onmt_train \
  -accum_count 4 \
  -batch_size 6144 \
  -batch_type tokens \
  -data ${PREPROCESSED_DATA_DIR}/preprocessed  \
  -data_ids sub1 sub2 \
  -data_weights ${SUB1_WEIGHT} ${SUB2_WEIGHT} \
  -dropout 0.1 \
  -gpu_ranks 0 \
  -keep_checkpoint 20 \
  -label_smoothing 0.0 \
  -max_generator_batches 32 \
  -normalization tokens \
  -report_every 1000 \
  -reset_optim none \
  -save_checkpoint_steps 5000 \
  -save_model ${CONTINUE_SAVE_MODEL_DIR}/model  \
  -seed $SEED \
  -train_from ${SAVE_MODEL_DIR}/model_step_260000.pt \
  -train_steps 500000 \
  -valid_batch_size 8 \
  -valid_steps 10000 \
  #-wandb \
  #-wandb_project_name client-ch1-common-training \
  #-wandb_run_name ${RUN_NAME} \
```
