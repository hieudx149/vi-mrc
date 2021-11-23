CUDA_LAUNCH_BLOCKING=1 python run_classifier.py \
  --model_type xlm_roberta \
  --model_name_or_path  anhtunguyen98/xlm-base-vi \
  --do_eval \
  --train_file data/train_data.json \
  --predict_file data/valid_data.json \
  --train_batch_size=12 \
  --eval_batch_size=8 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 256 \
  --warmup_steps=814 \
  --output_dir checkpoint \
  --overwrite_output_dir
 # --output_dir /kaggle/working/ \