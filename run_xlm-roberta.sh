python run.py \
  --model_type xlm-r \
  --model_name_or_path  xlm-roberta-base \
  --do_train \
  --do_eval \
  --train_file train_data.json \
  --predict_file valid_data.json \
  --per_gpu_train_batch_size=6 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_query_length=64 \
  --output_dir /kaggle/working/ \
  --version_2_with_negative \
  --overwrite_output_dir \
  --save_steps 2500 \
  --n_best_size=20 \
  --max_answer_length=30