python run_joint.py \
  --model_type mbert \
  --model_name_or_path  bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --train_file data/train_data.json \
  --predict_file data/valid_data.json \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --version_2_with_negative  \
  --output_dir checkpoint
 # --output_dir /kaggle/working/ \