python finetune_polyglot.py \
    --base_model 'EleutherAI/polyglot-ko-1.3b' \
    --data_path data/step1_inst_format.jsonl \
    --output_dir ckpt/ \
    --prompt_template_name kullm \
    --batch_size 128 \
    --micro_batch_size 1 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 1024 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "[query_key_value]" \
    --train_on_inputs \
    --eval_steps 0 \
    --weight_decay 0.1 \
    --warmup_steps 0 \
    --warmup_ratio 0.1 \
    --add_eos_token true \
    --lr_scheduler_type "cosine"

