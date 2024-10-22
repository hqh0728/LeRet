if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=LeRet

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021
pred_len=96
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --e_layers 3 \
    --batch_size 24 \
    --learning_rate 5e-4 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 8\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1   >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
pred_len=192
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --e_layers 3 \
    --batch_size 24 \
    --learning_rate 5e-4 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 8\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1   >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
pred_len=336
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --e_layers 3 \
    --batch_size 24 \
    --learning_rate 5e-4 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 8\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1   >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
pred_len=720
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --e_layers 3 \
    --batch_size 24 \
    --learning_rate 5e-4 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 8\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1   >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 