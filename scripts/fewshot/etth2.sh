if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/fewshot" ]; then
    mkdir ./logs/LongForecasting/fewshot
fi

seq_len=336
root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2-fewshot
data_name=ETTh2
model_name=LeRet
if [ ! -d "./logs/LongForecasting/fewshot/$data_name" ]; then
    mkdir ./logs/LongForecasting/fewshot/$data_name
fi
dir=./logs/LongForecasting/fewshot/$data_name

percent=10
random_seed=2021
seq_len=336

pred_len=96
python -u run_longExp.py \
    --percent $percent \
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
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --e_layers 2 \
    --batch_size 256 \
    --learning_rate 5e-3 \
    --loss 'mae'\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 8\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --patience 10\
    --gpu 2\
    --itr 1   >$dir/$model_id_name'_'$seq_len'_'$pred_len.log  
pred_len=192
python -u run_longExp.py \
    --percent $percent \
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
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --e_layers 2 \
    --batch_size 256 \
    --learning_rate 5e-3 \
    --loss 'mae'\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 8\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --patience 10\
    --gpu 2\
    --itr 1   >$dir/$model_id_name'_'$seq_len'_'$pred_len.log  

pred_len=336
python -u run_longExp.py \
    --percent $percent \
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
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --e_layers 1 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --loss 'mae'\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 8\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --patience 10\
    --gpu 2\
    --itr 1   >$dir/$model_id_name'_'$seq_len'_'$pred_len.log 

pred_len=720
python -u run_longExp.py \
    --percent $percent \
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
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --e_layers 1 \
    --batch_size 64 \
    --learning_rate 5e-3 \
    --loss 'mae'\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 8\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --patience 10\
    --gpu 2\
    --itr 1   >$dir/$model_id_name'_'$seq_len'_'$pred_len.log 