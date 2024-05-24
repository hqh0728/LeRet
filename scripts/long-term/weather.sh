if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=LeRet
str='weather'
if [ ! -d "./logs/LongForecasting/$str" ]; then
    mkdir ./logs/LongForecasting/$str
fi
dir=./logs/LongForecasting/$str
root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom


random_seed=2021
patch_len=8
stride=8

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
    --enc_in 21 \
    --dropout 0.0\
    --e_layers 1 \
    --batch_size 128 \
    --learning_rate 5e-3 \
    --loss 'mae' \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 128 \
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 10\
    --loss $loss\
    --train_epochs 50\
    --gpu 0\
    --itr 1  >$dir/$model_id_name'_'$seq_len'_'$pred_len.log 
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
    --enc_in 21 \
    --dropout 0.0\
    --e_layers 1 \
    --batch_size 1024 \
    --learning_rate 5e-3 \
    --loss 'mae' \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 128 \
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 10\
    --loss $loss\
    --train_epochs 50\
    --gpu 0\
    --itr 1  >$dir/$model_id_name'_'$seq_len'_'$pred_len.log 
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
    --enc_in 21 \
    --dropout 0.0\
    --e_layers 2 \
    --batch_size 1024 \
    --learning_rate 5e-3 \
    --loss 'mae' \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 128 \
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 10\
    --loss $loss\
    --train_epochs 50\
    --gpu 0\
    --itr 1  >$dir/$model_id_name'_'$seq_len'_'$pred_len.log 
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
    --enc_in 21 \
    --dropout 0.0\
    --e_layers 1 \
    --batch_size 1024 \
    --learning_rate 5e-3 \
    --loss 'mae' \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 128 \
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 10\
    --loss $loss\
    --train_epochs 50\
    --gpu 0\
    --itr 1  >$dir/$model_id_name'_'$seq_len'_'$pred_len.log 