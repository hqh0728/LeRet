if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

random_seed=2021
patch_len=8
stride=8
if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/transfer" ]; then
    mkdir ./logs/LongForecasting/transfer
fi
seq_len=336
model_name=LeRet
str='ETTh2'
if [ ! -d "./logs/LongForecasting/transfer/$str" ]; then
    mkdir ./logs/LongForecasting/transfer/$str
fi
dir=./logs/LongForecasting/transfer/$str
root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2-mae
data_name=ETTh2

loss='mae'
for pred_len in 96 ;do #   192 336 720 
for lr in  5e-3 ;do # 1e-3 5e-4 1e-4
for batch_size in 1024 ; do # 512 128
for e_layers in 1 ; do # 2 3
for dropout in 0.0;do # 0.3  0.5 0.8
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
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 32 \
      --dropout $dropout\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len $patch_len\
      --stride $stride\
      --des 'Exp' \
      --patience 10\
      --loss $loss\
      --train_epochs 50\
      --gpu 3\
      --itr 1 --batch_size $batch_size --learning_rate $lr >$dir/$model_id_name'_'$seq_len'_'$pred_len'_'$window_size'_'$patch_len'_'$stride'_'$dropout'_'$e_layers'_'$batch_size'_'$lr'_'$loss.log 
done
done
done
done
done

# ETTh1
transfer_name=ETTh1
for pred_len in 96 ;do #   192 336 720 
for lr in  5e-3 ;do # 1e-3 5e-4 1e-4
for batch_size in 1024 ; do # 512 128
for e_layers in 1 ; do # 2 3
for dropout in 0.0;do # 0.3  0.5 0.8
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --transfer_name $transfer_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 32 \
      --dropout $dropout\
      --patch_len $patch_len\
      --stride $stride\
      --des 'Exp' \
      --loss $loss\
      --gpu 3\
      --itr 1 --batch_size $batch_size --learning_rate $lr >$dir/$model_id_name'_transfer'$transfer_name'_'$seq_len'_'$pred_len.log
done
done
done
done
done

transfer_name=ETTm2
for pred_len in 96 ;do #   192 336 720 
for lr in  5e-3 ;do # 1e-3 5e-4 1e-4
for batch_size in 1024 ; do # 512 128
for e_layers in 1 ; do # 2 3
for dropout in 0.0;do # 0.3  0.5 0.8
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --transfer_name $transfer_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 32 \
      --dropout $dropout\
      --patch_len $patch_len\
      --stride $stride\
      --des 'Exp' \
      --loss $loss\
      --gpu 3\
      --itr 1 --batch_size $batch_size --learning_rate $lr >$dir/$model_id_name'_transfer'$transfer_name'_'$seq_len'_'$pred_len.log
done
done
done
done
done