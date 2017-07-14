# lr=0.001
# gamma=0.95
# kernel_out=1

# for batch in 4 8 16 32
# do
#     for worlds in 5 10 20
#     do
#         save_path=logs/reinforce_${lr}lr_${gamma}gamma_${batch}batch_${kernel_out}attn_${worlds}worlds
#         mkdir ${save_path}
#         sbatch  -c 2 --gres=gpu:titan-x:1 --qos=tenenbaum --time=4-12:0 --mem=40G -J ${worlds}_${batch}_${lr} -o ${save_path}/out.txt run_reinforce.py \
#                 --lr ${lr} --attention_out_dim ${kernel_out} --num_worlds ${worlds} --save_path ${save_path} \
#                 --batch_size ${batch} --gamma ${gamma}
#     done
# done

lr=0.001
gamma=0.95
epochs=1250

for rep in 1
do
    for batch in 4
    do
        for model in norbf #uvfa-text cnn-lstm
        do
            for mode in local global
            do
                for num_turk in 1566
                do
                    save_path=rl_logs_analysis/qos_${epochs}epochs_reinforce_${model}model_${mode}mode_${num_turk}turk_${lr}lr_${gamma}gamma_${batch}batch_${rep}rep
                    mkdir ${save_path}
                    sbatch  -c 2 --gres=gpu:titan-x:1 --qos=tenenbaum --time=4-12:0 --mem=40G -J ${rep}${model}${num_turk}_${batch}_${lr} -o ${save_path}/_out.txt turk_reinforce.py \
                            --model ${model} --mode ${mode} --num_turk ${num_turk} \
                            --lr ${lr} --save_path ${save_path} \
                            --batch_size ${batch} --gamma ${gamma} --epochs ${epochs}
                done
            done
        done
    done
done

# -o ${save_path}/_out.txt