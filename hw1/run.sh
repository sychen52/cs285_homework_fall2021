#! /bin/bash

# This is the result the expert policy (I hack the code to get this)
# Eval_AverageReturn : 4758.9013671875
# Eval_StdReturn : 102.37667083740234
# Eval_MaxReturn : 4894.041015625
# Eval_MinReturn : 4632.6181640625
# Eval_AverageEpLen : 1000.0
# Train_AverageReturn : 4713.6533203125
# Train_StdReturn : 12.196533203125
# Train_MaxReturn : 4725.849609375
# Train_MinReturn : 4701.45654296875
# Train_AverageEpLen : 1000.0
# Train_EnvstepsSoFar : 0
# TimeSinceStart : 3.378551483154297
# Initial_DataCollection_AverageReturn : 4713.6533203125

# make the training steps 10x longer so that it is fair to compare with Dagger

for exp in Ant
do
    python cs285/scripts/run_hw1.py \
           --expert_policy_file cs285/policies/experts/$exp.pkl \
           --env_name $exp-v2 --exp_name bc_$exp --n_iter 1 \
           --expert_data cs285/expert_data/expert_data_$exp-v2.pkl \
           --batch_size 2000 \
           --num_agent_train_steps_per_iter 1000 \
           --eval_batch_size 5000 \
           --video_log_freq -1 \
           --n_layers 1 --size 64  -lr 0.005
done

for exp in Ant
do
    python cs285/scripts/run_hw1.py \
           --expert_policy_file cs285/policies/experts/$exp.pkl \
           --env_name $exp-v2 --exp_name dagger_$exp --n_iter 10 \
           --do_dagger --expert_data cs285/expert_data/expert_data_$exp-v2.pkl \
           --batch_size 200 \
           --num_agent_train_steps_per_iter 100 \
           --eval_batch_size 5000 \
           --video_log_freq -1 \
           --n_layers 1 --size 64  -lr 0.005
done


exp=Humanoid
python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/$exp.pkl \
        --env_name $exp-v2 --exp_name bc_$exp --n_iter 1 \
        --expert_data cs285/expert_data/expert_data_$exp-v2.pkl \
        --batch_size 2000 \
        --num_agent_train_steps_per_iter 10000 \
        --eval_batch_size 5000 \
        --video_log_freq -1 \
        --n_layers 1 --size 64  -lr 0.0005

python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/$exp.pkl \
        --env_name $exp-v2 --exp_name dagger_$exp --n_iter 30 \
        --do_dagger --expert_data cs285/expert_data/expert_data_$exp-v2.pkl \
        --batch_size 1000 \
        --num_agent_train_steps_per_iter 1000 \
        --eval_batch_size 5000 \
        --video_log_freq -1 \
        --n_layers 1 --size 64  -lr 0.0005
