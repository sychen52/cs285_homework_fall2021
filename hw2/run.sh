#! /bin/bash

# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
# -dsa --exp_name q1_sb_no_rtg_dsa
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
# -rtg -dsa --exp_name q1_sb_rtg_dsa
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
# -rtg --exp_name q1_sb_rtg_na
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -dsa --exp_name q1_lb_no_rtg_dsa
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -rtg -dsa --exp_name q1_lb_rtg_dsa
# python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
# -rtg --exp_name q1_lb_rtg_na

#python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
#--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 300 -lr 1e-2 -rtg \
#--exp_name q2_b500_r5e-3

#python cs285/scripts/run_hw2.py \
#--env_name LunarLanderContinuous-v2 --ep_len 1000 \
#--discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 \
#--reward_to_go --nn_baseline --exp_name q3_b40000_r0.005

# Experiment 4
# for b in 10000 30000 50000
# do
#     for r in 0.005 0.01 0.02
#     do
#         python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
#                --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline \
#                --exp_name q4_search_b${b}_lr${r}_rtg_nnbaseline
#     done
# done

# The larger the lr, the better the performance;
# however, when batch size is not large enough (10k), it explodes half way
# The larger the batch size, the more stable the curve (less variance),
# and slightly better performance

# Experiment 5
b=50000
r=0.02
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b ${b} -lr ${r} \
--exp_name q4_b${b}_r${r}
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b ${b} -lr ${r} -rtg \
--exp_name q4_b${b}_r${r}_rtg
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b ${b} -lr ${r} --nn_baseline \
--exp_name q4_b${b}_r${r}_nnbaseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b ${b} -lr ${r} -rtg --nn_baseline \
--exp_name q4_b${b}_r${r}_rtg_nnbaseline
