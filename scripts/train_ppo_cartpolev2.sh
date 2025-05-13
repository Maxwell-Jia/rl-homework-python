#! /bin/bash

python -m rl_zoo3.train \
    --algo ppo \
    --env CartPole-v2 \
    --vec-env subproc \
    --seed 42 \
    --eval-freq 50000 \
    --n-eval-envs 4 \
    --save-freq 50000 \
    --log-interval -2 \
    -P \
    --device cpu \
    --track \
    --wandb-project-name "rl-homework-python" \
    --wandb-entity "Maxwell-Jia" \