#! /bin/bash

# python -m rl_zoo3.enjoy --algo ppo --env CartPole-v2 --folder logs --load-best --seed 1234

python -m rl_zoo3.record_video \
    --algo ppo \
    --env CartPole-v2 \
    --folder logs \
    --load-best \
    --seed 1234 \
    --output-folder videos