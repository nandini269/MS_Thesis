#!/bin/bash
python3 ensemble_alg.py --dataset "cifar10" --filtered True --num_iters 5 --num_epochs 5
python3 ensemble_alg.py --dataset "mnist" --filtered False --num_iters 5 --num_epochs 5
python3 ensemble_alg.py --dataset "cifar10" --filtered False --num_iters 5 --num_epochs 5
