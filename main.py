# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import argparse
from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from env_runner.environment import Environment
from adaptation.test_time_adaptation import Adapter

def add_train_args(parser):
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Meta learning rate')
    parser.add_argument('--adapt-lr', type=float, default=1e-2, help='Adaptation learning rate')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')

def add_eval_args(parser):
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--improve', type=str, choices=['attention', 'bn'], help='Improvement method')

def add_adapt_args(parser):
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--episodes', type=int, default=20, help='Number of adaptation episodes')
    parser.add_argument('--adapt-steps', type=int, default=10, help='Adaptation steps')
    parser.add_argument('--improve', type=str, choices=['attention', 'bn'], help='Improvement method')

def train(args):
    env = Environment()
    ssm = SSM()
    meta_maml = MetaMAML(ssm, meta_lr=args.meta_lr, adapt_lr=args.adapt_lr)
    meta_maml.train(env, episodes=args.episodes)
    if args.checkpoint:
        meta_maml.save(args.checkpoint)

def evaluate(args):
    env = Environment()
    ssm = SSM()
    meta_maml = MetaMAML(ssm)
    meta_maml.load(args.checkpoint)
    results = meta_maml.evaluate(env, episodes=args.episodes)
    print(f'Evaluation results: {results}')

def adapt(args):
    env = Environment()
    ssm = SSM()
    meta_maml = MetaMAML(ssm)
    meta_maml.load(args.checkpoint)
    adapter = Adapter(meta_maml)
    adapter.adapt(env, episodes=args.episodes, steps=args.adapt_steps)
    results = adapter.evaluate(env)
    print(f'Adaptation results: {results}')

def main():
    parser = argparse.ArgumentParser(description='SSM MetaRL Training and Testing')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    add_train_args(train_parser)
    
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    add_eval_args(eval_parser)
    
    adapt_parser = subparsers.add_parser('adapt', help='Adapt the model')
    add_adapt_args(adapt_parser)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'adapt':
        adapt(args)

if __name__ == '__main__':
    main()
