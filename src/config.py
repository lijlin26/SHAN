import argparse
from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.004, 'learning rate'),
        'dropout': (0.6, 'feature dropout probability in DGL.GAT'),
        'atten_drop': (0.6, 'attention weight drop ratio in DGL.GAT'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (300, 'maximum number of epochs to train for'),   
        'weight_decay': (0.0005, 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'patience': (50, 'patience for early stopping'),
        'min_epochs': (100, 'do not early stop before min-epochs')
    },
    'model_config': {
        'dim': (64, 'embedding dimension'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': ('None', 'hyperbolic radius, set to None for trainable curvature'),
        'num_layers': (1, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'alpha': (0.05, 'alpha for leakyrelu in graph attention networks'),
        'use_att': (1, 'whether to use hyperbolic attention or not'),
        'sample_times': (0, 'times of sampling target nodes randomly '),
        'K': (3, 'the max order of simplices to construct'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'decoder_residual': (1, 'whether to use residual in DGL.GAT decoder'),
        'decoder': ('gat', 'which decoder to use, can be any of [linear, gat]'),
        'encoder_heads': (3, 'heads of DGL.GAT encoder'),
        'decoder_heads': (3, 'heads of DGL.GAT decoder'),
    },
    'data_config': {
        'dataset': ('dblp', 'which dataset to use'),
        'threshold': ('111', 'the minimum common non-target neighbors'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)