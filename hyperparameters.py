"""
This module defines the hyperparametres for the network
"""
dropout_value = 0.20        # not used
LR = 0.01                   # learning rate      
MOMENTUM = 0.9              
MILESTONES = [10, 20, 30]   # schedule for adjusting LR
GAMMA = 0.1                 # LR adjust rate
EPOCHS=1
BATCH_SIZE = 300