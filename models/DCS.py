import torch 
import torch.nn as nn


class DeepwiseSeparableConv(nn.Module):
    
    def __init__(self, n_in, n_out):
        
        super(DeepwiseSeparableConv, self).__init__()
        self.deptwise = nn.Conv3d(n_in, n_in, kernel_size= (3, 3, 3), padding=(1, 1, 1). groups = n_in)
        self.
        