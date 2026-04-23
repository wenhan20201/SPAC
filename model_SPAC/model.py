import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import torchvision
import os
import warnings
from utils import pad_features
from loss import (
    prop_topk_loss,
    decay_weight,
    SS_weight_decay,
    PC_loss,
    SS_loss,
    SC_loss
)

warnings.filterwarnings("ignore", category=UserWarning)

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device) 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

class Backbone_Proposal(torch.nn.Module):
    """
    Backbone for single modal in P-MIL framework
    """
    def __init__(self, feat_dim, n_class, dropout_ratio, roi_size):
        super().__init__()
        embed_dim = feat_dim // 2
        self.roi_size = roi_size

        self.prop_fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.prop_classifier = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, n_class+1, 1),
        )
        self.prop_attention = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )
        self.prop_completeness = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )

    def forward(self, feat):
        """
        Inputs:
            feat: tensor of size [B, M, roi_size, D]

        Outputs:
            prop_cas:  tensor of size [B, C, M]
            prop_attn: tensor of size [B, 1, M]
            prop_iou:  tensor of size [B, 1, M]
        """
        feat1 = feat[:, :,                   : self.roi_size//6  , :].max(2)[0]
        feat2 = feat[:, :, self.roi_size//6  : self.roi_size//6*5, :].max(2)[0]
        feat3 = feat[:, :, self.roi_size//6*5:                   , :].max(2)[0]
        feat = torch.cat((feat2-feat1, feat2, feat2-feat3), dim=2)

        feat_fuse = self.prop_fusion(feat)                              # [B, M, D]
        feat_fuse = feat_fuse.transpose(-1, -2)                         # [B, D, M]

        prop_cas = self.prop_classifier(feat_fuse)                      # [B, C, M]
        prop_attn = self.prop_attention(feat_fuse)                      # [B, 1, M]
        prop_iou = self.prop_completeness(feat_fuse)                    # [B, 1, M]

        return prop_cas, prop_attn, prop_iou, feat_fuse




    
