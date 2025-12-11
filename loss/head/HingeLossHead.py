import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeLossHead(nn.Module):
    """
    Head using Multiclass Hinge Loss (Crammer and Singer).
    Takes normalized features and class weights, outputs scaled logits, which is used as input for the hinge loss
    Need to use nn.MultiMarginLoss in main.py. Can tune margin param which has default val of 1
    """

    #same implementaiton for cosine as siglip
    def __init__(self, feat_dim, num_class, scale=16, temperature=0.07):
        super(HingeLossHead, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.temperature = temperature

    def forward(self, feats, labels=None):
        # Normalize features and class weights
        feats = F.normalize(feats)
        weights = F.normalize(self.weight, dim=0)

        # Cosine similarity between features and class weights
        logits = torch.mm(feats, weights) / self.temperature
        logits = logits * self.scale  # optional scaling for stability

        return logits
