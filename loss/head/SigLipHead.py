import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLipHead(nn.Module):
    """
    Sigmoid (SigLIP-style) head in ArcFace/AM-Softmax format.
    Takes normalized features and class weights, outputs scaled logits.
    Can be used with BCEWithLogitsLoss.
    """
    def __init__(self, feat_dim, num_class, scale=16, temperature=0.07):
        super(SigLipHead, self).__init__()
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
