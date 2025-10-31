import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceSigmoid(nn.Module):
    """
    Sigmoid variant of ArcFace (Additive Angular Margin Loss)
    Inspired by SigLIP + ArcFace.
    Output logits suitable for BCEWithLogitsLoss.
    Note: really this is a classifier head with a learned weight parameter.
    """
    def __init__(self, feat_dim, num_class, margin_arc=0.35, margin_am=0.0, scale=32):
        super(ArcFaceSigmoid, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, num_class)) #these weights are for classifier wieghts, for the classification heads
        nn.init.xavier_uniform_(self.weight)
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale
        self.cos_m = math.cos(margin_arc)
        self.sin_m = math.sin(margin_arc)
        self.th = math.cos(math.pi - margin_arc)
        self.mm = math.sin(math.pi - margin_arc) * margin_arc  # optional clamp term

    def forward(self, feats, labels=None):
        # Normalize features and weights
        feats = F.normalize(feats)
        weights = F.normalize(self.weight, dim=0)

        # Compute cosine similarities
        cosine = torch.matmul(feats, weights)
        cosine = cosine.clamp(-1.0, 1.0)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Add angular margin
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.margin_am)

        if labels is not None:
            # one-hot encoding for multi-class case
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            # replace true-class cosine with margin-adjusted version
            output = one_hot * phi + (1.0 - one_hot) * cosine
        else:
            # if labels are not provided, return normal cosine scores
            output = cosine

        # scale logits for stability
        output = output * self.scale

        # key difference: we don't apply softmax — use sigmoid downstream
        return output
