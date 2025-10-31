import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLipLoss(nn.Module):
    """
    Implementation of the Sigmoid contrastive loss from
    'Scaling Vision-Language Models with Sigmoid Loss' (Google, 2023).
    Works for CLIP-style models with image/text embeddings.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_embeds, txt_embeds):
        # Normalize embeddings
        img_embeds = F.normalize(img_embeds, dim=-1)
        txt_embeds = F.normalize(txt_embeds, dim=-1)

        # Similarity matrix (batch_size x batch_size)
        logits = torch.matmul(img_embeds, txt_embeds.T) / self.temperature

        # Binary targets: positive if i==j, else negative
        targets = torch.eye(logits.size(0), device=logits.device)

        # BCE with logits
        loss_i = F.binary_cross_entropy_with_logits(logits, targets)
        loss_t = F.binary_cross_entropy_with_logits(logits.T, targets)

        # Symmetric loss (image->text + text->image)
        loss = (loss_i + loss_t) / 2
        return loss
