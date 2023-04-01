import random

from torch import nn


class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens, mask_ratio, mask_last_token=False):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True
        self.mask_last_token = mask_last_token

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def mask_last_token_func(self):
        mask = list(range(int(self.num_tokens)))
        self.masked_tokens = mask[len(mask) - 1:]
        self.unmasked_tokens = mask[:len(mask)-1]
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        if not self.mask_last_token:
            self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        else:
            self.unmasked_tokens, self.mask_last_token = self.mask_last_token_func()
        return self.unmasked_tokens, self.masked_tokens
