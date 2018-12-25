from torch import nn
from torch.nn import Conv2d, MaxPool1d
import torch
from utils.params import ChrLevelCnnParams


# character CNN model: based on, Convolution Neural Networks for Sentence Classification, Yoon Kim, New York University
# + Natural Language Processing (almost) from Scratch, Ronan Collobert
# input:  character embed matrix representing single word
# flow:
# - Embedding layer (char-level)
# - CNN layer - 3 filters with different kernel combined with max_pool
# - MaxPool over Channels
# - Linear layer
# - Softmax
class CharacterCnnEmbed(nn.Module):
    def __init__(self, params: ChrLevelCnnParams):
        super(CharacterCnnEmbed, self).__init__()
        # Embedding layer
        self._embeddings = nn.Embedding(params.EMBED_vocab_dim, params.EMBED_dim)
        # Filters - 3 window size
        self._filter1 = Conv2d(params.CNN_chanel_in, params.CNN_chanel_out, params.CNN1_Kernel_dim, params.CNN_Stride)
        self._filter2 = Conv2d(params.CNN_chanel_in, params.CNN_chanel_out, params.CNN2_Kernel_dim, params.CNN_Stride)
        self._filter3 = Conv2d(params.CNN_chanel_in, params.CNN_chanel_out, params.CNN3_Kernel_dim, params.CNN_Stride)
        # for calculating out dim
        self._max_pool1_fix = params.CNN1_Kernel_dim[0] - 1
        self._max_pool2_fix = params.CNN2_Kernel_dim[0] - 1
        self._max_pool3_fix = params.CNN3_Kernel_dim[0] - 1

    def forward(self, x):
        # create max pool layers - every batch can have different thus max_pool layer should be dynamic
        mp1 = MaxPool1d(x.shape[2] - self._max_pool1_fix, 1)
        mp2 = MaxPool1d(x.shape[2] - self._max_pool2_fix, 1)
        mp3 = MaxPool1d(x.shape[2] - self._max_pool3_fix, 1)
        x = self._embeddings(x).unsqueeze(dim=2)  # out_dim = [batch, max_len_words, 1, max_len_char, embed_dim]

        # 3 Filters
        # each filter is have different window size: each word is represented as concatenation of
        # the max pool of the 3 filters [ max_pool(filter1), max_pool(filter2), max_pool(filter3) ]
        x1 = torch.stack([mp1(self._filter1(x[idx, :]).squeeze(dim=3)).squeeze(dim=1) for idx in range(x.shape[0])])
        x2 = torch.stack([mp2(self._filter2(x[idx, :]).squeeze(dim=3)).squeeze(dim=1) for idx in range(x.shape[0])])
        x3 = torch.stack([mp3(self._filter3(x[idx, :]).squeeze(dim=3)).squeeze(dim=1) for idx in range(x.shape[0])])
        x = torch.cat([x1, x2, x3], dim=2)       # out_dim = [ batch_size,
        return x


if __name__ == "__main__":
    import os
    from utils.data_loader import SNLIDataset
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC
    from torch.utils.data import DataLoader

    ds = SNLIDataset(os.path.join("..", TRAIN_SRC), os.path.join("..", PRE_TRAINED_SRC))
    params_ = ChrLevelCnnParams(chr_vocab_dim=ds.len_chars_vocab)
    model = CharacterCnnEmbed(params_)
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )
    for i, (p, h, pw, hw, pc, wc) in enumerate(dl):
        out = model(pc)
        e = 0