from torch.nn import Module, Linear
from torch.nn.functional import softmax
from utils.params import TopLayerParams
import torch


class TopLayerModel(Module):
    def __init__(self, params: TopLayerParams):
        super(TopLayerModel, self).__init__()
        # useful info in forward function
        self._layer0 = Linear(params.LINEAR_in_dim, params.LINEAR_hidden_dim_0)
        self._layer1 = Linear(params.LINEAR_hidden_dim_0, params.LINEAR_hidden_dim_1)
        self._output_layer = Linear(params.LINEAR_hidden_dim_1, params.LINEAR_out_dim)
        self._activation = params.Activation

    def forward(self, premise, hypothesis):
        x = torch.cat([premise, hypothesis, (premise - hypothesis), (premise * hypothesis)], dim=1)
        x = self._layer0(x)
        x = self._activation(x)
        x = self._layer1(x)
        x = self._activation(x)
        x = self._output_layer(x)
        x = softmax(x, dim=1)
        return x


if __name__ == "__main__":
    import os
    from utils.data_loader import SNLIDataset
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, ChrLevelCnnParams, SequenceEncoderParams
    from torch.utils.data import DataLoader
    from utils.cnn_character_level_model import CharacterCnnEmbed
    from utils.bi_lstm_sequence_model import SequenceEncoderModel

    ds = SNLIDataset(os.path.join("..", TRAIN_SRC), os.path.join("..", PRE_TRAINED_SRC))
    chr_params_ = ChrLevelCnnParams(chr_vocab_dim=ds.len_chars_vocab)
    word_params_ = SequenceEncoderParams(word_vocab_dim=ds.len_words_vocab, pre_trained=ds.word_embed_mx)
    top_layer_params_ = TopLayerParams()
    chr_model = CharacterCnnEmbed(chr_params_)
    seq_model = SequenceEncoderModel(word_params_)
    top_layer_model = TopLayerModel(top_layer_params_)
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )
    for i, (p, h, pw, hw, pc, hc, label) in enumerate(dl):
        premise_v = seq_model(pw, chr_model(pc))
        hypothesis_v = seq_model(hw, chr_model(hc))
        out = top_layer_model(premise_v, hypothesis_v)
        e = 0
