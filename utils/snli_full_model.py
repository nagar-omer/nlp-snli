from torch.nn import Module
from utils.bi_lstm_sequence_model import SequenceEncoderModel
from utils.cnn_character_level_model import CharacterCnnEmbed
from utils.nn_top_layer_model import TopLayerModel
from utils.params import SNLIFullModelParams


class SNLIModel(Module):
    def __init__(self, params: SNLIFullModelParams):
        super(SNLIModel, self).__init__()
        self._chr_embed_model = CharacterCnnEmbed(params.CHARACTER_params)                    # character embed (CNN)
        self._seq_model = SequenceEncoderModel(params.SEQUENSE_premise_params)                # premise to vec (LSTM)
        # self._hypothesis_seq_model = SequenceEncoderModel(params.SEQUENCE_hypothesis_params)# hypothesis to vec (LSTM)
        self._top_layer_model = TopLayerModel(params.TOP_LAYAER_params)                       # combine with (NN)
        self.optimizer = self.set_optimizer(params.LEARNING_RATE, params.OPTIMIZER)

    # init optimizer with RMS_prop
    def set_optimizer(self, lr, opt):
        return opt(self.parameters(), lr=lr)

    def forward(self, premise_words, premise_chr, hypothesis_words, hypothesis_chr):
        premise_v = self._seq_model(premise_words, self._chr_embed_model(premise_chr))
        hypothesis_v = self._seq_model(hypothesis_words, self._chr_embed_model(hypothesis_chr))
        return self._top_layer_model(premise_v, hypothesis_v)


if __name__ == "__main__":
    import os
    from utils.data_loader import SNLIDataset
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, ChrLevelCnnParams, SequenceEncoderParams, TopLayerParams
    from torch.utils.data import DataLoader

    ds = SNLIDataset(os.path.join("..", TRAIN_SRC), os.path.join("..", PRE_TRAINED_SRC))
    params = SNLIFullModelParams(ChrLevelCnnParams(chr_vocab_dim=ds.len_chars_vocab),
                                 SequenceEncoderParams(word_vocab_dim=ds.len_words_vocab, pre_trained=ds.word_embed_mx),
                                 SequenceEncoderParams(word_vocab_dim=ds.len_words_vocab, pre_trained=ds.word_embed_mx),
                                 TopLayerParams())
    model = SNLIModel(params)
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )
    for i, (p, h, pw, hw, pc, hc, label) in enumerate(dl):
        out = model(pw, pc, hw, hc)
        e = 0
