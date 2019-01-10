from torch.nn import MaxPool1d, Module, Embedding, LSTM, AvgPool1d, Dropout
from utils.params import SequenceEncoderParams
import torch


class SequenceEncoderModel(Module):
    def __init__(self, params: SequenceEncoderParams):
        super(SequenceEncoderModel, self).__init__()
        # word embed layer
        self._embeddings = self._load_pre_trained(params.EMBED_pre_trained, params.GPU) if params.EMBED_use_pre_trained\
            else Embedding(params.EMBED_vocab_dim, params.EMBED_dim)
        # Bi-LSTM layers
        self._lstm_layer_0 = LSTM(params.EMBED_dim + params.EMBED_chr_dim, params.LSTM_hidden_dim, params.LSTM_layers,
                                  batch_first=True, bidirectional=True)
        self._lstm_layer_1 = LSTM(params.EMBED_dim + params.EMBED_chr_dim + (2 * params.LSTM_hidden_dim),
                                  params.LSTM_hidden_dim, params.LSTM_layers, batch_first=True, bidirectional=True)
        self._lstm_layer_2 = LSTM(params.EMBED_dim + params.EMBED_chr_dim + (2 * params.LSTM_hidden_dim),
                                  params.LSTM_hidden_dim, params.LSTM_layers, batch_first=True, bidirectional=True)
        self._dropout_0 = Dropout(p=params.LSTM_dropout_0)
        self._dropout_1 = Dropout(p=params.LSTM_dropout_1)
        self._dropout_2 = Dropout(p=params.LSTM_dropout_2)

    @staticmethod
    def _load_pre_trained(weights_matrix, gpu, non_trainable=False):
        weights_matrix = torch.Tensor(weights_matrix).cuda() if gpu else torch.Tensor(weights_matrix).cuda()
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer

    # implement attention  Main paper -- 3.3 Composition Layer --
    def _calc_attention_coefficients(self):
        # get LSTM gate parameters
        # w_ii, w_if, w_ic, w_io = self._lstm_layer.weight_ih_l0.chunk(4, 0)
        w_hi, w_hf, w_hc, w_ho = self._lstm_layer_2.weight_hh_l0.chunk(4, 0)
        reverse_w_hi, w_hf, w_hc, w_ho = self._lstm_layer_2.weight_hh_l0_reverse.chunk(4, 0)
        norm_out_gates = torch.norm(torch.cat([w_hi, reverse_w_hi], dim=0), dim=1)
        attention_coefficient = norm_out_gates / torch.sum(norm_out_gates)
        return attention_coefficient

    def forward(self, words_embed, chr_rep):
        attention_coefficients = self._calc_attention_coefficients()
        # dynamic average and max pool according to batch sentence length
        activate_avg_pool = AvgPool1d(words_embed.shape[1], 1)
        activate_max_pool = MaxPool1d(words_embed.shape[1], 1)

        # embed_word concat with embed chr level -> Bi-LSTM layer
        x = self._embeddings(words_embed)
        x = torch.cat([x, chr_rep], dim=2)

        # 3 layers Bi-LSTM + skip connections + dropout layers in between
        output_seq, _ = self._lstm_layer_0(x)
        output_seq = self._dropout_0(output_seq)
        output_seq, _ = self._lstm_layer_1(torch.cat([x, output_seq], dim=2))
        output_seq = self._dropout_1(output_seq)
        output_seq, _ = self._lstm_layer_2(torch.cat([x, output_seq], dim=2))
        output_seq = self._dropout_2(output_seq)

        # final vec + attention
        avg_pool = activate_avg_pool(output_seq.transpose(1, 2)).squeeze(dim=2)
        max_pool = activate_max_pool(output_seq.transpose(1, 2)).squeeze(dim=2)
        gate_attention = torch.sum(output_seq * attention_coefficients, dim=1)
        x = torch.cat([gate_attention, avg_pool, max_pool], dim=1)
        return x


if __name__ == "__main__":
    import os
    from utils.data_loader import SNLIDataset
    from utils.params import TRAIN_SRC, PRE_TRAINED_SRC, ChrLevelCnnParams
    from torch.utils.data import DataLoader
    from utils.cnn_character_level_model import CharacterCnnEmbed

    ds = SNLIDataset(os.path.join("..", TRAIN_SRC), os.path.join("..", PRE_TRAINED_SRC))
    chr_params_ = ChrLevelCnnParams(chr_vocab_dim=ds.len_chars_vocab)
    word_params_ = SequenceEncoderParams(word_vocab_dim=ds.len_words_vocab, pre_trained=ds.word_embed_mx)
    chr_model = CharacterCnnEmbed(chr_params_)
    seq_model = SequenceEncoderModel(word_params_)
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )
    for i, (p, h, pw, hw, pc, wc, label) in enumerate(dl):
        out = seq_model(pw, chr_model(pc))
        e = 0
