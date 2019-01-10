import os
import torch
from torch.nn.functional import cross_entropy, relu
from torch.optim import Adam

# ------------------------------ Data Params ------------------------------
# names of labels according to SNLI dadaset
ENTAILMENT = "entailment"
CONTRADICTION = "contradiction"
NATURAL = "neutral"

PRE_TRAINED_SRC = os.path.join("data", "GloVe_vocab", "glove.6B.300d.txt")
TRAIN_SRC = os.path.join("data", "snli_1.0", "snli_1.0_train.txt")
DEV_SRC = os.path.join("data", "snli_1.0", "snli_1.0_dev.txt")
TEST_SRC = os.path.join("data", "snli_1.0", "snli_1.0_test.txt")
EMBEDDING_DIM = 300         # GloVe embedding dim = 300/../50
START = "<S>"               # token for sentence start / end of sentence
END = "</s>"
UNKNOWN = "UUNNKK"          # token for unknown words
PAD = "<p>"                 # token for artificial padding


# ----------------------------- Models Params -----------------------------
class ChrLevelCnnParams:
    def __init__(self, chr_vocab_dim=129):
        self.EMBED_dim = 300
        self.EMBED_vocab_dim = chr_vocab_dim
        self.CNN_chanel_in = 1
        self.CNN_chanel_out = 1
        self.CNN1_Kernel_dim = (3, self.EMBED_dim)
        self.CNN2_Kernel_dim = (4, self.EMBED_dim)
        self.CNN3_Kernel_dim = (5, self.EMBED_dim)
        self.CNN_Stride = 1


class SequenceEncoderParams:
    def __init__(self, word_vocab_dim=129, pre_trained=None, gpu=True):
        self.EMBED_pre_trained = pre_trained
        self.EMBED_use_pre_trained = True if pre_trained is not None else False
        self.EMBED_dim = 300
        self.EMBED_chr_dim = 3                  # number of filters at chr level model
        self.EMBED_vocab_dim = word_vocab_dim
        self.LSTM_hidden_dim = 300
        self.LSTM_layers = 1
        self.LSTM_dropout_0 = 0.1
        self.LSTM_dropout_1 = 0.1
        self.LSTM_dropout_2 = 0
        self.GPU = gpu
        # OUT_DIM = self.LSTM_hidden_dim * 2 * 3,  (2 for Bi-LSTM, 3 for cat [attention | max |avg])


class TopLayerParams:
    def __init__(self):
        self.LINEAR_in_dim = 24 * 300             # should be 4 * SequenceEncoderParams::OUT_DIM
        self.LINEAR_hidden_dim_0 = 4000
        self.LINEAR_hidden_dim_1 = 1000
        self.LINEAR_out_dim = 3
        self.Activation = relu


class SNLIFullModelParams:
    def __init__(self, chr_params, premise_params, hypo_params, top_layer_params):
        self.CHARACTER_params = chr_params
        self.SEQUENSE_premise_params = premise_params
        self.SEQUENCE_hypothesis_params = hypo_params
        self.TOP_LAYAER_params = top_layer_params
        self.LEARNING_RATE = 0.0005
        self.OPTIMIZER = Adam


# ----------------------------- Activator Params -----------------------------
class SNLIActivatorParams:
    def __init__(self):
        self.LOSS = cross_entropy
        self.BATCH_SIZE = 64
        self.GPU = True
        self.EPOCHS = 30
        self.VALIDATION_RATE = 200
