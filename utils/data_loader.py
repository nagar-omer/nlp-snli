from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
import torch
from utils.params import NATURAL, CONTRADICTION, ENTAILMENT, PAD, START, END, UNKNOWN, EMBEDDING_DIM, PRE_TRAINED_SRC, \
    TRAIN_SRC


# Dataset:
#   - src_path:     path to SNLI data file (train/test/dev)
#   - vocab_path:   path to vocabulary file (article uses GloVe vectors - https://nlp.stanford.edu/projects/glove/
class SNLIDataset(Dataset):
    def __init__(self, src_path, vocab_src=None):
        self._base_dir = __file__.replace("/", os.sep)                       # absolute path to base project dir
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._char_embed = {chr(i): i for i in range(128)}                   # ASCII characters to idx vocab
        self._char_embed[PAD] = 128
        self._word_embed, self._idx_to_word, self._word_embed_mx = \
            self._load_vocab(vocab_src) if vocab_src else (None, None, None) # words vocab
        self._label_to_idx = {ENTAILMENT: 0, CONTRADICTION: 1, NATURAL: 2}   # label to idx (for softmax)
        self._idx_to_label = [ENTAILMENT, CONTRADICTION, NATURAL]
        self._data = self._read_file(src_path)                               # data = [ .. (premise, hypo, label) .. ]

    def label(self, idx):
        return self._idx_to_label[idx]

    def __len__(self):
        return len(self._data)

    @property
    def word_vocabulary(self):
        return self._word_embed, self._idx_to_word

    def load_word_vocabulary(self, vocab):
        self._word_embed, self._idx_to_word = vocab

    @property
    def len_chars_vocab(self):
        return len(self._char_embed)

    @property
    def len_words_vocab(self):
        return len(self._word_embed)

    @property
    def word_embed_mx(self):
        return self._word_embed_mx

    # read SNLI source file to data list [ ..(premise, hypothesis, label).. ]
    @staticmethod
    def _read_file(src_path):
        data = []
        src = open(src_path, "rt", encoding="utf-8")
        next(src).split()                                                   # skip header
        for row in src:
            label, _, _, _, _, premise, hypothesis, _, _, _, _, _, _, _ = row.split("\t")
            if label == "-":
                continue
            data.append((premise.split(), hypothesis.split(), label))
        return data

    # read vocabulary + vectors from file (or from pkl if possible)
    def _load_vocab(self, vocab_src):
        # load pickle if exists
        pkl_path = os.path.join(self._base_dir, "pkl", vocab_src.rsplit(os.sep, 1)[1].strip(".txt") + ".pkl")
        if os.path.exists(pkl_path):
            return pickle.load(open(pkl_path, "rb"))

        vocab_list = [START, END, PAD, UNKNOWN]                         # fixed tags
        mx_list = [np.zeros(EMBEDDING_DIM), np.zeros(EMBEDDING_DIM),
                   np.zeros(EMBEDDING_DIM), np.zeros(EMBEDDING_DIM)]    # init vectors to zero vectors
        src = open(vocab_src, "rt", encoding="utf-8")
        for row in src:
            word, vec = row.split(" ", 1)
            mx_list.append(np.fromstring(vec, sep=" "))                 # append pre trained vector
            vocab_list.append(word)                                     # append word
        mx = np.vstack(mx_list)                                         # concat vectors

        vocab = {word: i for i, word in enumerate(vocab_list)}
        # save as pickle
        pickle.dump((vocab, vocab_list, mx), open(pkl_path, "wb"))
        return vocab, vocab_list, mx

    # get word level embedding + character level embeddings for a sentence
    def _get_full_embed(self, sentence):
        # embed word level - each word given an idx from loaded vocab (or UNKNOWN idx)
        embed_word_level = []
        for i, word in enumerate(sentence):
            embed_word_level.append(self._word_embed.get(word.lower(), self._word_embed[UNKNOWN]))

        # embed char level each word given a vector of char embeddings
        embed_char_level = []
        max_len_char = 0
        for i, word in enumerate(sentence):
            embed_i = []
            for c in word:
                embed_i.append(self._char_embed[c])
            embed_char_level.append(embed_i)
            max_len_char = len(embed_i) if len(embed_i) > max_len_char else max_len_char     # get max len word

        return embed_word_level, embed_char_level, max_len_char

    # function for torch Dataloader - creates batch matrices using Padding
    def collate_fn(self, batch):
        lengths_premise_words = []
        lengths_premise_chrs = []
        lengths_hypothesis_words = []
        lengths_hypothesis_chrs = []
        labels = []
        # calculate max word len + max char len
        for sample in batch:
            lengths_premise_words.append(sample[2])
            lengths_premise_chrs.append(sample[6])
            lengths_hypothesis_words.append(sample[3])
            lengths_hypothesis_chrs.append(sample[9])
            labels.append(sample[10])
        # in order to pad all batch to a single dimension max length is needed
        max_premise_words = np.max(lengths_premise_words)
        max_premise_chrs = np.max(lengths_premise_chrs)
        max_hypothesis_words = np.max(lengths_hypothesis_words)
        max_hypothesis_chrs = np.max(lengths_hypothesis_chrs)

        # new batch variables
        premise_batch = []
        hypothesis_batch = []
        premise_word_embed_batch = []
        hypothesis_word_embed_batch = []
        premise_char_embed_batch = []
        hypothesis_char_embed_batch = []
        for sample in batch:
            # original sentences as lists no need to pad
            premise_batch.append(sample[0])
            hypothesis_batch.append(sample[1])
            # pad word vectors
            premise_word_embed_batch.append(sample[4] + [self._word_embed[PAD]] * (max_premise_words - len(sample[4])))
            hypothesis_word_embed_batch.append(sample[7] + [self._word_embed[PAD]] * (max_hypothesis_words - len(sample[7])))
            # pad char vector at char level and at word level
            temp = [[self._char_embed[PAD]] * (max_premise_chrs - len(chars)) + chars for chars in sample[5]]
            premise_char_embed_batch.append([[self._char_embed[PAD]] * max_premise_chrs] *
                                            (max_premise_words - len(sample[4])) + temp)
            temp = [[self._char_embed[PAD]] * (max_hypothesis_chrs - len(chars)) + chars for chars in sample[8]]
            hypothesis_char_embed_batch.append([[self._char_embed[PAD]] * max_hypothesis_chrs] *
                                               (max_hypothesis_words - len(sample[7])) + temp)

        return premise_batch, hypothesis_batch, torch.Tensor(premise_word_embed_batch).long(), \
               torch.Tensor(hypothesis_word_embed_batch).long(), torch.Tensor(premise_char_embed_batch).long(), \
               torch.Tensor(hypothesis_char_embed_batch).long(), torch.Tensor(labels).long()

    def __getitem__(self, index):
        premise, hypothesis, label = self._data[index]                                               # sentences
        prem_words_embed, prem_letters_embed, prem_max_word_len = self._get_full_embed(premise)      # full embeddings
        hypo_words_embed, hypo_letters_embed, hypo_max_word_len = self._get_full_embed(hypothesis)   # full embeddings
        return premise, hypothesis, len(premise), len(hypothesis), \
               prem_words_embed, prem_letters_embed, prem_max_word_len, \
               hypo_words_embed, hypo_letters_embed, hypo_max_word_len, self._label_to_idx[label]


if __name__ == "__main__":
    ds = SNLIDataset(os.path.join("..", TRAIN_SRC), os.path.join("..", PRE_TRAINED_SRC))
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )
    for i, (p, h, pw, hw, pc, wc, label) in enumerate(dl):
        print(i, p, pw, pc)
