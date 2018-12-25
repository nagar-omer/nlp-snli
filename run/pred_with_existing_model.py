import sys

from utils.data_loader import SNLIDataset
from utils.snli_model_activator import SNLIActivator


def _get_activator(model_name):
    model, params, vocab = pickle.load(open(os.path.join("..", "pkl", model_name + ".trained_model"), "rb"))
    return SNLIActivator(model, params), params, vocab


def _get_test_dataset(test_src, vocab):
    ds_test = SNLIDataset(test_src)
    ds_test.load_word_vocabulary(vocab)
    return ds_test


def _to_files(model_name, results, loss, accuracy):
    loss_acc = open(model_name + "_loss_acc.txt", "wt")
    loss_acc.write("accuracy = " + str(accuracy) + "\nloss = " + str(loss))
    loss_acc.close()

    res = open(model_name + ".pred", "wt")
    for label, premise, hypothesis in results:
        res.write(label + "\t\t" + premise + "\t\t" + hypothesis + "\n")
    res.close()


if __name__ == "__main__":
    import os
    import pickle
    from utils.params import TRAIN_SRC
    args = sys.argv

    # TODO DELETE
    args = ["", "test_model", os.path.join("..", TRAIN_SRC)]

    model_name_ = args[1]
    test_src_ = args[2]

    activator_, params_, vocab_ = _get_activator(model_name_)
    ds_test_ = _get_test_dataset(test_src_, vocab_)
    results_, loss_, accuracy_ = activator_.predict(ds_test_)
    _to_files(model_name_, results_, loss_, accuracy_)
