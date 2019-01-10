import sys
sys.path.insert(0, "..")
from bokeh.plotting import figure, save
from bokeh.resources import Resources
from utils.data_loader import SNLIDataset
from utils.params import ChrLevelCnnParams, SequenceEncoderParams, SNLIFullModelParams, TopLayerParams, \
    SNLIActivatorParams
from utils.snli_full_model import SNLIModel
from utils.snli_model_activator import SNLIActivator


def _load_datasets(train_src, pre_trained_src, dev_src=None):
    ds_dev = None
    ds_train = SNLIDataset(train_src, pre_trained_src)
    if dev_src:
        ds_dev = SNLIDataset(dev_src)
        ds_dev.load_word_vocabulary(ds_train.word_vocabulary)
    return ds_train, ds_dev


def _get_model(ds):
    model_params = \
        SNLIFullModelParams(ChrLevelCnnParams(chr_vocab_dim=ds.len_chars_vocab),
                            SequenceEncoderParams(word_vocab_dim=ds.len_words_vocab, pre_trained=ds.word_embed_mx),
                            SequenceEncoderParams(word_vocab_dim=ds.len_words_vocab),
                            TopLayerParams())
    return SNLIModel(model_params)


def _get_activator(model, ds_train, ds_dev):
    params_ = SNLIActivatorParams()
    return SNLIActivator(model, params_, ds_train, ds_dev), params_


def _plot_loss_and_acc(loss_train, loss_dev, acc_train, acc_dev, model_name, loss=True, acc=True):
    header = "loss" if loss else "acc"
    header = "loss and accuracy" if (loss and acc) else header
    if "fig" not in os.listdir(os.path.join("..")):
        os.mkdir(os.path.join("..", "fig"))
    p = figure(plot_width=600, plot_height=250, title="SNLI - Train/Dev " + header,
               x_axis_label="epochs", y_axis_label=header)
    if loss:
        x1, y1 = get_x_y_axis(loss_train)
        x2, y2 = get_x_y_axis(loss_dev)
        p.line(x1, y1, line_color='red', legend="loss Train")
        p.line(x2, y2, line_color='orange', legend="loss Dev")
    if acc:
        x3, y3 = get_x_y_axis(acc_train)
        x4, y4 = get_x_y_axis(acc_dev)
        p.line(x3, y3, line_color='green', legend="accuracy Train")
        p.line(x4, y4, line_color='blue', legend="accuracy Dev")
    p.legend.background_fill_alpha = 0.5
    save(p, os.path.join("..", "fig", model_name + " " + header + ".html"),
         title=model_name + " " + header + ".html", resources=Resources(mode="inline"))


def get_x_y_axis(curve):
    x_axis = []
    y_axis = []
    for x, y in curve:
        x_axis.append(x)
        y_axis.append(y)
    return x_axis, y_axis


if __name__ == "__main__":
    import os
    import pickle
    from utils.params import TRAIN_SRC, DEV_SRC, PRE_TRAINED_SRC
    args = sys.argv

    # TODO DELETE
    # args = ["", "test_model", os.path.join("..", PRE_TRAINED_SRC), os.path.join("..", TRAIN_SRC),
    #         os.path.join("..", DEV_SRC)]

    model_name_ = args[1]
    pre_trained_ = args[2]
    train_src_ = args[3]
    dev_src_ = None if len(args) < 4 else args[4]

    # prepare data and model
    train_, dev_ = _load_datasets(train_src_, pre_trained_, dev_src_)
    snli_model_ = _get_model(train_)
    snli_activator_, params_ = _get_activator(snli_model_, train_, dev_)

    # train
    snli_activator_.train()

    # output results
    if "pkl" not in os.listdir(os.path.join("..")):
        os.mkdir(os.path.join("..", "pkl"))
    pickle.dump((snli_activator_.model, params_, train_.word_vocabulary),
                open(os.path.join("..", "pkl", model_name_ + ".trained_model"), "wb"))
    pickle.dump(snli_activator_.get_loss_and_accuracy, open(os.path.join("..", "pkl", model_name_ + ".loss_acc_plot"), "wb"))
    loss_train_, loss_dev_, acc_train_, acc_dev_ = snli_activator_.get_loss_and_accuracy()
    _plot_loss_and_acc(loss_train_, loss_dev_, acc_train_, acc_dev_, model_name_)
    _plot_loss_and_acc(loss_train_, loss_dev_, acc_train_, acc_dev_, model_name_, loss=False)
    _plot_loss_and_acc(loss_train_, loss_dev_, acc_train_, acc_dev_, model_name_, acc=False)
