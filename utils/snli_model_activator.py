import random

import torch
from torch.autograd import Variable
from utils.data_loader import SNLIDataset
from utils.loggers import PrintLogger
from utils.params import SNLIActivatorParams
from sys import stdout
from utils.snli_full_model import SNLIModel
from torch.utils.data import DataLoader, Subset


class SNLIActivator:
    def __init__(self, model: SNLIModel, params: SNLIActivatorParams, train: SNLIDataset=None, dev: SNLIDataset=None):
        self._model = model
        self._epochs = params.EPOCHS
        self._validation_rate = params.VALIDATION_RATE
        self._batch_size = params.BATCH_SIZE
        self._gpu = params.GPU
        self._loss_func = params.LOSS
        if self._gpu:
            self._model.cuda()
        self._load_data(train, dev)
        self._init_loss_and_acc_vec()

    @property
    def model(self):
        return self._model

    def get_loss_and_accuracy(self):
        return self._loss_vec_train, self._loss_vec_dev, self._accuracy_vec_train, self._accuracy_vec_dev

    # load dataset
    def _load_data(self, train_dataset, dev_dataset):
        self._dev_loader = None
        self._train_loader = None
        # set train loader
        if train_dataset is not None:
            self._train_loader = DataLoader(
                train_dataset,
                batch_size=self._batch_size,
                collate_fn=train_dataset.collate_fn,
                shuffle=True
            )
            self._train_validation_loader = DataLoader(
                Subset(train_dataset,
                       list(set(random.sample(range(1, len(train_dataset)), int(0.05 * len(train_dataset)))))),
                batch_size=self._batch_size,
                collate_fn=train_dataset.collate_fn,
                shuffle=True
            )
        # set validation loader
        if dev_dataset is not None:
            self._dev_loader = DataLoader(
                dev_dataset,
                batch_size=self._batch_size,
                collate_fn=train_dataset.collate_fn,
                shuffle=True
            )

    def _init_loss_and_acc_vec(self):
        self._loss_vec_dev = []
        self._loss_vec_train = []
        self._accuracy_vec_dev = []
        self._accuracy_vec_train = []

    def _validate_train_and_dev(self, epoch_num):
        with torch.no_grad():
            # validate Train
            loss, accuracy = self._validate(self._train_validation_loader, job="Train")
            self._loss_vec_train.append((epoch_num, loss))
            self._accuracy_vec_train.append((epoch_num, accuracy))
            # validate Dev
            if self._dev_loader is not None:
                loss, accuracy = self._validate(self._dev_loader, job="Dev")
                self._loss_vec_dev.append((epoch_num, loss))
                self._accuracy_vec_dev.append((epoch_num, accuracy))

    def _to_variables(self, pw, pc, hw, hc, label):
        pw = Variable(pw).cuda() if self._gpu else Variable(pw)
        pc = Variable(pc).cuda() if self._gpu else Variable(pc)
        hw = Variable(hw).cuda() if self._gpu else Variable(hw)
        hc = Variable(hc).cuda() if self._gpu else Variable(hc)
        label = Variable(label).cuda() if self._gpu else Variable(label)
        return pw, pc, hw, hc, label

    def _to_test_variables(self, pw, pc, hw, hc, label):
        pw = pw.cuda() if self._gpu else pw
        pc = pc.cuda() if self._gpu else pc
        hw = hw.cuda() if self._gpu else hw
        hc = hc.cuda() if self._gpu else hc
        label = label.cuda() if self._gpu else label
        return pw, pc, hw, hc, label

    # train a model, input is the enum of the model type
    def train(self):
        logger = PrintLogger("NN_train")
        if self._train_loader is None:
            logger.info("load train file to train model")
            return
        logger.info("start_train")
        self._init_loss_and_acc_vec()

        for epoch_num in range(self._epochs):
            logger.info("epoch:" + str(epoch_num))
            # set model to train mode
            self._model.train()
            # calc number of iteration in current epoch
            len_data = len(self._train_loader)
            for batch_index, (p, h, pw, hw, pc, hc, label) in enumerate(self._train_loader):
                stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len_data) + "%")
                stdout.flush()

                pw, pc, hw, hc, label = self._to_variables(pw, pc, hw, hc, label)
                self._model.zero_grad()                         # zero gradients
                output = self._model(pw, pc, hw, hc)            # calc output of current model on the current batch
                loss = self._loss_func(output, label)           # calculate loss
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights

                if self._validation_rate and batch_index % self._validation_rate == 0:
                    logger.info("\nvalidating dev...    epoch:" + "\t" + str(epoch_num + 1) + "/" + str(self._epochs))
                    self._validate_train_and_dev(epoch_num + (batch_index / len_data))
                    self._model.train()

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        logger = PrintLogger(job + "_NN_validation")
        loss_count = 0
        good_pred = 0
        all_pred = 0
        self._model.eval()
        len_data = len(data_loader)
        for batch_index, (p, h, pw, hw, pc, hc, label) in enumerate(data_loader):
            stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len_data) + "%")
            stdout.flush()

            pw, pc, hw, hc, label = self._to_variables(pw, pc, hw, hc, label)
            output = self._model(pw, pc, hw, hc)
            # calculate total loss
            loss_count += self._loss_func(output, label)
            # calculate accuracy
            good_pred += sum([1 if i.item() == j.item() else 0 for i, j in zip(torch.argmax(output, dim=1), label)])
            all_pred += label.shape[0]

        loss = float(loss_count / len(data_loader))
        accuracy = good_pred / all_pred
        logger.info("loss=" + str(loss) + "  ------  accuracy=" + str(accuracy))
        return loss, accuracy

    def predict(self, dataset: SNLIDataset):
        self.model.eval()
        test_data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False
        )
        with torch.no_grad():
            loss_count = 0
            good_pred = 0
            all_pred = 0
            results = []
            len_data = len(test_data_loader)
            for batch_index, (p, h, pw, hw, pc, hc, label) in enumerate(test_data_loader):
                stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len_data) + "%")
                stdout.flush()

                pw, pc, hw, hc, label = self._to_test_variables(pw, pc, hw, hc, label)
                output = self._model(pw, pc, hw, hc)
                # calculate accuracy and loss
                loss_count += self._loss_func(output, label)
                good_pred += sum([1 if i.item() == j.item() else 0 for i, j in zip(torch.argmax(output, dim=1), label)])
                all_pred += label.shape[0]

                for premise, hypothesis, pred in zip(p, h, torch.argmax(output, dim=1)):
                    results.append((dataset.label(pred.item()), " ".join(premise), " ".join(hypothesis)))
            loss = float(loss_count / len(test_data_loader))
            accuracy = good_pred / all_pred
            print("loss=" + str(loss) + "  ------  accuracy=" + str(accuracy))
        return results, loss, accuracy


if __name__ == "__main__":
    import os
    from utils.data_loader import SNLIDataset
    from utils.params import TRAIN_SRC, DEV_SRC, TEST_SRC, PRE_TRAINED_SRC, ChrLevelCnnParams, SequenceEncoderParams,\
        TopLayerParams, SNLIFullModelParams

    # datasets
    ds_train = SNLIDataset(os.path.join("..", TRAIN_SRC), os.path.join("..", PRE_TRAINED_SRC))
    ds_dev = SNLIDataset(os.path.join("..", DEV_SRC))
    ds_test = SNLIDataset(os.path.join("..", TEST_SRC))
    ds_dev.load_word_vocabulary(ds_train.word_vocabulary)
    ds_test.load_word_vocabulary(ds_train.word_vocabulary)

    # model
    model_params = SNLIFullModelParams(ChrLevelCnnParams(chr_vocab_dim=ds_train.len_chars_vocab),
                     SequenceEncoderParams(word_vocab_dim=ds_train.len_words_vocab, pre_trained=ds_train.word_embed_mx),
                     SequenceEncoderParams(word_vocab_dim=ds_train.len_words_vocab), TopLayerParams())
    model_ = SNLIModel(model_params)

    # activator
    params_ = SNLIActivatorParams()
    activator = SNLIActivator(model_, params_, ds_train, ds_dev)

    # train + predict
    activator.train()
    activator.predict(ds_test)
