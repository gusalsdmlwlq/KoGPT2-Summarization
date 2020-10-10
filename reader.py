import os
import random
from copy import deepcopy
import logging
import time
import json

import numpy as np
import torch
from sentencepiece import SentencePieceProcessor as sp
from kogpt2.utils import get_tokenizer

from config import Config


class Reader:
    def __init__(self, config):
        self.tokenizer = sp(get_tokenizer())
        self.train_data = []
        self.dev_data = []
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx
        self.pad_idx = config.pad_idx

    def load_data(self):
        self.train_data = json.load(open(os.path.join(self.data_path, "train_data.json"), "r"))        
        self.dev_data = json.load(open(os.path.join(self.data_path, "dev_data.json"), "r"))

    def make_batch(self, mode="train"):
        if mode == "train":
            data = self.train_data
        else:
            data = self.dev_data
        all_batches = []
        batch = []
        for doc_id, doc in data.items():
            batch.append(doc)
            if len(batch) == self.batch_size:
                all_batches.append(batch)
                batch = []
        if len(batch) > 0:
            all_batches.append(batch)
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch
                
    def make_input(self, batch, train=True):
        batch_size = len(batch)
        inputs = torch.ones(batch_size, self.max_length, dtype=torch.int64).cuda() * self.pad_idx
        labels = torch.ones(batch_size, self.max_length, dtype=torch.int64).cuda() * self.pad_idx
        doc_lengths = []
        max_length = 0
        max_label_length = 0
        for batch_idx in range(batch_size):
            document = self.tokenizer.EncodeAsIds(batch[batch_idx]["document"] + " ; Summary: ")
            summary = self.tokenizer.EncodeAsIds(batch[batch_idx]["summary"])
            doc_lengths.append(len(document))
            if train:
                document = document[-(self.max_length - len(summary) - 1):]
                context = document + summary
            else:
                document = document[-self.max_length:]
                context = document
            length = len(context)
            inputs[batch_idx, :length] = torch.tensor(context, dtype=torch.int64)
            if train:
                labels[batch_idx, :length] = torch.tensor(context[1:] + [self.eos_idx], dtype=torch.int64)
            else:
                label_length = len(summary) + 1
                labels[batch_idx, :label_length] = torch.tensor(summary + [self.eos_idx], dtype=torch.int64)
                max_label_length = max(max_label_length, len(summary)+1)
            max_length = max(max_length, length)
        inputs = inputs[:, :max_length]
        labels = labels[:, :max_length] if train else labels[:, :max_label_length]

        return inputs, labels, doc_lengths


if __name__ == "__main__":
    config = Config()
    parser = config.parser
    config = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    reader = Reader(config)
    logger.info("Load data...")
    start = time.time()
    reader.load_data()
    end = time.time()
    logger.info("{} secs".format(end-start))

    logger.info("Make batch...")
    start = time.time()
    iterator = reader.make_batch("dev")
    end = time.time()
    logger.info("{} secs".format(end-start))

    for batch in iterator:
        inputs, labels, doc_lengths = reader.make_input(batch)

