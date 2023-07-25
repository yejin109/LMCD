"""
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1302
"""
import os
import torch
from torch import nn
from transformers import Trainer
from torch.nn import CrossEntropyLoss


class MLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        labels = labels.view(-1)
        logits = logits.view(-1, model.config.vocab_size)
        logits[:, int(os.environ['VOCAB_SIZE'])+1:] *= 0

        # logits = logits[labels != -100][:, :int(os.environ['VOCAB_SIZE'])+1]
        # labels = labels[labels != -100]
        #
        # ind = torch.argsort(labels)
        # logits = logits[ind]
        # labels = labels[ind]
        #
        # loss_fct = nn.LogSoftmax(dim=-1)
        # one_hot = nn.functional.one_hot(labels)
        # masked_lm_loss = torch.mean(loss_fct(logits) * one_hot)

        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(logits, labels)

        return (masked_lm_loss, outputs) if return_outputs else masked_lm_loss
