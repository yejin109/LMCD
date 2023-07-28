import os
import torch


class CustomMLMCollator:
    def __init__(self, tk, mask_prob):
        self.tokenizer = tk
        self.mask_prob = mask_prob

    def __call__(self, inputs):
        first = inputs[0]['input_ids']
        agg = torch.cat(([i['input_ids'] for i in inputs])).reshape((-1, first.size(-1)))
        batch = dict()
        batch['input_ids'], batch['labels'] = masking_tokens(agg, self.mask_prob)

        batch['attention_mask'] = torch.cat(([i['attention_mask'] for i in inputs])).reshape((-1, first.size(-1)))

        return batch

    def masking_tokens(self, inputs, mask_prob):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mask_prob)

        if self.tokenizer is not None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token) if self.tokenizer is not None else 103
        #
        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def custom_mlm_collator(inputs):
    mask_prob = float(os.environ['MASKING_P'])
    first = inputs[0]['input_ids']
    agg = torch.cat(([i['input_ids'] for i in inputs])).reshape((-1, first.size(-1)))
    batch = dict()
    batch['input_ids'], batch['labels'] = masking_tokens(agg, mask_prob)

    batch['attention_mask'] = torch.cat(([i['attention_mask'] for i in inputs])).reshape((-1, first.size(-1)))
    # iter_step = int(os.environ['ITERATION_STEP'])
    # if (iter_step+1) % 500 == 0:
    #     torch.save(batch['input_ids'], f"{os.environ['LOG_DIR']}/batch/input_ids_{iter_step+1}.pt")
    #     torch.save(batch['labels'], f"{os.environ['LOG_DIR']}/batch/labels_{iter_step+1}.pt")
    # os.environ['ITERATION_STEP'] = str(iter_step+1)
    return batch


def masking_tokens(inputs, mask_prob):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mask_prob)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    inputs[indices_replaced] = 103
    #
    # # 10% of the time, we replace masked input tokens with random word
    # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    # inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
