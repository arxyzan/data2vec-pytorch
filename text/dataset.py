import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class WikiText(Dataset):
    """
    A Dataset instance for WikiText dataset loaded from HuggingFace datasets.
    """
    def __init__(self, path, split, tokenizer, mlm_probability=0.15):
        super(WikiText, self).__init__()
        self.data = load_dataset('wikitext', path)[split]
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Only return tokens from raw text with no additions e.g, padding, bos/eos, etc.
        Args:
            index: sample index to pick from dataset

        Returns:
            tokenized outputs
        """
        raw_text = self.data[index]['text']
        tokens = self.tokenizer(raw_text)
        return tokens

    def _mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Ported
         from `transformers.data.DataCollatorForLanguageModeling.torch_mask_tokens()`
        Args:
            inputs: batch of input tokens
            special_tokens_mask:

        Returns:
            a dict batch of masked and padded inputs/labels

        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = self.tokenizer.pad_token_id
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def collate_fn(self, batch):
        """
        Collate the batch of data using BERT masking strategy. carefully ported from
         transformers.data.DataCollatorForLanguageModeling
        Args:
            batch: batch of data

        Returns:
            same batch of data masked and padded
        """

        batch = self.tokenizer.pad(batch, return_tensors="pt")

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self._mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch


if __name__ == '__main__':
    from transformers.models.roberta import RobertaTokenizer
    from torch.utils.data import DataLoader

    dataset = WikiText('wikitext-103-v1', 'train', RobertaTokenizer.from_pretrained('roberta-base'))
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))
    print(batch)
