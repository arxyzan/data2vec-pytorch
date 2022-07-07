import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm


class WikiText(Dataset):
    """
    A Dataset instance for WikiText dataset loaded from HuggingFace datasets.

    Args:
        cfg (DictConfig): config object
        split: Split to load ['train', 'test']
        tokenizer: A HuggingFace Tokenizer model like BPE
        **kwargs: extra args which are set as dataset properties
    """

    def __init__(self, cfg, split, tokenizer, **kwargs):
        super(WikiText, self).__init__()
        self.cfg = cfg
        self.path = cfg.dataset.name
        self.mlm_probability = cfg.dataset.mlm_probability
        raw_data = load_dataset('wikitext', self.path)[split]
        self.data = self.clean_dataset(raw_data) if self.cfg.dataset.clean_dataset else raw_data
        self.tokenizer = tokenizer
        self.__dict__.update(kwargs)

    def clean_dataset(self, data):
        """
        Cleanup dataset by removing invalid sized samples, etc.
        """
        print('Cleaning dataset ...')
        min_seq_len, max_seq_len = self.cfg.data.valid_seq_lenghts
        texts = []
        with tqdm(data, desc='Removing invalid sized inputs: ') as tbar:
            for i, x in enumerate(tbar):
                if len(x['text']) in range(min_seq_len, max_seq_len + 1):
                    texts.append(x)
        return texts

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
        tokens = self.tokenizer(raw_text, return_attention_mask=False)
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
        return inputs, labels, masked_indices

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
        src, trg, masked_indices = self._mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return src, trg, masked_indices


if __name__ == '__main__':
    from transformers.models.roberta import RobertaTokenizer
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    cfg = OmegaConf.load('configs/roberta-pretraining.yaml')
    dataset = WikiText(cfg, 'train', RobertaTokenizer.from_pretrained('roberta-base'))
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    data_iter = iter(dataloader)
    batch = next(data_iter)
    print(batch)
