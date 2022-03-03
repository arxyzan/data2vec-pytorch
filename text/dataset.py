import torch
from torch.utils.data import Dataset
from transformers.data import DataCollatorForLanguageModeling
from datasets import load_dataset


class WikiText(Dataset):
    def __init__(self, path, split, tokenizer):
        super(WikiText, self).__init__()
        self.data = load_dataset('wikitext', path)[split]
        self.tokenizer = tokenizer
        self._collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_text = self.data[index]['text']
        tokens = self.tokenizer(raw_text)
        return tokens

    def collate_fn(self, batch):
        """
        Collate the batch of data using BERT masking strategy
        Args:
            batch: batch of data

        Returns:
            same batch of data masked and padded
        """
        batch = self._collate_fn(batch)
        batch['labels'][batch['labels'] == -100] = self.tokenizer.pad_token_id
        return batch


if __name__ == '__main__':
    from transformers.models.roberta import RobertaTokenizer
    from torch.utils.data import DataLoader

    dataset = WikiText('wikitext-103-v1', 'train', RobertaTokenizer.from_pretrained('roberta-base'))
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))
    print(batch)
