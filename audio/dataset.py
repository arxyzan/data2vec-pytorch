import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers import Wav2Vec2FeatureExtractor


class TIMIT(Dataset):
    def __init__(self, cfg, split, **kwargs):
        super(TIMIT, self).__init__()
        path = cfg.dataset.path
        self.data = load_dataset(path, 'clean')[split]
        self.feature_extractor = Wav2Vec2FeatureExtractor(cfg.model.encoder_checkpoint)
        self.__dict__.update(kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]['audio']
        x = self.feature_extractor(x['array'], sampling_rate=x['sampling_rate'], padding=True, return_tensors='pt')['input_values']
        return {'input_values': x[0]}


class DataCollatorForWav2Vec2Pretraining:  # copied from transformers/examples/pytorch/speech-pretraining
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices for self-supervised
    pretraining. Args: model (:class:`~transformers.Wav2Vec2ForPreTraining`): The Wav2Vec2 model used for
    pretraining. The data collator needs to have access to config and ``_get_feat_extract_output_lengths`` function
    for correct padding. feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`): The processor used for
    processing the data. padding (:obj:`bool`, :obj:`str` or
    :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`): Select a
    strategy to pad the returned sequences (according to the model's padding side and padding index) among: *
    :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided). * :obj:`'max_length'`: Pad to a maximum length specified with the argument
    :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not provided. *
    :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths). max_length (:obj:`int`, `optional`): Maximum length of the ``input_values`` of the returned list and
    optionally padding length (see above). pad_to_multiple_of (:obj:`int`, `optional`): If set will pad the sequence
    to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA
    hardware with compute capability >= 7.5 (Volta).
    """

    def __init__(self, model, feature_extractor, padding, max_length=None, pad_to_multiple_of=None):
        self.model = model
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )
        mask_time_indices = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        src = batch['input_values']

        return src, mask_time_indices


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from transformers import Wav2Vec2Model, Wav2Vec2Config

    cfg = OmegaConf.load('configs/wav2vec2-pretraining.yaml')
    model = Wav2Vec2Model(Wav2Vec2Config())
    feature_extractor = Wav2Vec2FeatureExtractor()
    dataset = TIMIT(cfg, 'train')
    collate_fn = DataCollatorForWav2Vec2Pretraining(model, feature_extractor, padding='longest')
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    itr = iter(loader)
    sample = next(itr)
    print(sample)
