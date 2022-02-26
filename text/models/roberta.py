from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch.nn as nn


class Roberta(nn.Module):
    """
    Roberta model using HuggingFace.

    Args:
        cfg: An omegaconf.DictConf instance containing all the configurations.
        vocab_size: Total size of the tokens dictionary
    """

    def __init__(self, cfg, **kwargs):
        super(Roberta, self).__init__()
        self.cfg = cfg
        self.encoder = RobertaModel(RobertaConfig(**kwargs))

    def apply_mask(self):
        ...

    def forward(self, src, **kwargs):
        """
        Fetch outputs from the encoder model. This method directly calls the forward method of RobertaModel. In case you
        need to get specific outputs from the model, provide them as keyword args.
        Args:
            src: source tokens
            **kwargs: Input parameters to transformers.RobertaModel

        Returns:
            A dictionary of encoder outputs
        """
        outputs = self.encoder(input_ids=src, **kwargs)
        return outputs

    def extract_features(self, src):
        """
        Extract features from encoder and attention layers.

        Args:
            src: source tokens. masked

        Returns:
            A dictionary of encoder outputs including encoder outputs and attentions outputs

        """
        outputs = self(src, output_hidden_states=True, output_attentions=True)
        encoder_states = outputs['hidden_states'][:-1]  # encoder layers outputs separately
        encoder_out = outputs['hidden_states'][-1]      # last encoder output (accumulated)
        attentions = outputs['attentions']
        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }


if __name__ == '__main__':
    model = Roberta(None, 50265)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")
    features = model.extract_features(inputs['input_ids'])
    outputs = model(inputs['input_ids'], output_hidden_states=True, output_attentions=True)
    print(outputs)
