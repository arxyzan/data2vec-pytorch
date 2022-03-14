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

    def forward(self, src, **kwargs):
        """
        Forward inputs through the encoder and extract transformer/attention layers outputs

        Args:
            src: masked source tokens

        Returns:
            A dictionary of encoder outputs including encoder outputs and attentions outputs

        """
        outputs = self.encoder(src, output_hidden_states=True, output_attentions=True, **kwargs)
        encoder_states = outputs['hidden_states'][:-1]  # encoder layers outputs separately
        encoder_out = outputs['hidden_states'][-1]      # last encoder output (accumulated)
        attentions = outputs['attentions']
        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }


if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = Roberta(None, vocab_size=tokenizer.vocab_size)
    inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")
    outputs = model(inputs['input_ids'])
    print(outputs)
