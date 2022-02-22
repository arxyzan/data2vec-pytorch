from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch.nn as nn


class Roberta(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(Roberta, self).__init__()
        self.cfg = cfg
        self.encoder = RobertaModel(RobertaConfig(vocab_size=vocab_size))

    def apply_mask(self): ...

    def forward(self, src, **kwargs):
        outputs = self.encoder(input_ids=src, output_hidden_states=True, output_attentions=True, **kwargs)
        return outputs

    def extract_features(self, src, **kwargs):
        outputs = self(src, **kwargs)
        outputs['hidden_states'] = outputs['hidden_states'][:-1]  # Ignore last item. we just need encoder outputs
        return list(outputs)


if __name__ == '__main__':
    model = Roberta(None, 50265)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")
    outputs = model(inputs['input_ids'])
    print(outputs)
