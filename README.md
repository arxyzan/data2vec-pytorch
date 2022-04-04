# data2vec-pytorch
##### PyTorch implementation of "[data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555)" from Meta AI (FAIR)
Data2Vec is the first high-performance self-supervised algorithm that learns the same way in multiple modalities, including speech, vision and text. 
Most machines learn exclusively from labeled data. However, through self-supervised learning, machines are able to learn about the world just by observing it 
and then figuring out the structure of images, speech or text. This is a more scalable and efficient approach for machines to tackle new complex tasks,
such as understanding text for more spoken languages. 

![](data2vec.png)

In summary, the method is as follows: <br>
1. The encoder extracts features from the masked inputs. These features are outputs of every transformer/linear layer.
2. The teacher which is an EMA instance of the encoder (in eval model), extracts features from the unmasked inputs.
3. Optional normalizations are applied to the layers/outputs of the teacher.
4. Encoder outputs are regressed by a projection block/layer.
5. The loss is calculated from encoder outputs and teacher outputs.

You can read the paper for more detail.

## Implementation
Data2Vec is already implemented in [fairseq](https://github.com/pytorch/fairseq/tree/main/examples/data2vec) in which for all modalities there is a seperate implementation (text, vision, audio). According to the paper:
> <cite>Our primary is to design a single learning mechanism for different modalities. 
Despite the unified learning regime, we still use modality-specific features extractors and masking strategies. 
This makes sense given the vastly different nature of the input data.</cite>

This implementation differs in the fact that a single Data2Vec model is provided powered by a custom encoder (implemented using PyTorch + HuggingFace Transformers) and tries to unify the whole concept in a single module. 
The key concept is that there must be modality-specific feature extractions and masking strategies.

- **Masking:** For each modality, the Dataset instance must return the masked source, the target and the mask tensor.

- **Feature Extraction:** Features are the outputs from the transformer/attention layers. So the forward method must return outputs from all Encoder blocks of the transformer model. HuggingFace Transformers/Fairseq models return transformer layers outputs separately out of the box.

This implementation uses HuggingFace Transformers models as encoders for Data2Vec which you can inspect in the `encoder.py` files for each modality. Although, you can provide your own encoder model. Just make sure that your encoder must be Transformer-based according to the paper and outputs from every encoder layer must be provided.

**Note**: This implementation's goal is to provide the necessary building blocks of Data2Vec so anyone can adapt it to their own use case with ease, so in order to make it easy to get hands on, some functionalities like mixed precision, distributed training, etc are not included to keep it as clean & simple as possible. If you only need to train a standard large scale Data2Vec model use the [official repo](https://github.com/pytorch/fairseq/tree/main/examples/data2vec).

## Train
First things first, install the requirements:
```bash
pip install -r requirements.txt
```

#### **NLP**
Train a Language Model based on RoBERTa (HuggingFace) on WikiText103

Configure the related properties in `text/configs/roberta-pretraining.yaml` and run:
```bash
python train.py --config text/configs/roberta-pretraining.yaml 
```

#### **Vision**
Run a Masked Image modeling training based on BEiT (HuggingFace)

Pass the path to the image dataset in the config file at `vision/configs/beit-pretraining.yaml` under dataset > path > train/test and modify other properties as you desire and run the following:
```bash
python train.py --config vision/configs/beit-pretraining.yaml 
```

#### **Speech**
In Progress ...


## Fine-tuning
A data2vec model consists of an encoder and regression layers on top. To fine-tune any model pretrained using Data2Vec, you can just take the main encoder from the saved checkpoint and fine-tune it as you would fine-tune a regular model.
```python
# load a checkpoint for finetuning
from transformers import RobertaModel, RobertaConfig
roberta = RobertaModel(RobertaConfig)
checkpoint = torch.load('path/to/data2vec.pt')
roberta_state_dict = checkpoint['encoder']
# load roberta weights from the encoder part of the data2vec model
encoder = roberta.load_state_dict(roberta_state_dict)

# Now fine-tune a regular HuggingFace RoBERTa model as usual
...
```


## Contributions
Any contribution regarding training, development and issues are welcome!
