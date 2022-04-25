import omegaconf

from text.trainer import TextTrainer
from vision.trainer import VisionTrainer
from audio.trainer import AudioTrainer

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to yaml config file')
    args = parser.parse_args()

    cfg_path = args.config
    cfg = omegaconf.OmegaConf.load(cfg_path)
    modality = cfg.modality

    trainers_dict = {
        'text': TextTrainer,
        'vision': VisionTrainer,
        'audio': AudioTrainer
    }
    assert modality in trainers_dict.keys(), f'invalid modality `{cfg.modality}`, expected {list(trainers_dict.keys())}'
    trainer = trainers_dict[modality](cfg)
    trainer.train()
