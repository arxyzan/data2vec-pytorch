import omegaconf

from text.trainer import Trainer as TextTrainer
from vision.trainer import Trainer as VisionTrainer
from audio.trainer import Trainer as AudioTrainer

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
    trainer = trainers_dict[modality](cfg)
    trainer.train()
