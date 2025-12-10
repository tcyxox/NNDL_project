import torch

from core.config import config
from core.train import run_training
from core.utils import set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(config.experiment.seed)


if __name__ == "__main__":
    run_training(
        feature_dir=config.paths.split_features,
        output_dir=config.paths.dev,
        feature_dim=config.model.feature_dim,
        batch_size=config.experiment.batch_size,
        learning_rate=config.experiment.learning_rate,
        epochs=config.experiment.epochs,
        enable_feature_gating=config.experiment.enable_feature_gating,
        device=device
    )
