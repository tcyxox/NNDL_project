"""
命令行脚本：使用 core.data_split 模块划分特征数据并保存
"""
from core.config import config
from core.data_split import split_features
from core.utils import set_seed

if __name__ == "__main__":
    set_seed(config.experiment.seed)
    
    split_features(
        feature_dir=config.paths.features,
        novel_ratio=config.split.novel_ratio,
        train_ratio=config.split.train_ratio,
        val_test_ratio=config.split.val_test_ratio,
        novel_sub_index=config.osr.novel_sub_index,
        output_dir=config.paths.split_features,
        verbose=True
    )
