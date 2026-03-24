from typing import Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch_geometric.loader import DataLoader

from src.smart.datasets import MultiDataset

from .target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal


class MultiDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        train_raw_dir: str,
        val_raw_dir: str,
        test_raw_dir: str,
        val_gt_scenario_dir: Optional[str],
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        train_max_num: int,
        random_scene_scale_config: Optional[dict] = None,
        random_time_shift_config: Optional[dict] = None,
        random_time_mask_config: Optional[dict] = None,
        random_scene_crop_config: Optional[dict] = None,
    ) -> None:
        super(MultiDataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_gt_scenario_dir = val_gt_scenario_dir
        self.random_scene_scale_config = random_scene_scale_config
        self.random_time_shift_config = random_time_shift_config
        self.random_time_mask_config = random_time_mask_config
        self.random_scene_crop_config = random_scene_crop_config

        self.train_transform = WaymoTargetBuilderTrain(train_max_num)
        self.val_transform = WaymoTargetBuilderVal()
        self.test_transform = WaymoTargetBuilderVal()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = MultiDataset(self.train_raw_dir, 
                                              self.train_transform,
                                              random_scene_scale_config=self.random_scene_scale_config,
                                              random_time_shift_config=self.random_time_shift_config,
                                              random_time_mask_config=self.random_time_mask_config,
                                              random_scene_crop_config=self.random_scene_crop_config)
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                gt_scenario_dir=self.val_gt_scenario_dir,
            )
        elif stage == "validate":
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                gt_scenario_dir=self.val_gt_scenario_dir,
            )
        elif stage == "test":
            self.test_dataset = MultiDataset(self.test_raw_dir, self.test_transform)
        else:
            raise ValueError(f"{stage} should be one of [fit, validate, test]")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # 0
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
