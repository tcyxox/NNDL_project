"""
New Data Split Module

Provides functionality to split features into Train/Test and Train/Val (SubTrain/Val)
supporting Superclass Novelty and Subclass Novelty injection.
"""
import os
from dataclasses import dataclass
from typing import Set, Tuple

import numpy as np
import torch


@dataclass
class Dataset:
    features: torch.Tensor
    super_labels: torch.Tensor
    sub_labels: torch.Tensor

    def __len__(self):
        return len(self.features)
    
    def to(self, device):
        return Dataset(
            self.features.to(device),
            self.super_labels.to(device),
            self.sub_labels.to(device)
        )


@dataclass
class SplitOutput:
    train_set: Dataset  # Or subtrain
    test_set: Dataset   # Or val
    known_subclasses: Set[int]
    novel_subclasses: Set[int]
    novel_super_classes: Set[int]


def load_full_dataset(feature_dir: str) -> Dataset:
    """Load the full training dataset from disk."""
    features = torch.load(os.path.join(feature_dir, "train_features.pt"))
    super_labels = torch.load(os.path.join(feature_dir, "train_super_labels.pt"))
    sub_labels = torch.load(os.path.join(feature_dir, "train_sub_labels.pt"))
    return Dataset(features, super_labels, sub_labels)


def get_sub_to_super_map(sub_labels: torch.Tensor, super_labels: torch.Tensor) -> dict:
    """Build a mapping from Subclass ID to Superclass ID."""
    unique_subs = torch.unique(sub_labels)
    sub_to_super = {}
    for sub in unique_subs:
        # Find first occurrence to get the super label
        idx = (sub_labels == sub).nonzero(as_tuple=True)[0][0]
        sub_to_super[sub.item()] = super_labels[idx].item()
    return sub_to_super


def filter_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    """Create a new Dataset containing only the specified indices."""
    # Convert numpy array to torch tensor for indexing
    idx_tensor = torch.from_numpy(indices)
    return Dataset(
        dataset.features[idx_tensor],
        dataset.super_labels[idx_tensor],
        dataset.sub_labels[idx_tensor]
    )


def split_full_train_to_train_test(
    full_dataset: Dataset,
    test_ratio: float = 0.2,
    test_sub_novel_ratio: float = 0.1,
    novel_sub_index: int = None,
    seed: int = 42,
    verbose: bool = True
) -> SplitOutput:
    """
    Splits the full dataset into Train (Pure Known) and Test (Known + Novel).
    
    Args:
        full_dataset: The complete dataset.
        test_ratio: Target size ratio for Test set (e.g., 0.2).
        test_sub_novel_ratio: Ratio of subclasses to treat as Novel (only in Test).
        seed: Random seed.
        
    Returns:
        SplitOutput containing train_set, test_set, and class info.
    """
    np.random.seed(seed)
    
    sub_to_super = get_sub_to_super_map(full_dataset.sub_labels, full_dataset.super_labels)
    all_subclasses = np.array(list(sub_to_super.keys()))
    
    # 1. Select Novel Subclasses for Test
    n_total = len(all_subclasses)
    n_novel = int(n_total * test_sub_novel_ratio)
    
    novel_subclasses = set(np.random.choice(all_subclasses, n_novel, replace=False))
    known_subclasses = set(all_subclasses) - novel_subclasses
    
    # 2. Identify indices
    sub_labels_np = full_dataset.sub_labels.numpy()
    
    # Indices for Novel samples (MUST go to Test)
    novel_indices = [i for i, x in enumerate(sub_labels_np) if x in novel_subclasses]
    
    # Indices for Known samples (Can go to Train or Test)
    known_indices = [i for i, x in enumerate(sub_labels_np) if x in known_subclasses]
    
    # 3. Split Known samples to fill remaining Test capacity
    # Target Test Size = Total * test_ratio
    target_test_size = int(len(full_dataset) * test_ratio)
    needed_known = target_test_size - len(novel_indices)
    
    # Ensure we don't need negative amount
    if needed_known < 0:
        needed_known = 0
    
    # Randomly select known samples for Test
    np.random.shuffle(known_indices)
    test_known_indices = known_indices[:needed_known]
    train_known_indices = known_indices[needed_known:]
    
    # Combine indices
    test_indices = np.array(novel_indices + test_known_indices)
    train_indices = np.array(train_known_indices)
    
    # Shuffle final sets
    np.random.shuffle(test_indices)
    np.random.shuffle(train_indices)
    
    test_set = filter_dataset(full_dataset, test_indices)

    # Apply masking if index provided
    if novel_sub_index is not None:
        idx_np = test_set.sub_labels.numpy()
        mask = np.isin(idx_np, list(novel_subclasses))
        if mask.any():
            test_set.sub_labels[torch.from_numpy(mask)] = novel_sub_index

    if verbose:
        print(f"[Split: Full -> Train/Test] Seed: {seed}")
        print(f"  Train (Pure Known): {len(train_indices)} samples")
        print(f"  Test  (Mix):        {len(test_indices)} samples")
        print(f"  Novel Subclasses:   {len(novel_subclasses)} (Ratio: {test_sub_novel_ratio:.1%})")
        print(f"  Known Subclasses:   {len(known_subclasses)}")
    
    return SplitOutput(
        train_set=filter_dataset(full_dataset, train_indices),
        test_set=test_set,  # Use already-modified test_set
        known_subclasses=known_subclasses,
        novel_subclasses=novel_subclasses,
        novel_super_classes=set() # No novel super classes introduced in this stage
    )


def split_train_to_subtrain_val(
    train_dataset: Dataset,
    val_ratio: float = 0.2,
    val_sub_novel_ratio: float = 0.1,
    val_include_novel: bool = True,
    force_super_novel: bool = False,
    target_super_novel: int = None,  # Specify which superclass to use as novel
    novel_sub_index: int = None,
    novel_super_index: int = None,
    seed: int = 42,
    verbose: bool = True
) -> SplitOutput:
    """
    Splits Train (Known) into SubTrain and Val.
    Val may contain "Novel" classes relative to SubTrain.
    
    Args:
        train_dataset: Input dataset (Result of previous split, pure known relative to outer loop).
        val_ratio: Target size ratio for Val set.
        val_sub_novel_ratio: Ratio of subclasses to treat as Novel in Val.
        val_include_novel: Whether Val should include novel classes.
        force_super_novel: Whether to force one Superclass to be Novel in Val.
        target_super_novel: Specific superclass index to use as novel (if None, random choice).
        seed: Random seed.
        
    Returns:
        SplitOutput containing train_set (SubTrain), test_set (Val), and class info.
    """
    np.random.seed(seed)
    
    sub_to_super = get_sub_to_super_map(train_dataset.sub_labels, train_dataset.super_labels)
    all_subclasses = np.array(list(sub_to_super.keys())) # All currently known
    
    novel_super_subclasses = set()
    novel_ordin_subclasses = set()
    novel_super_classes = set()
    
    if val_include_novel:
        # 1. Super Novel Selection
        if force_super_novel:
            all_supers = sorted(set(sub_to_super.values()))  # Sorted for determinism
            if all_supers: # Safety check
                if target_super_novel is not None and target_super_novel in all_supers:
                    target_super = target_super_novel
                else:
                    target_super = np.random.choice(all_supers)
                novel_super_classes.add(target_super)
                
                # Find all subclasses for this super
                for sub, sup in sub_to_super.items():
                    if sup == target_super:
                        novel_super_subclasses.add(sub)
        
        # 2. Subclass Novel Selection (Ordinary)
        # Goal: reach val_sub_novel_ratio
        current_n_novel = len(novel_super_subclasses)
        target_n_novel = int(len(all_subclasses) * val_sub_novel_ratio)
        
        needed = target_n_novel - current_n_novel
        remaining_candidates = [s for s in all_subclasses if s not in novel_super_subclasses]
        
        if needed > 0 and len(remaining_candidates) >= needed:
            picked = np.random.choice(remaining_candidates, needed, replace=False)
            novel_ordin_subclasses.update(picked)
        elif needed > 0:
            # Not enough candidates (unlikely if ratio small), take all
            novel_ordin_subclasses.update(remaining_candidates)
            
    # Combine Novels
    all_novel_subclasses = novel_super_subclasses.union(novel_ordin_subclasses)
    known_subclasses = set(all_subclasses) - all_novel_subclasses
    
    # Indices
    sub_labels_np = train_dataset.sub_labels.numpy()
    
    novel_indices = [i for i, x in enumerate(sub_labels_np) if x in all_novel_subclasses]
    known_indices = [i for i, x in enumerate(sub_labels_np) if x in known_subclasses]
    
    # Split Knowns to fill Val
    target_val_size = int(len(train_dataset) * val_ratio)
    n_novel = len(novel_indices)
    
    if val_include_novel and n_novel > 0:
        # We have novel samples
        val_novel_indices = novel_indices  # All novel to Val (keep SubTrain pure)
        
        # Calculate needed known: max of (fill target) or (50% of novel)
        # Rule: known must be >= 50% of unknown for proper threshold calibration
        min_known_for_balance = n_novel // 2
        needed_for_target = max(0, target_val_size - n_novel)
        needed_known = max(needed_for_target, min_known_for_balance)
    else:
        # No novel in val
        val_novel_indices = []
        needed_known = target_val_size
            
    np.random.shuffle(known_indices)
    val_known_indices = known_indices[:needed_known]
    subtrain_known_indices = known_indices[needed_known:]
    
    # Val gets Novel + Some Known
    val_indices = np.array(val_novel_indices + val_known_indices)
    subtrain_indices = np.array(subtrain_known_indices)
    
    np.random.shuffle(val_indices)
    np.random.shuffle(subtrain_indices)

    test_set = filter_dataset(train_dataset, val_indices) # Val set

    # Apply masking
    if novel_sub_index is not None and len(all_novel_subclasses) > 0:
        idx_np = test_set.sub_labels.numpy()
        mask = np.isin(idx_np, list(all_novel_subclasses))
        if mask.any():
            test_set.sub_labels[torch.from_numpy(mask)] = novel_sub_index
            
    if novel_super_index is not None and len(novel_super_classes) > 0:
        idx_np = test_set.super_labels.numpy()
        mask = np.isin(idx_np, list(novel_super_classes))
        if mask.any():
            test_set.super_labels[torch.from_numpy(mask)] = novel_super_index
    
    if verbose:
        print(f"[Split: Train -> SubTrain/Val] Seed: {seed}")
        print(f"  SubTrain (Pure Known): {len(subtrain_indices)} samples")
        print(f"  Val      ({'Mix' if val_include_novel else 'Pure Known'}):          {len(val_indices)} samples")
        super_novel_info = f"{len(novel_super_classes)} classes" if force_super_novel else "None"
        print(f"  Super Novel:           {super_novel_info}")
        print(f"  Sub Novel (Total):     {len(all_novel_subclasses)} classes")
        print(f"  Known Subclasses:      {len(known_subclasses)}")

    return SplitOutput(
        train_set=filter_dataset(train_dataset, subtrain_indices), # SubTrain
        test_set=test_set,  # Use already-modified val_set
        known_subclasses=known_subclasses,
        novel_subclasses=all_novel_subclasses,
        novel_super_classes=novel_super_classes
    )
