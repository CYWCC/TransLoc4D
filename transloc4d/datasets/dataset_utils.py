# Code adapted or modified from MinkLoc3DV2 repo: https://github.com/jac99/MinkLoc3Dv2

# Warsaw University of Technology

from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree
import numpy as np
from torch.utils.data import SubsetRandomSampler

from .base_datasets import EvaluationTuple, TrainingDataset
from .augmentation import TrainSetTransform, ValSetTransform
from .pointnetvlad.pnv_train import PNVTrainingDataset
from .pointnetvlad.pnv_train import TrainTransform as PNVTrainTransform
from .samplers import BatchSampler
from misc import TrainingParams


def make_datasets(params: TrainingParams, validation: bool = True):
    # Create training and validation datasets
    datasets = {}
    train_set_transform = TrainSetTransform(params.set_aug_mode)  # data_prepocess augumentation  rotation flip
    val_set_transform = ValSetTransform(params.set_aug_mode)  
    train_transform = PNVTrainTransform(params.aug_mode) 

    datasets['train'] = PNVTrainingDataset(params.dataset_folder, params.train_file,
                                           transform=train_transform, set_transform=train_set_transform,)
    if validation:
        datasets['val'] = PNVTrainingDataset(params.dataset_folder, params.val_file, set_transform=val_set_transform)

    return datasets


def make_collate_fn(dataset: TrainingDataset, quantizer, batch_split_size=None, input_representation="R",):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    # batch_split_size: if not None, splits the batch into a list of multiple mini-batches with batch_split_size elems
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]

        if dataset.set_transform is not None:
            clouds = dataset.set_transform(torch.stack(clouds))

            # lens = [len(cloud) for cloud in clouds]
            # clouds = torch.cat(clouds, dim=0)
            # xyz = clouds[:, :3]
            # xyz = dataset.set_transform(xyz)
            # clouds[:, :3] = xyz
            # clouds = clouds.split(lens)

        if isinstance(clouds, list):  # val: list to tensor
            clouds = torch.stack(clouds)

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)


        # Convert to polar (when polar coords are used) and quantize
        # Use the first value returned by quantizer
        coords_quant = [quantizer(e)[0] for e in clouds]
        coords = [e[:, :3] for e in coords_quant]
        if input_representation == "RV":
            feats = [e[:, 3:4] for e in coords_quant]
        elif input_representation == "RI":
            feats = [e[:, 4:] for e in coords_quant]
        else:
            feats = [e[:, 3:] for e in coords_quant]
            
        if batch_split_size is None or batch_split_size == 0:
            coords = ME.utils.batched_coordinates(coords)
            if input_representation == "R":
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            else:
                feats = torch.cat(feats, 0)
            # batch = {'coords': coords, 'features': feats, 'batch': clouds[:,:,:3]}
            batch = {'coords': coords, 'features': feats}

        else:
            # Split the batch into chunks
            batch = []

            for i in range(0, len(coords), batch_split_size):
                temp = coords[i:i + batch_split_size]
                c = ME.utils.batched_coordinates(temp)
                if input_representation == "R":
                    f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                else:
                    f = torch.cat(feats[i : i + batch_split_size], 0)
                batch_temp = clouds[i:i + batch_split_size]
                # batch_temp =  torch.cat(batch_temp, 0)[:,:3]
                # minibatch = {'coords': c, 'features': f, 'batch': batch_temp[:,:,:3]}
                # minibatch = {'coords': c, 'features': f, 'batch': batch_temp}
                minibatch = {'coords': c, 'features': f}
                batch.append(minibatch)

        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(params: TrainingParams, validation=True, cluster_nIm=-1):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, validation=validation)  # choose datasets

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)  # generate batches

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn(datasets['train'],  quantizer, params.batch_split_size, params.model_params.input_representation,)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
                                     # pin_memory = True, worker_init_fn = worker_init_fn)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer, params.batch_split_size, params.model_params.input_representation,)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)
    
    if cluster_nIm > 0:
        sampler = SubsetRandomSampler(
            np.random.choice(len(datasets["train"]), cluster_nIm, replace=False)
        )
        dataloders["cluster"] = DataLoader(
            dataset=datasets["train"],
            collate_fn=train_collate_fn,
            num_workers=params.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] "
          f"radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

