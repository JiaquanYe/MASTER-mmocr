import copy

from mmdet.datasets import DATASETS, ConcatDataset, build_dataset


@DATASETS.register_module()
class WeightedConcatDataset(ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    weights sample.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, weights, len_epoch=None, separate_eval=True, **kwargs):
        datasets = [build_dataset(c, kwargs) for c in datasets]
        super().__init__(datasets, separate_eval)
        self.weights = []
        self.sample_num = []
        self.dataset_sizes = []
        for dataset, weight in zip(datasets, weights):
            if isinstance(datasets, WeightedConcatDataset):
                sample_weights = [w * weight / sum(dataset.weights) for w in dataset.weights]
                sample_num = dataset.sample_num
                self.dataset_sizes.extend(dataset.dataset_sizes)
            else:
                sample_weights = [weight]
                sample_num = [len(dataset)]
                self.dataset_sizes.extend(sample_num)
        self.recurrent_cumulative_sizes = []
        s = 0
        for ds in self.dataset_sizes:
            s += ds
            self.recurrent_cumulative_sizes.append(s)
        self.original_weight = weights
        self.len_epoch = len_epoch

