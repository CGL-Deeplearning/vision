import torch
from torch.utils.data import sampler


class BalancedSampler(sampler.Sampler):
    """
    A sampler that balances the data set.
    Arguments:
        A data set containing images
    """

    def __init__(self, dataset):
        print("Sampler initialization starting")
        self.indices = list(range(len(dataset)))

        self.num_samples = len(self.indices)

        label_frequency = {}
        for idx in self.indices:
            label = dataset[idx][1]

            if label in label_frequency:
                label_frequency[label] += 1
            else:
                label_frequency[label] = 1

        weights = [1.0 / label_frequency[dataset[idx][1]] for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)
        print("Sampler initialization finished")

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
