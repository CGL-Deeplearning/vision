import torch
from torch.utils.data import sampler
import pandas as pd


class BalancedSampler(sampler.Sampler):
    """
    A sampler that balances the data set.
    Arguments:
        A data set containing images
    """

    def __init__(self, csv_path):
        pandas_df = pd.read_csv(csv_path, header=None)[2]
        labels = pandas_df.str.split('\t', expand=True)[8]

        self.indices = list(range(len(labels)))
        self.num_samples = len(self.indices)

        label_frequency = labels.value_counts().to_dict()

        data = {'label': labels, 'weight': 0.0}
        weights = pd.DataFrame(data=data)

        weights.loc[weights.label == '0', 'weight'] = 1.0 / label_frequency['0']
        weights.loc[weights.label == '1', 'weight'] = 1.0 / label_frequency['1']
        weights.loc[weights.label == '2', 'weight'] = 1.0 / label_frequency['2']
        print("Sampler weights:",
              "\n0: ", 1.0 / label_frequency['0'],
              "\n1: ", 1.0 / label_frequency['1'],
              "\n2: ", 1.0 / label_frequency['2'])

        weights = weights['weight'].tolist()
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
