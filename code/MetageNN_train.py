import sys
import random
import argparse
import json

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import seed_everything
from torch.autograd import Function

#In case you want train for a specific rank
#phylum = 0
#class = 1
#order = 2
#family = 3
#genus = 4
#species = 5
rank_idx = 5

class LoadDatasetInMemory(Dataset):
    def __init__(self,
                 dataset_dir_x,
                 dataset_dir_y,
                 total_samples,
                 loader_type = 'train'
                 ):

        self.dataset_dir_x = dataset_dir_x
        self.dataset_dir_y = dataset_dir_y
        self.total_samples = total_samples
        self.loader_type = loader_type

        print('loading {} data...'.format(loader_type))

        with open(self.dataset_dir_x, 'rb') as read_file:
            self.x_data = np.load(read_file)

        print('training data shape: {}'.format(self.x_data.shape))
        
        print('loading y data...')
        self.y_data = pd.read_csv(self.dataset_dir_y, header=None)

        self.num_classes = self.y_data.iloc[:,rank_idx].nunique()

        self.class_weights = self.y_data.sort_values(by=[rank_idx]).groupby([rank_idx])[rank_idx].count()/self.y_data.shape[0]
        self.class_weights = 1 / self.class_weights.astype(np.float32)
        self.class_weights = torch.as_tensor(self.class_weights/ np.sum(self.class_weights))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return torch.tensor(self.x_data[idx], dtype=torch.float), torch.tensor(self.y_data.iloc[idx, rank_idx])

class MetageNN(LightningModule):

    def __init__(self,
                 parser):
        super().__init__()

        self.parser = parser

        #load datasets
        print('loading train dataset...')
        self.dataset_train = LoadDatasetInMemory(
            dataset_dir_x=self.parser.data_loader_train['settings']['dataset_file_x'],
            dataset_dir_y=self.parser.data_loader_train['settings']['dataset_file_y'],
            total_samples=self.parser.data_loader_train['settings']['total_samples'],
            loader_type=self.parser.data_loader_train['settings']['loader_type'],

        )

        self.lr = lr
        self.loss_weight = self.parser.model_config['settings']['loss_weight']
        self.loss =  nn.CrossEntropyLoss(reduction='mean')

        if self.loss_weight:
            self.loss =  nn.CrossEntropyLoss(reduction='mean', weight=self.dataset_train.class_weights)


        self.layer_dim = self.parser.model_config['settings']['layer_dim']
        self.k_mer_dim = self.parser.model_config['settings']['k_mer_dim']
        self.num_classes = self.dataset_train.num_classes

        self.fc1 = nn.Linear(self.k_mer_dim, self.layer_dim)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.layer_dim, self.layer_dim)
        self.logits = nn.Linear(self.layer_dim, self.num_classes)
        
    def forward(self, x):
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.dp1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        logits = self.logits(x)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        self.log('train/loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True)
        return optimizer

    def train_dataloader(self):
        dataloader = DataLoader(self.dataset_train, batch_size=self.parser.data_loader_train['settings']['batch_size'],
                                shuffle=self.parser.data_loader_train['settings']['shuffle'],
                                num_workers=self.parser.data_loader_train['settings']['num_workers'],
                                drop_last=self.parser.data_loader_train['settings']['drop_last'],
                                pin_memory=True)
        return dataloader


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='MetageNN settings')
    args.add_argument('-s',
                      '--settings',
                      type=str,
                      help='settings file path')

    parser = args.parse_args()

    with open(parser.settings, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        parser = args.parse_args(namespace=t_args)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed, workers=True)

    save_trained_model_folder = parser.model_config['settings']['save_folder']
    lr = parser.model_config['settings']['lr']
    epochs = parser.model_config['settings']['epochs']
    gpus = parser.model_config['settings']['gpus']
    
    model = MetageNN(parser)

    trainer = Trainer(
        gpus=gpus,
        max_epochs=epochs, 
        deterministic=True,
        default_root_dir=save_trained_model_folder,
        reload_dataloaders_every_n_epochs=1
    )
    

    print('Training MetageNN')
    trainer.fit(model)
    
    