import os
import time
import logging
import argparse
import numpy as np
import warnings

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pyhocon import ConfigFactory

from dataloader import CLSDataset
from model.network import Classifier
from model.frequential import FreqNetwork
from model.sequential import SeqNetwork
from model.positional import PosNetwork
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataloader import CLSDataset

from transformers import BertTokenizer

torch.set_default_dtype(torch.float32)
class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        self.is_continue = is_continue
        self.mode = mode

        # Networks
        params_to_train = []
        self.freq = FreqNetwork(**self.conf['model.freq']).to(self.device)
        self.seq = SeqNetwork(**self.conf['model.seq']).to(self.device)
        self.pos = PosNetwork(**self.conf['model.pos']).to(self.device)
        self.cls = Classifier(self.freq, self.seq, self.pos, **self.conf['model.cls']).to(self.device)
        # params_to_train += list(self.freq.parameters())
        # params_to_train += list(self.seq.parameters())
        params_to_train += list(self.cls.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        # Load Data
        self.dataset = CLSDataset(**self.conf['dataset'])
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),map_location=self.device)
        self.cls.load_state_dict(checkpoint['cls'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'cls': self.cls.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def train(self):
        self.update_learning_rate()
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = None

        for epoch in range(self.end_iter):
            for X_train, y_train in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.end_iter}", leave=False):
                freq_input, seq_input, pos_input = X_train
                y_train = y_train[:,None].to(self.device)
                pred = self.cls(freq_input.to(self.device),seq_input.to(self.device),pos_input.to(self.device))
                # pred = self.cls(freq_input.to(self.device), seq_input.to(self.device))
                # Loss
                loss = criterion(pred, y_train)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.iter_step += 1
            print("loss=", loss.detach().cpu().numpy())
            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
            self.update_learning_rate()

if __name__ == '__main__':

    #FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    #logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/binarycls.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--case', type=str, default='fox_nbc')

    args = parser.parse_args()
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()