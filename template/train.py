import re
import argparse

import torch
import pandas as pd
import pickle as pkl
from tqdm.auto import tqdm
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger

from dataloader import *
from models import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/dataset/train/train_split.csv')
    parser.add_argument('--dev_path', default='/opt/ml/dataset/train/val_split.csv')
    parser.add_argument('--test_path', default='/opt/ml/dataset/train/val_split.csv')
    parser.add_argument('--predict_path', default='/opt/ml/dataset/test/test_data.csv')
    args = parser.parse_args(args=[])  

    dataloader = Dataloader(
        args.tokenizer_name,
        args.batch_size,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
        shuffle=True
    )

    model = Model(
        args.model_name, 
        args.learning_rate,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epoch, 
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    
    model_name = re.sub(r'[/]', '-', args.model_name)

    torch.save(model, f'{model_name}.pt')