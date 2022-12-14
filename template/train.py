import re
import argparse
import warnings

import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from tqdm.auto import tqdm
from dataloader import Dataloader
from models import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../dataset/train/train.csv')
    parser.add_argument('--dev_path', default='../dataset/train/validation.csv')
    parser.add_argument('--test_path', default='../dataset/test/evaluation.csv')
    parser.add_argument('--predict_path', default='../dataset/test/evaluation.csv')
    args = parser.parse_args(args=[])  

    # Ignore UserWarning
    warnings.filterwarnings(action='ignore', category=UserWarning)

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