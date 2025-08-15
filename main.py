
import torch
torch.cuda.empty_cache()

import world
from model.RecModel import *
from utils.utils import set_seed
from utils.dataloader import load_data
from utils.train_loop import TrainLoop

import sys

if __name__ == "__main__":
    print(f"Using {world.device} device: {torch.cuda.current_device()}")
    set_seed(world.seed)
    
    train_loader, test_loader = load_data(world.dataset)
    model, target_model = create_model(world,
                     train_loader.n_items,
                     train_loader.n_users,
                     train_loader.pad,
                     train_loader.getInterGraph())
    model = model.to(world.device)
    target_model = target_model.to(world.device)
    TrainLoop(world,
              train_loader,
              test_loader,
              model,
              target_model).run_loop()
    
    
