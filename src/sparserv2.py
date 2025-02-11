import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning.pytorch as pl
import sys

class Sparser(pl.LightningModule):
    def __init__(self,
                 )