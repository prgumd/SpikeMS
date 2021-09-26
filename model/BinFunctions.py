import torch
import numpy as np

class OrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputAdded):
        start_dtype  = inputAdded.dtype
        inputBool = inputAdded.bool().type(start_dtype)
        return inputBool

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput 