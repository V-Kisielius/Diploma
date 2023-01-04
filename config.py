import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)
print(f'Selected devie is {device}')

BOUND_THRASHOLD = 0.3

PATH_TO_PM_DATA = './data/pm_format'
PATH_TO_MARKUP_DATA = './data/CL_23.txt'
PATH_TO_EPOCH_OUTS = './imgs/epoch_outs'
PATH_TO_TARGET0 = './imgs/target0/'
PATH_TO_TARGET = './imgs/target/'
PATH_TO_INPUT = './imgs/input/'