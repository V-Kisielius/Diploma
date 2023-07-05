import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)

LOSSES = ['loss', 'f_abs_integral', 'bound_integral', 'orientation_integral', 'f_integral']

PATH_TO_PM_DATA = './Kislovodsk/pm_format'
PATH_TO_MARKUP_DATA = './Kislovodsk/CL_23.txt'
PATH_TO_EPOCH_OUTS = './imgs/epoch_outs'
PATH_TO_TARGET0 = './imgs/target0/'
PATH_TO_TARGET = './imgs/target/'
PATH_TO_INPUT = './imgs/input/'