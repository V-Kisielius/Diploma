import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # uncomment this line to run on CPU
torch.cuda.set_device(0)

LOSSES = ['loss', # the main loss which is the sum of all the following
          'f_abs_integral',
          'bound_integral',
          'orientation_integral',
          'f_integral',
          'MSE_help']

PATH_TO_PM_DATA = './Kislovodsk/pm_format' # path to the folder with PM data for Kislovodsk's maps
PATH_TO_MARKUP_DATA = './Kislovodsk/CL_23.txt' # path to the markup file for Kislovodsk's maps
PATH_TO_MCINTOSH = './imgs/fits/McIntosh' # path to the folder with McIntosh's maps
FITS_SHAPE = (256, 512) # shape to which the McIntosh's maps will be resized