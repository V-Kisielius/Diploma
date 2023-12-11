import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from scripts.imagedata import ImageData
from scripts.neutralliner import NeutralLiner
from scripts.config import PATH_TO_MCINTOSH

def test_me(n_runs=2,
            num_epochs=10,
            lr=5e-3,
            arch=(3, 6, 12, 24, 12, 6, 3, 1),
            weight_decay=1e-4,
            batch_num=''):
    device = torch.device(f'cuda:{int(batch_num)}' if torch.cuda.is_available() else 'cpu')
    filenames = os.listdir(os.path.join(PATH_TO_MCINTOSH, batch_num))
    mapnames = [filename[:-5] for filename in filenames]
    path_to_save = os.path.join(f'./Tests/Fits/Mean/big/', batch_num)
    os.makedirs(path_to_save, exist_ok=True)

    for i, (filename, mapname) in tqdm(enumerate(zip(filenames, mapnames)), desc=f'Predicting maps'):
        cur_dir = os.path.join(path_to_save, mapname)
        os.makedirs(cur_dir, exist_ok=True)
        predictions = []
        imgdata = ImageData(os.path.join(PATH_TO_MCINTOSH, filename), data_mode='fits')
        for n in tqdm(range(n_runs), desc=f'Predicting map {i+1}/{len(filenames)}'):
            model = NeutralLiner(image_list=[imgdata],
                                lr=lr,
                                help_step_size=1,
                                mode='3d',
                                arch=arch,
                                weight_decay=weight_decay)
            model.to(device)
            model.start_training(num_epochs=num_epochs, need_plot=False)
            prediction = model.test_model(need_plot=False)[0].view(model.image_list[0].img_array.shape).cpu().detach()
            predictions.append(prediction)
            model.save_state_dict(os.path.join(cur_dir, mapname + f'_{n}.pt'))
        predictions = torch.stack(predictions)
        prediction = torch.mean(predictions, dim=0)
        torch.save(prediction, os.path.join(cur_dir, mapname + '_mean.pt'))