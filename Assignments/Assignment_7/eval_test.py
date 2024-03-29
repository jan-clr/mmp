import torch
import os

from model import MmpNet
from dataset import get_anchor_grid, get_dataloader
from main import evaluate_test


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    
    IMSIZE = 320
    SCALE_FACTOR = 32
    WIDTHS = [IMSIZE * i for i in [0.8, 0.65, 0.5, 0.4, 0.3, 0.2]]
    ASPECT_RATIOS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    NSM_THRESHOLD = 0.3

    RUN_ROOT_DIR = './runs/runs'
    #run_dir = f'{RUN_ROOT_DIR}/correctannot_filter_0.5_sgd_gridv3_sf_{SCALE_FACTOR}_negr{NEGATIVE_RATIO}_nsm_{NSM_THRESHOLD}_lr_{LR}_bs_{BATCH_SIZE}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    run_dir = f'{RUN_ROOT_DIR}/crop_False_flip_True_solarize_True_gauss_False_sgd_gridv3_sf_32_negr2.0_nsm_0.3_lgminiou_0.5_nodes_4800_lr_0.0001_bs_16_2023-06-20_21-36-47'

    anchor_grid = get_anchor_grid(int(IMSIZE / SCALE_FACTOR), int(IMSIZE / SCALE_FACTOR), scale_factor=SCALE_FACTOR, anchor_widths=WIDTHS, aspect_ratios=ASPECT_RATIOS)

    test_dataloader = get_dataloader('./dataset_mmp/test/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=True, apply_transforms_on_init=True)
    model = MmpNet(len(WIDTHS), len(ASPECT_RATIOS), IMSIZE, SCALE_FACTOR).to(DEVICE)

    # Continue Training
    #model.load_state_dict(torch.load(f'{run_dir}/best_model.pth'))
    evaluate_test(model, test_dataloader, DEVICE, anchor_grid, f'test_results.txt', threshold=NSM_THRESHOLD)


if __name__ == '__main__':
    main()
