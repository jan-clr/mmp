import torch
from model import MmpNet
from dataset import get_anchor_grid, get_dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
from evallib import calculate_ap_pr
from main import batch_inference, evaluate
from matplotlib import pyplot as plt


def calc_prcurve(model: MmpNet, loader: DataLoader, device: torch.device, nsm_threshold: float, anchor_grid, filter_threshold: float = 0.0):
    det_boxes_scores = {}
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for b_nr, (input, target, img_id, annotations) in loop:
        detected = batch_inference(model, input, device, anchor_grid, nsm_threshold, filter_threshold=filter_threshold)
        # filter out boxes with score < 0.5
        det_boxes_scores.update({img_id[i]: detected[i] for i in range(len(img_id))})
    _, pr, rc = calculate_ap_pr(det_boxes_scores, loader.dataset.transformed_annotations)
    print(_)
    return pr, rc


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    
    IMSIZE = 224
    SCALE_FACTOR = 32
    WIDTHS = [IMSIZE * i for i in [0.8, 0.65, 0.5, 0.4, 0.3, 0.2]]
    ASPECT_RATIOS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    NSM_THRESHOLD = 0.3

    RUN_ROOT_DIR = './runs/runs'
    run_dir = f'{RUN_ROOT_DIR}/crop_False_flip_True_solarize_True_gauss_False_adam_gridv3_sf_32_negr15.0_nsm_0.3_lgminiou_0.5_nodes_4800_lr_0.0001_bs_16_2023-07-30_12-39-45'
    #run_dir = f'{RUN_ROOT_DIR}/best_until_now'

    anchor_grid = get_anchor_grid(int(IMSIZE / SCALE_FACTOR), int(IMSIZE / SCALE_FACTOR), scale_factor=SCALE_FACTOR, anchor_widths=WIDTHS, aspect_ratios=ASPECT_RATIOS)

    val_dataloader = get_dataloader('./dataset_mmp/val/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=False, apply_transforms_on_init=True)
    model = MmpNet(len(WIDTHS), len(ASPECT_RATIOS), IMSIZE, SCALE_FACTOR).to(DEVICE)
    model.load_state_dict(torch.load(f'{run_dir}/best_model.pth'))
    #pr, rc = calc_prcurve(model, val_dataloader, DEVICE, 0.3, anchor_grid, filter_threshold=0.0)
    _, pr, rc = evaluate(model, val_dataloader, DEVICE, anchor_grid, nms_threshold=NSM_THRESHOLD, filter_threshold=0.0)

    plt.plot(rc, pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f'pr_curve.png')


if __name__ == '__main__':
    main()
