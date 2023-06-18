from annotation import AnnotationRect, read_groundtruth_file
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Read all groundtruth annotations in the dataset at 'dataset_mmp/train'
    annotation_files = glob('dataset_mmp/train/*.txt')
    annotations = []
    for annotation_file in annotation_files:
        annotations.extend(read_groundtruth_file(annotation_file))

    # Create a histogramm of widths and aspect ratios of annotations
    widths = []
    aspect_ratios = []
    for annotation in annotations:
        widths.append(annotation.x2 - annotation.x1)
        aspect_ratios.append((annotation.y2 - annotation.y1) / (annotation.x2 - annotation.x1))

    widths = np.array(widths) / 320
    num_bins = 20
    # show histogramms as subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    bins = np.linspace(0.0, 1.0, num_bins)
    print(bins)
    ax1.hist(widths, bins=bins)
    ax1.set_title('Widths')
    bins = np.linspace(min(aspect_ratios), max(aspect_ratios), num_bins)
    ax2.hist(aspect_ratios, bins=bins)
    ax2.set_title('Aspect ratios')
    plt.savefig(f'grid_stats_{num_bins}.png')
    plt.show()


if __name__ == '__main__':
    main()
