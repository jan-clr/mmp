from annotation import read_groundtruth_file
from glob import glob
import os


def print_label_dist(dir:str):
    annotation_files = glob(os.path.join(dir, '*.gt_data.txt'))

    dist = [0 for i in range(7)]

    for file in annotation_files:
        boxes = read_groundtruth_file(file)
        dist[len(boxes)] += 1
 
    print(dist)
    nr_multiple = sum(dist[2:])
    nr_single = dist[1]
    total = sum(dist)

    print(f"\nTotal: {total}\nMultiple: {nr_multiple} = {nr_multiple / total} * total\nSingle: {nr_single} = {nr_single / total} * total\n")

    
def main():
    val_dir = './dataset_mmp/val/'
    train_dir = './dataset_mmp/train/'

    print(train_dir)
    print_label_dist(train_dir)

    print(val_dir)
    print_label_dist(val_dir)
    

if __name__ == '__main__':
    main()
