import argparse
import os
import shutil
import glob


def create_datasets(dataset_dir, use_symlinks=True):
    activity_classes = ['Eyes Closed', 'Forward', 'Shoulder', 'Left Mirror', 'Lap', 'Speedometer', 'Radio', 'Rearview', 'Right Mirror']
    splits = ['train', 'val', 'test']
    
    # all data
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'all_data', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, '*_*_*', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'all_data', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'all_data', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating all_data dataset!')

    # no glasses
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'no_glasses', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, 'no_glasses_*', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'no_glasses', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'no_glasses', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating no_glasses dataset!')

    # with glasses
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'with_glasses', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, 'with_glasses_*', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'with_glasses', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'with_glasses', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating with_glasses dataset!')

    # day
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'day', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, '*_day', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'day', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'day', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating day dataset!')

    # night
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'night', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, '*_night', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'night', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'night', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating night dataset!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare gaze dataset')
    parser.add_argument('-o', '--dataset-dir', default='/home/akshay/data/lisat_gaze_data',
                        help='The dataset directory [default "/home/akshay/data/lisat_gaze_data"]')
    parser.add_argument('-s', '--no-symlinks', action='store_true',
                        help='Copy files instead of making symlinks')

    args = parser.parse_args()
    print('Creating datasets...')
    create_datasets(dataset_dir=args.dataset_dir, use_symlinks=not args.no_symlinks)
