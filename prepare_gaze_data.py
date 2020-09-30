import argparse
import os
import shutil
import glob


def create_datasets(dataset_dir, use_symlinks=True):
    activity_classes = ['Eyes Closed', 'Forward', 'Shoulder', 'Left Mirror', 'Lap', 'Speedometer', 'Radio', 'Rearview', 'Right Mirror']
    splits = ['train', 'val', 'test']
    
    # IR all data
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'ir_all_data', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, 'ir_*_*', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'ir_all_data', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'ir_all_data', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating ir_all_data dataset!')

    # IR no glasses
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'ir_no_glasses', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, 'ir_no_glasses_*', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'ir_no_glasses', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'ir_no_glasses', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating ir_no_glasses dataset!')

    # IR with glasses
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'ir_with_glasses', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, 'ir_with_glasses_*', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'ir_with_glasses', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'ir_with_glasses', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating ir_with_glasses dataset!')

    # IR day
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'ir_day', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, 'ir_*_day', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'ir_day', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'ir_day', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating ir_day dataset!')

    # IR night
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'ir_night', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, 'ir_*_night', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'ir_night', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'ir_night', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating ir_night dataset!')

    # RGB all data
    for split in splits:
        for activity in activity_classes:
            count = 0
            os.makedirs(os.path.join(dataset_dir, 'rgb_all_data', split, activity), exist_ok=True)
            images = glob.glob(os.path.join(dataset_dir, 'rgb_*_*', split, activity, '*.jpg'))
            for im in images:
                if use_symlinks:
                    os.symlink(im, os.path.join(dataset_dir, 'rgb_all_data', split, activity, '%.6d.jpg' % (count,)))
                else:
                    shutil.copyfile(im, os.path.join(dataset_dir, 'rgb_all_data', split, activity, '%.6d.jpg' % (count,)))
                count += 1
    print('Done creating rgb_all_data dataset!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare gaze dataset')
    parser.add_argument('-o', '--dataset-dir', default='/home/akshay/data/lisat_gaze_data',
                        help='The dataset directory [default "/home/akshay/data/lisat_gaze_data"]')
    parser.add_argument('-s', '--no-symlinks', action='store_true',
                        help='Copy files instead of making symlinks')

    args = parser.parse_args()
    print('Creating datasets...')
    create_datasets(dataset_dir=args.dataset_dir, use_symlinks=not args.no_symlinks)
