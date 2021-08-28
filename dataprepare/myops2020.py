import os, shutil
import cv2
import numpy as np
import nibabel as nb
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import KFold

from utils import helpers


def nii_to_npy(dataset_dir, save_path, to_png=True):
    """
    @LynnHg:
    MyoPS2020 Datasets
    three modals:
        bSSFP CMR, LGE CMR, T2 CMR
    Useage: run this function to split the 3d images to 2d slice, save data to npy or png
    :param dataset_dir: 原始3D nii 数据所在路径
    :param save_path: 处理后的数据存储路径
    :param to_png: 是否保存png格式数据
    :return:
    """
    im_gt_path = ['train25', 'train25_myops_gd']
    im_modals = ['c0', 'lge', 't2']
    # 所有模态数据
    all_modal_im_namelist = os.listdir(os.path.join(dataset_dir, im_gt_path[0]))
    all_modal_gt_namelist = os.listdir(os.path.join(dataset_dir, im_gt_path[1]))

    # 区分每个模态数据
    c0_im_namelist = sorted([item for item in all_modal_im_namelist if 'C0' in item])
    lge_im_namelist = sorted([item for item in all_modal_im_namelist if 'DE' in item])
    t2_im_namelist = sorted([item for item in all_modal_im_namelist if 'T2' in item])

    # 创建每个模态对应目录
    if os.path.exists('../media/LIBRARY/Datasets/MyoPS2020/npy'):
        # 若该目录已存在，则先删除，用来清空数据
        print('清空原始数据中...')
        shutil.rmtree(os.path.join('../media/LIBRARY/Datasets/MyoPS2020/npy'))
        print('原始数据已清空。')
    if os.path.exists('../media/LIBRARY/Datasets/MyoPS2020/png'):
        # 若该目录已存在，则先删除，用来清空数据
        print('清空原始数据中...')
        shutil.rmtree(os.path.join('../media/LIBRARY/Datasets/MyoPS2020/png'))
        print('原始数据已清空。')

    for m in im_modals:
        os.makedirs(os.path.join(save_path, 'npy', 'Images', m))
    os.makedirs(os.path.join(save_path, 'npy', 'Labels'))

    if to_png:
        for m in im_modals:
            os.makedirs(os.path.join(save_path, 'png', 'Images', m))
        os.makedirs(os.path.join(save_path, 'png', 'Labels'))

    # 每个模态下的所有病例
    for n in tqdm(range(len(all_modal_gt_namelist))):
        c0_im = nb.load(os.path.join(dataset_dir, im_gt_path[0], c0_im_namelist[n])).dataobj
        lge_im = nb.load(os.path.join(dataset_dir, im_gt_path[0], lge_im_namelist[n])).dataobj
        t2_im = nb.load(os.path.join(dataset_dir, im_gt_path[0], t2_im_namelist[n])).dataobj
        gt = nb.load(os.path.join(dataset_dir, im_gt_path[1], all_modal_gt_namelist[n])).dataobj
        h, w, c = gt.shape

        # 每一个病例的所有切片
        for i in range(c):
            mask = gt[:, :, i]

            # 跳过gt全黑图
            if np.sum(mask) == 0:
                continue
            c0_npy = c0_im[:, :, i]
            lge_npy = lge_im[:, :, i]
            t2_npy = t2_im[:, :, i]

            # 保存png格式
            if to_png:
                c0_png = helpers.array_to_img(np.expand_dims(c0_npy, axis=2))
                lge_png = helpers.array_to_img(np.expand_dims(lge_npy, axis=2))
                t2_png = helpers.array_to_img(np.expand_dims(t2_npy, axis=2))
                mask_png = helpers.array_to_img(np.expand_dims(mask, axis=2))

                file_name = all_modal_gt_namelist[n].split('.')[0][:-2] + '{}.png'.format(i)

                c0_png.save(os.path.join(save_path, 'png', 'Images/c0', file_name))
                lge_png.save(os.path.join(save_path, 'png', 'Images/lge', file_name))
                t2_png.save(os.path.join(save_path, 'png', 'Images/t2', file_name))
                mask_png.save(os.path.join(save_path, 'png', 'Labels', file_name))

            # 保存npy格式
            file_name = all_modal_gt_namelist[n].split('.')[0][:-2] + '{}.npy'.format(i)
            np.save(os.path.join(save_path, 'npy', 'Images/c0', file_name), c0_npy)
            np.save(os.path.join(save_path, 'npy', 'Images/lge', file_name), lge_npy)
            np.save(os.path.join(save_path, 'npy', 'Images/t2', file_name), t2_npy)
            np.save(os.path.join(save_path, 'npy', 'Labels', file_name), mask)

            with open(os.path.join(save_path, 'npy', 'all.txt'), 'a') as f:
                f.write(file_name)
                f.write('\n')

    print('数据预处理完成...')


def nii_to_npy_only(dataset_dir, save_path, to_png=True):
    """
    @LynnHg:
    MyoPS2020 Datasets
    three modals:
        bSSFP CMR, LGE CMR, T2 CMR
    Useage: run this function to split the 3d images to 2d slice, save data to npy or png
    :param dataset_dir: 原始3D nii 数据所在路径
    :param save_path: 处理后的数据存储路径
    :param to_png: 是否保存png格式数据
    :return:
    """
    im_gt_path = ['train25', 'train25_myops_gd']
    im_modals = ['c0', 'lge', 't2']
    # 所有模态数据
    all_modal_im_namelist = os.listdir(os.path.join(dataset_dir, im_gt_path[0]))
    all_modal_gt_namelist = os.listdir(os.path.join(dataset_dir, im_gt_path[1]))

    # 区分每个模态数据
    c0_im_namelist = sorted([item for item in all_modal_im_namelist if 'C0' in item])
    lge_im_namelist = sorted([item for item in all_modal_im_namelist if 'DE' in item])
    t2_im_namelist = sorted([item for item in all_modal_im_namelist if 'T2' in item])

    # 创建每个模态对应目录
    if os.path.exists('../media/LIBRARY/Datasets/MyoPS2020/only/npy'):
        # 若该目录已存在，则先删除，用来清空数据
        print('清空原始数据中...')
        shutil.rmtree(os.path.join('../media/LIBRARY/Datasets/MyoPS2020/only/npy'))
        print('原始数据已清空。')
    if os.path.exists('../media/LIBRARY/Datasets/MyoPS2020/only/png'):
        # 若该目录已存在，则先删除，用来清空数据
        print('清空原始数据中...')
        shutil.rmtree(os.path.join('../media/LIBRARY/Datasets/MyoPS2020/only/png'))
        print('原始数据已清空。')

    for m in im_modals:
        os.makedirs(os.path.join(save_path, 'only/npy', 'Images', m))
    os.makedirs(os.path.join(save_path, 'only/npy', 'Labels'))

    if to_png:
        for m in im_modals:
            os.makedirs(os.path.join(save_path, 'only/png', 'Images', m))
        os.makedirs(os.path.join(save_path, 'only/png', 'Labels'))

    # 每个模态下的所有病例
    for n in tqdm(range(len(all_modal_gt_namelist))):
        c0_im = nb.load(os.path.join(dataset_dir, im_gt_path[0], c0_im_namelist[n])).dataobj
        lge_im = nb.load(os.path.join(dataset_dir, im_gt_path[0], lge_im_namelist[n])).dataobj
        t2_im = nb.load(os.path.join(dataset_dir, im_gt_path[0], t2_im_namelist[n])).dataobj
        gt = nb.load(os.path.join(dataset_dir, im_gt_path[1], all_modal_gt_namelist[n])).dataobj
        h, w, c = gt.shape

        # 每一个病例的所有切片
        for i in range(c):
            mask = gt[:, :, i]

            # 跳过gt全黑图
            if np.sum(mask) == 0:
                continue
            c0_npy = c0_im[:, :, i]
            lge_npy = lge_im[:, :, i]
            t2_npy = t2_im[:, :, i]
            mask_copy = mask.copy()
            # mask.flags.writeable = True
            mask_copy[mask_copy == 200] = 0
            mask_copy[mask_copy == 500] = 0
            mask_copy[mask_copy == 600] = 0
            # 保存png格式
            if to_png:
                c0_png = helpers.array_to_img(np.expand_dims(c0_npy, axis=2))
                lge_png = helpers.array_to_img(np.expand_dims(lge_npy, axis=2))
                t2_png = helpers.array_to_img(np.expand_dims(t2_npy, axis=2))
                mask_png = helpers.array_to_img(np.expand_dims(mask_copy, axis=2))

                file_name = all_modal_gt_namelist[n].split('.')[0][:-2] + '{}.png'.format(i)

                c0_png.save(os.path.join(save_path, 'only/png', 'Images/c0', file_name))
                lge_png.save(os.path.join(save_path, 'only/png', 'Images/lge', file_name))
                t2_png.save(os.path.join(save_path, 'only/png', 'Images/t2', file_name))
                mask_png.save(os.path.join(save_path, 'only/png', 'Labels', file_name))

            # 保存npy格式
            file_name = all_modal_gt_namelist[n].split('.')[0][:-2] + '{}.npy'.format(i)
            np.save(os.path.join(save_path, 'only/npy', 'Images/c0', file_name), c0_npy)
            np.save(os.path.join(save_path, 'only/npy', 'Images/lge', file_name), lge_npy)
            np.save(os.path.join(save_path, 'only/npy', 'Images/t2', file_name), t2_npy)
            np.save(os.path.join(save_path, 'only/npy', 'Labels', file_name), mask_copy)

            with open(os.path.join(save_path, 'only/npy', 'all.txt'), 'a') as f:
                f.write(file_name)
                f.write('\n')

    print('数据预处理完成...')


# 数据增广
def data_augmentation(dataset_dir, save_path, to_png=True, rotate=False, flip=False, scale=False):
    from utils.misc import data_rotate
    from utils.joint_transforms import RandomScaleCrop
    save_path_init = save_path
    aug_data_file = ['train1', 'train2', 'train3', 'train4']
    # aug_data_file = ['train1']
    # aug_data_file = ['train80']

    for aug_data_file in aug_data_file:
        items = []
        im_modals = ['c0', 'lge', 't2']
        img_path = os.path.join(dataset_dir, 'Images')
        mask_path = os.path.join(dataset_dir, 'Labels')
        data_list = [l.strip('\n') for l in open(os.path.join(
            dataset_dir, '{}.txt'.format(aug_data_file))).readlines()]

        # 获取图像路径
        for it in data_list:
            item = ((os.path.join(img_path, 'c0', it),
                     os.path.join(img_path, 'lge', it),
                     os.path.join(img_path, 't2', it)),
                    os.path.join(mask_path, it))
            items.append(item)

        # 创建增广数据目录
        if os.path.exists(os.path.join('../media/LIBRARY/Datasets/MyoPS2020/Augdatas', aug_data_file)):
            # 若该目录已存在，则先删除，用来清空数据
            print('清空原始数据中...')
            shutil.rmtree(os.path.join('../media/LIBRARY/Datasets/MyoPS2020/Augdatas', aug_data_file))
            print('原始数据已清空。')

        save_path = os.path.join(save_path_init, 'Augdatas', aug_data_file)
        npy_save_path = os.path.join(save_path, 'npy')

        for m in im_modals:
            os.makedirs(os.path.join(npy_save_path, 'Images', m))
        os.makedirs(os.path.join(npy_save_path, 'Labels'))

        if to_png:
            png_save_path = os.path.join(save_path, 'png')
            for m in im_modals:
                os.makedirs(os.path.join(png_save_path, 'Images', m))
            os.makedirs(os.path.join(png_save_path, 'Labels'))

        # 加载图像
        for item in tqdm(items):
            img_paths, mask_path = item
            file_name = mask_path.split('\\')[-1][:-4]
            all_modals_img = []
            for i in range(len(im_modals)):
                all_modals_img.append(np.load(img_paths[i]))
            gt = np.load(mask_path)

            # 旋转图像
            if rotate:
                angles = [45, 90, 135, 180, 225, 270, 315]
                for angle in angles:
                    for i in range(len(im_modals)):
                        img_ = data_rotate(all_modals_img[i], angle)
                        gt_ = data_rotate(gt, angle)
                        extra = '_rotate{}'.format(angle)

                        if to_png:
                            png_file_name = file_name + '{}.png'.format(extra)
                            png_im = helpers.array_to_img(np.expand_dims(img_, axis=2))
                            png_gt = helpers.array_to_img(np.expand_dims(gt_, axis=2))
                            png_im.save(os.path.join(png_save_path, 'Images/{}'.format(im_modals[i]), png_file_name))
                            png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                        npy_file_name = file_name + '{}.npy'.format(extra)
                        np.save(os.path.join(npy_save_path, 'Images/{}'.format(im_modals[i]), npy_file_name), img_)
                        np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)

                        # 三个模态只需保存任意一次名字
                        if i == 0:
                            with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                                f.write(npy_file_name)
                                f.write('\n')
            # 随机缩放
            if scale:
                scale_rate = [0.75, 0.8, 0.9, 1, 1.1, 1.25]
                for sr in scale_rate:
                    SR = RandomScaleCrop(512, 512, scale_rate=sr)
                    for i in range(len(im_modals)):
                        extra = '_scale{}'.format(sr)
                        img = all_modals_img[i]
                        img_, gt_ = SR(Image.fromarray(img), Image.fromarray(gt))
                        img_, gt_ = np.array(img_), np.array(gt_)

                        if to_png:
                            png_file_name = file_name + '{}.png'.format(extra)
                            png_im = helpers.array_to_img(np.expand_dims(img_, axis=2))
                            png_gt = helpers.array_to_img(np.expand_dims(gt_, axis=2))
                            png_im.save(os.path.join(png_save_path, 'Images/{}'.format(im_modals[i]), png_file_name))
                            png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                        npy_file_name = file_name + '{}.npy'.format(extra)
                        np.save(os.path.join(npy_save_path, 'Images/{}'.format(im_modals[i]), npy_file_name), img_)
                        np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)

                        if i == 0:
                            with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                                f.write(npy_file_name)
                                f.write('\n')
            # 翻转图像
            if flip:
                # 翻转:水平 垂直 水平垂直
                flipCodes = [1, 0, -1]
                for code in flipCodes:
                    for i in range(len(im_modals)):
                        img_ = cv2.flip(all_modals_img[i], code)
                        gt_ = cv2.flip(gt, code)

                        if to_png:
                            png_file_name = file_name + '_flip{}.png'.format(code)
                            png_im = helpers.array_to_img(np.expand_dims(img_, axis=2))
                            png_gt = helpers.array_to_img(np.expand_dims(gt_, axis=2))
                            png_im.save(os.path.join(png_save_path, 'Images/{}'.format(im_modals[i]), png_file_name))
                            png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                        npy_file_name = file_name + '_flip{}.npy'.format(code)
                        np.save(os.path.join(npy_save_path, 'Images/{}'.format(im_modals[i]), npy_file_name), img_)
                        np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)
                        if i == 0:
                            with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                                f.write(npy_file_name)
                                f.write('\n')
        # shutil.copyfile(os.path.join(dataset_dir, 'val.txt'), os.path.join(save_path, 'val.txt'))


def dataset_kfold(dataset_dir, save_path):
    data_list = [l.strip('\n') for l in open(os.path.join(
        dataset_dir, 'all.txt')).readlines()]
    print(data_list)

    kf = KFold(5, True, 12345)

    for i, (tr, val) in enumerate(kf.split(data_list), 1):
        print(len(tr), len(val))
        if os.path.exists(os.path.join(save_path, 'train{}.txt'.format(i))):
            # 若该目录已存在，则先删除，用来清空数据
            print('清空原始数据中...')
            os.remove(os.path.join(save_path, 'train{}.txt'.format(i)))
            os.remove(os.path.join(save_path, 'val{}.txt'.format(i)))
            print('原始数据已清空。')

        for item in tr:
            with open(os.path.join(save_path, 'train{}.txt'.format(i)), 'a') as f:
                f.write(data_list[item])
                f.write('\n')

        for item in val:
            with open(os.path.join(save_path, 'val{}.txt'.format(i)), 'a') as f:
                f.write(data_list[item])
                f.write('\n')


if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    # 1.3d nii 图像转为 2d 切片
    # nii_to_npy(r'C:\Learning\Datasets\MyoPS2020', r'../media/LIBRARY/Datasets/MyoPS2020')

    # 2. 5折交叉验证数据集划分
    # dataset_kfold(r'..\media\LIBRARY\Datasets\MyoPS2020\npy', r'..\media\LIBRARY\Datasets\MyoPS2020\npy')

    # 3. 数据集增广
    data_augmentation(r'../media/LIBRARY/Datasets/MyoPS2020/npy', r'../media/LIBRARY/Datasets/MyoPS2020', rotate=True,
                      flip=True, scale=True)

