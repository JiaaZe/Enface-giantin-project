import random

import cv2
import numpy as np
import tensorflow.keras.utils
from matplotlib import pyplot as plt
from patchify import patchify
from read_roi import read_roi_zip
import copy
import math

from matplotlib.patches import Ellipse


def purify_mask_withPlot(img, show_plot=True, has_sideview=False):
    """

    :param has_sideview: exist sideview roi
    :param img:  roi of giantin
    :param show_plot: show plots
    :return:
    """
    row, column = 1, 4
    sub = 0
    h, w = img.shape
    task_img = np.copy(img)
    if show_plot:
        mask = task_img / (task_img + 1) * 255
        mask = mask.astype(np.uint8)
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.figure(figsize=(10, 10))
        plt.subplot(row, column, 1)
        plt.imshow(mask)
    sub_count = 0
    while True:
        if sub > 0:
            sub_count += 1
            for i in range(h):
                for j in range(w):
                    if task_img[i][j] > sub:
                        task_img[i][j] = task_img[i][j] - sub
                    else:
                        task_img[i][j] = 0

        mask = task_img / (task_img + 1) * 255
        mask = mask.astype(np.uint8)
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours)
        accept_contours = []
        if sub_count == 300:
            return mask, -1
        # sort by contour area
        contours = sorted(contours, key=lambda x: cv2.contourArea(x))
        do_sub = False
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) <= 100:
                # clear the small contours
                cv2.drawContours(mask, contours, i, 0, -1)
                cv2.drawContours(task_img, contours, i, 0, -1)
                num_contours -= 1
                continue
            c_x = contour[:, :, 0].reshape(-1, )
            c_y = contour[:, :, 1].reshape(-1, )
            if h - 1 in c_x or w - 1 in c_y or 0 in c_x or 0 in c_y:
                # contour in the edge
                if num_contours == 1:
                    # do the background subtraction
                    do_sub = True
                    break
                else:
                    # clear the edge contours
                    cv2.drawContours(mask, contours, i, 0, -1)
                    cv2.drawContours(task_img, contours, i, 0, -1)
                    num_contours -= 1
                    continue
            accept_contours.append(contour)
        if do_sub or len(accept_contours) > 1:
            sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
            # sub = 40
        else:
            if num_contours == 0:
                return mask, -1
            elif num_contours == 1:
                if show_plot:
                    plt.subplot(row, column, 2)
                    plt.imshow(mask)

                # Fill the center hole
                cv2.drawContours(mask, accept_contours, 0, 1, -1)

                # rect
                rect = cv2.minAreaRect(accept_contours[0])
                f1 = min(rect[1]) / max(rect[1])
                # circle
                (r_x, r_y), radius = cv2.minEnclosingCircle(accept_contours[0])
                # ellipse
                ellipse = cv2.fitEllipse(accept_contours[0])

                pixel_counts = len(np.where(mask > 0)[0])
                ellpise_area = np.pi * ellipse[1][0] / 2 * ellipse[1][1] / 2
                circle_area = radius ** 2 * np.pi

                f2 = pixel_counts / circle_area
                f3 = ellpise_area / circle_area
                print("f1:{}, f2:{}, f3:{}".format(f1, f2, f3))

                def before_return():
                    if show_plot:
                        ax3 = plt.subplot(row, column, 3)
                        plt_circle = plt.Circle(xy=(r_x, r_y), radius=radius, fc='green', alpha=0.5)
                        plt_ellpise = Ellipse(xy=ellipse[0], width=ellipse[1][0], height=ellipse[1][1],
                                              angle=ellipse[2],
                                              fc="red", alpha=0.5)
                        ax3.add_artist(plt_circle)
                        ax3.add_artist(plt_ellpise)
                        plt.imshow(mask)

                        ax4 = plt.subplot(row, column, 4)
                        box = cv2.boxPoints(rect)
                        plt_box = plt.Rectangle((box[1]), rect[1][0], rect[1][1], rect[2], alpha=0.5)
                        ax4.add_artist(plt_box)
                        plt.imshow(mask)

                        plt.show()

                if f1 < 0.6:
                    # side view
                    before_return()
                    return mask, -1
                if f3 > 0.7 or f2 > 0.6:
                    # accpet en face
                    before_return()
                    return mask, 1
                else:
                    if has_sideview or f2 < 0.45:
                        # reject enface
                        before_return()
                        return mask, 0
                    else:
                        sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)


def select_roi(folder_num, image_num, roi_path, img, show_plt=True, has_sideview=False):
    roi = read_roi_zip(roi_path)
    roi_coords = []
    accept = []
    reject = []
    ret_mask = np.zeros_like(img, dtype=np.uint8)
    row, col = 1, 2
    # a = Progbar(len(roi.values()))
    for i, v in enumerate(roi.values()):
        print(folder_num, image_num, i)
        if v['type'] == 'rectangle':
            roi_x, roi_y = v['left'], v['top']
            roi_width, roi_height = v['width'], v['height']
            roi_coords.append([roi_x, roi_y, roi_width, roi_height])
            roi_rect = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            mask, flag = purify_mask_withPlot(roi_rect, show_plt, has_sideview)

            plt.figure(figsize=(8, 8))
            plt.subplot(row, col, 1)
            plt.imshow(roi_rect)
            plt.title("raw giantin roi")

            plt.subplot(row, col, 2)
            plt.title("giantin mask")
            if flag == 1:
                accept.append(mask)
                reject.append(None)
                plt.imshow(mask, cmap="Greens")
                plt.title("accept enface")
                ret_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = mask
            elif flag == 0:
                reject.append(mask)
                accept.append(None)
                plt.imshow(mask, cmap="Oranges")
                plt.title("reject enface")
            elif flag == -1:
                reject.append(mask)
                accept.append(None)
                plt.imshow(mask, cmap="PuRd_r")
                plt.title("side view")

                # plt.subplot(row, col, 3)
                # plt.title("dilate raw giantin roi")
                # kernel = np.ones((2, 2), np.uint8)
                # opening = cv2.morphologyEx(roi_rect, cv2.MORPH_DILATE, kernel, iterations=3)
                # plt.imshow(opening)
                #
                # plt.subplot(row, col, 4)
                # plt.title("dilate raw giantin roi mask")
                # opening = opening / (opening + 1) * 255
                # opening = opening.astype(np.uint8)
                # _, opening = cv2.threshold(opening, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # plt.imshow(opening)
            plt.show()
        else:
            print("No.{} roi type is not rectangle".format(i))
        # a.update(i + 1)
    return roi_coords, accept, reject


def filter_roi(folder_roi_list, num_lists, accept_flag=True):
    ret_list = copy.deepcopy(folder_roi_list)
    for num_list in num_lists:
        folder_num, image_num, roi_num = num_list
        for roi in roi_num:
            if accept_flag:
                # accept roi
                reject_roi = ret_list[folder_num][image_num]["reject_roi"][roi]
                if reject_roi is not None:
                    ret_list[folder_num][image_num]["accept_roi"][roi] = reject_roi
                    ret_list[folder_num][image_num]["reject_roi"][roi] = None
                else:
                    print("[{},{},{}] reject is None".format(folder_num, image_num, roi))
            else:
                # reject roi
                accept_roi = ret_list[folder_num][image_num]["accept_roi"][roi]
                if accept_roi is not None:
                    ret_list[folder_num][image_num]["reject_roi"][roi] = accept_roi
                    ret_list[folder_num][image_num]["accept_roi"][roi] = None
                else:
                    print("[{},{},{}] reject is None".format(folder_num, image_num, roi))
    return ret_list


def roiList_to_mask(folder_roi_list):
    giantin_list = []
    mask_list = []
    for folder in folder_roi_list:
        for images in folder:
            giantin_tif = images['giantin_tif']
            mask = np.zeros_like(giantin_tif, dtype=np.bool)
            roi_coords = images['roi_coords']
            accept_roi = images['accept_roi']
            for i, roi in enumerate(accept_roi):
                if roi is not None:
                    roi_x, roi_y, roi_width, roi_height = roi_coords[i]
                    mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = np.logical_or(
                        mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], roi)
            giantin_list.append(giantin_tif)
            mask = mask.astype(np.uint8)
            mask_list.append(mask)
    return giantin_list, mask_list


def get_img_pad(img, patch_size=256, patchify_step=206):
    h, w = img.shape
    max_side = max(h, w)
    target_size = math.ceil((max_side - patch_size) / patchify_step) * patchify_step + patch_size
    diff_h = target_size - h
    diff_w = target_size - w
    if diff_h % 2 != 0:
        h_pad = (round(diff_h / 2) + 1, round(diff_h / 2) - 1)
    else:
        h_pad = round(diff_h / 2)
    if diff_w % 2 != 0:
        w_pad = (round(diff_w / 2) + 1, round(diff_w / 2) - 1)
    else:
        w_pad = round(diff_w / 2)
    # print(h_pad, w_pad)
    img_pad = np.pad(img, (h_pad, w_pad), "constant", constant_values=0)
    return target_size, img_pad


def padding_image(image_list, do_patchify=True, clear_edge_roi=True, patch_size=(256, 256), patch_step=206):
    pad_image_list = []
    patches_list = []
    for image in image_list:
        target_size, pad_image = get_img_pad(image)
        print(target_size)
        pad_image_list.append(pad_image)
        if do_patchify:
            patches = patchify(pad_image, patch_size, step=patch_step)
            if clear_edge_roi:
                patches = clear_mask_patches(patches)
            patches = patches.reshape((-1, patch_size[0], patch_size[1]))
            patches_list.append(patches)
    return pad_image_list, patches_list


def clear_mask_patches(patches_in):
    cleared_patches = np.copy(patches_in)
    n = patches_in.shape[0]
    m = patches_in.shape[1]
    h, w = patches_in.shape[2:]
    for i in range(n):
        for j in range(m):
            if cleared_patches[i][j].sum() == 0:
                continue
            contours, _ = cv2.findContours(cleared_patches[i][j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for n, contour in enumerate(contours):
                c_x = contour[:, :, 0].reshape(-1, )
                c_y = contour[:, :, 1].reshape(-1, )
                if h - 1 in c_x or w - 1 in c_y or 0 in c_x or 0 in c_y:
                    # clear the edge contours
                    cv2.drawContours(cleared_patches[i][j], contours, n, 0, -1)
                    continue
    return cleared_patches


def make_model_input(image_list, do_norm=True, data_shape=(-1, 256, 256, 1)):
    number_patches_perimage = []
    out = image_list[0]
    number_patches_perimage.append(out.shape[0])
    for image in image_list[1:]:
        number_patches_perimage.append(image.shape[0])
        out = np.append(out, image, axis=0)
    if do_norm:
        out = tensorflow.keras.utils.normalize(out, axis=1)
    out = out.reshape(data_shape)
    return out, number_patches_perimage


def clear_blank_mask(images, masks, ratio_of_blank=0.1):
    new_masks_tmp = []
    new_images_tmp = []
    zero_masks_tmp = []
    zero_images_tmp = []
    for i, mask in enumerate(masks):
        if np.sum(mask.flatten()) > 0:
            new_masks_tmp.append(mask)
            new_images_tmp.append(images[i])
        else:
            zero_masks_tmp.append(mask)
            zero_images_tmp.append(images[i])
    if ratio_of_blank > 0:
        random.shuffle(zero_images_tmp)
    n = int(ratio_of_blank * len(new_images_tmp))
    ret_images = np.array(new_images_tmp + zero_images_tmp[:n])
    ret_masks = np.array(new_masks_tmp + zero_masks_tmp[:n])
    return ret_images, ret_masks
