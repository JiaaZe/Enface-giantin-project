import random

import cv2
import numpy as np
import tensorflow.keras.utils
from matplotlib import pyplot as plt
from patchify import patchify
from read_roi import read_roi_zip
import copy
import math
from tensorflow.python.keras.utils.generic_utils import Progbar

from matplotlib.patches import Ellipse


def purify_mask_withPlot(img, show_plot=True, has_sideview=False):
    """
    Used for find giantin mask when preprocessing imaging and roi.zip
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


def get_img_unpad(img, pad_size):
    height, width = img.shape[0:2]
    if len(img.shape) == 3:
        unpad = img[pad_size:height - pad_size, pad_size:width - pad_size, :]
    else:
        unpad = img[pad_size:height - pad_size, pad_size:width - pad_size]

    return unpad


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


def unpadding_image(padding_image_list, original_image_list):
    unpad_img_list = []
    for i, pad_img in enumerate(padding_image_list):
        original_shape = original_image_list[i].shape
        pad_img_shape = pad_img.shape
        pad_size = int((pad_img_shape[0] - original_shape[0]) / 2)
        unpad_img = get_img_unpad(pad_img, pad_size)
        unpad_img_list.append(unpad_img)
    return unpad_img_list


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


def unpatchify(patches, imsize, patch_step=206):
    """
    unpatchify the patches, the overlapped pixels are considered as the max value.
    :param patches: shape(n,n,h,h)
    :param imsize:
    :param patch_step:
    :return:
    """
    patches_size = patches.shape
    nrow, ncol, p_h, p_w = patches_size
    ret_image = np.zeros(shape=imsize, dtype=patches.dtype)
    for i in range(nrow):
        for j in range(ncol):
            cur_patch = np.copy(patches[i][j])
            start_x = j * patch_step
            start_y = i * patch_step
            if i == 0 and j == 0:
                ...
                # ret_image[start_x:start_x + p_w, start_y:start_y + p_h] = cur_patch
            elif i == 0:
                left_overlap = ret_image[start_y:start_y + p_h, start_x:start_x + p_w - patch_step]
                cur_patch[:, 0:p_w - patch_step] = np.max([cur_patch[:, 0:p_w - patch_step], left_overlap], axis=0)
            elif j == 0:
                top_overlap = ret_image[start_y:start_y + p_h - patch_step, start_x:start_x + p_w]
                cur_patch[0:p_h - patch_step, :] = np.max([cur_patch[0:p_h - patch_step, :], top_overlap], axis=0)
            else:
                left_overlap = ret_image[start_y:start_y + p_h, start_x:start_x + p_w - patch_step]
                cur_patch[:, 0:p_w - patch_step] = np.max([cur_patch[:, 0:p_w - patch_step], left_overlap], axis=0)
                top_overlap = ret_image[start_y:start_y + p_h - patch_step, start_x:start_x + p_w]
                cur_patch[0:p_h - patch_step, :] = np.max([cur_patch[0:p_h - patch_step, :], top_overlap], axis=0)
            ret_image[start_y:start_y + p_h, start_x: start_x + p_w] = cur_patch
    return ret_image


def pred_to_mask(preds, num_patches_per_image, patch_size=(256, 256), patch_step=206):
    ret_mask_list = []
    ret_mask_patches_list = []
    last_num = 0
    for num_patches in num_patches_per_image:
        mask_patches = preds[last_num:num_patches + last_num]
        last_num = num_patches + last_num
        num_rows_patches = int(np.sqrt(num_patches))
        mask_patches_np = np.array(mask_patches).reshape((num_rows_patches, num_rows_patches, patch_size[0],
                                                          patch_size[1]))
        imsize = patch_size[0] + (num_rows_patches - 1) * patch_step
        padded_mask = unpatchify(mask_patches_np, imsize=(imsize, imsize), patch_step=patch_step)
        ret_mask_list.append(padded_mask)
        ret_mask_patches_list.append(mask_patches)
    return ret_mask_list, ret_mask_patches_list


def check_contours(golgi_image, pred_mask, contour, min_giantin_area, giantin_possibility_threshold,
                   giantin_channel, rect_size=40, sub_list=None, show_plt=False):
    """
    Check pred_masks' contours
    :param sub_list: Last time bgst value in each channel. None then first sub.
    :param golgi_image: [h,w,c]
    :param pred_mask:
    :param contour:
    :param giantin_channel:
    :param min_giantin_area: minimum area of contour
    :param giantin_possibility_threshold: threshold of mean possibility of one giantin
    :param rect_size:
    :param show_plt:
    :return:
    """
    x, y, w, h = cv2.boundingRect(contour)
    max_size = max(w, h)
    if max_size >= rect_size:
        rect_size = (max_size // 10 + 1) * 10
    w_pad = rect_size - w
    h_pad = rect_size - h
    new_x = x - math.ceil(w_pad / 2)
    new_y = y - math.ceil(h_pad / 2)
    crop_golgi = np.copy(golgi_image[new_y:new_y + rect_size, new_x:new_x + rect_size, :])
    crop_mask = np.copy(pred_mask[new_y:new_y + rect_size, new_x:new_x + rect_size])
    print(x, y)
    if show_plt:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.imshow(crop_golgi[:, :, 0])
        plt.subplot(132)
        plt.imshow(crop_golgi[:, :, 1])
        plt.subplot(133)
        plt.imshow(crop_golgi[:, :, 2])
        plt.show()
    clear_golgi, giantin_mask, giantin_contour, flag, sub_list = check_golgi_crop(crop_golgi, crop_mask,
                                                                                  giantin_channel=giantin_channel,
                                                                                  sub_list=sub_list,
                                                                                  min_giantin_area=min_giantin_area,
                                                                                  giantin_possibility_threshold=
                                                                                  giantin_possibility_threshold)
    if show_plt:
        if flag:
            cmap = "Purples_r"
        else:
            cmap = "PuRd_r"
        plt.figure(figsize=(18, 10))
        plt.subplot(141)
        plt.imshow(clear_golgi[:, :, 0], cmap=cmap)
        plt.subplot(142)
        plt.imshow(clear_golgi[:, :, 1], cmap=cmap)
        plt.subplot(143)
        plt.imshow(clear_golgi[:, :, 2], cmap=cmap)
        if flag:
            plt.subplot(144)
            plt.imshow(giantin_mask, cmap=cmap)
        plt.show()
    return clear_golgi, flag, sub_list


def check_golgi_crop(golgi, pred_mask, giantin_channel, sub_list=None, blank_channel=-1, min_giantin_area=200,
                     giantin_possibility_threshold=0.5):
    """
    Check if selected giantin is availiable. Also do bgst.
    :param sub_list: Last time bgst value in each channel. None then first sub.
    :param golgi: golgi crop image [h,w,c]
    :param pred_mask:  crop model output
    :param giantin_channel:
    :param blank_channel:
    :param min_giantin_area: minimum area of giantin
    :param giantin_possibility_threshold: mean possibility of giantin threshold
    :return: bgst_golgi, giantin_contour, boolean
    """
    ret_flag = True
    h, w, c = golgi.shape
    copy_golgi = np.copy(golgi)
    giantin_contour = None
    giantin_mask = None
    giantin_found = False
    giantin_2_contours_found = False
    if sub_list is None:
        sub_list = [0 for _ in range(c)]
    for c_ in range(c):
        if c_ == blank_channel:
            continue
        task_img = copy_golgi[:, :, c_]
        sub = sub_list[c_]
        while True:
            if sub > 0:
                if not giantin_found:
                    sub_list[c_] += sub
                for i in range(h):
                    for j in range(w):
                        if task_img[i][j] > sub:
                            task_img[i][j] = task_img[i][j] - sub
                        else:
                            task_img[i][j] = 0
            channel_mask = task_img / (task_img + 1) * 255
            channel_mask = channel_mask.astype(np.uint8)
            _, channel_mask = cv2.threshold(channel_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(channel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            num_contours = len(contours)
            # sort by contour area
            contours = sorted(contours, key=lambda x: cv2.contourArea(x))
            do_sub = False
            if c_ == giantin_channel:
                for i, contour in enumerate(contours):
                    if giantin_found:
                        giantin_contour = contour
                        break
                    contour_area = cv2.contourArea(contour)
                    if contour_area <= min_giantin_area:
                        # clear the small contours
                        cv2.drawContours(task_img, contours, i, 0, -1)
                        num_contours -= 1
                        continue
                    else:
                        # clear the contours in the edge
                        c_x = contour[:, :, 0].reshape(-1, )
                        c_y = contour[:, :, 1].reshape(-1, )
                        if h - 1 in c_x or w - 1 in c_y or 0 in c_x or 0 in c_y:
                            if num_contours == 1:
                                # only one contour -> bgst.
                                do_sub = True
                                continue
                            else:
                                # clear the edge contours
                                cv2.drawContours(task_img, contours, i, 0, -1)
                                num_contours -= 1
                                continue
                        elif num_contours == 1:
                            giantin_found = True
                            giantin_contour = contour
                if do_sub:
                    sub = max(np.min(np.where(task_img > 0, task_img, np.inf)), 20)
                    # sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
                    continue
                if num_contours == 1:
                    # calculate giantin possibility
                    giantin_mask = np.zeros_like(task_img, dtype=np.uint8)
                    giantin_mask = cv2.drawContours(giantin_mask, [giantin_contour], 0, 1, -1)

                    total_possibility = pred_mask[np.where(giantin_mask > 0)]
                    mean_possibility = total_possibility.mean()
                    # print(mean_possibility)
                    if mean_possibility < giantin_possibility_threshold:
                        print("low possibility: {}".format(mean_possibility))
                        ret_flag = False
                        return copy_golgi, _, _, ret_flag, sub_list
                    giantin_found = True

                    # Calculate ratio=contour perimeter/enclosing circle perimeter
                    # if ratio < 1 and only one contour found then do bgst further
                    center, radius = cv2.minEnclosingCircle(giantin_contour)
                    p_circle = 2 * radius * np.pi
                    p_cnt = cv2.arcLength(giantin_contour, True)
                    ratio = p_cnt / p_circle
                    if ratio < 1.05:
                        channel_mask = task_img / (task_img + 1) * 255
                        channel_mask = channel_mask.astype(np.uint8)
                        _, channel_mask = cv2.threshold(channel_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # hierachy contours
                        contours, hierachy = cv2.findContours(channel_mask, cv2.RETR_TREE,
                                                              cv2.CHAIN_APPROX_NONE)
                        if len(contours) == 1:
                            if giantin_2_contours_found:
                                break
                            sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
                            continue
                        elif len(contours) == 2:
                            giantin_2_contours_found = True
                            # not hierachy contours
                            if hierachy[0][0][2] != 1 and hierachy[0][1][3] != 0:
                                sub = max(np.min(np.where(task_img > 0, task_img, np.inf)), 20)
                                continue
                            area_outer_cnt = cv2.contourArea(contours[0])
                            area_inner_cnt = cv2.contourArea(contours[1])
                            if area_inner_cnt / area_outer_cnt < 0.08:
                                sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
                                continue
                            giantin_mask = channel_mask
                    else:
                        print("ratio larger than 1.05. {}".format(ratio))
                    break
                else:
                    # no contour in giantin channel
                    print("no contour in giantin channel")
                    ret_flag = False
                    return copy_golgi, _, _, ret_flag, sub_list
            else:
                # Other channel
                for i, contour in enumerate(contours):
                    c_x = contour[:, :, 0].reshape(-1, )
                    c_y = contour[:, :, 1].reshape(-1, )
                    if h - 1 in c_x or w - 1 in c_y or 0 in c_x or 0 in c_y:
                        if num_contours == 1 or i == len(contours) - 1:
                            do_sub = True
                            break
                        else:
                            cv2.drawContours(task_img, contours, i, 0, -1)
                            num_contours -= 1
                            continue
                if do_sub:
                    sub = max(np.min(np.where(task_img > 0, task_img, np.inf)), 20)
                    continue
                # Having overlapping area with giantin channel.
                dilated_img = cv2.dilate(task_img, np.ones((3, 3)))
                giantin_crop = copy_golgi[:, :, giantin_channel]
                overlap = np.multiply(dilated_img, giantin_crop).sum()
                if overlap > 0:
                    ret_flag = True
                    break
                else:
                    # on overlapping with giantin channel
                    print("on overlapping with giantin channel")
                    ret_flag = False
                    return copy_golgi, _, _, ret_flag

            # old version
            # for i, contour in enumerate(contours):
            #     if giantin_found:
            #         giantin_contour = contour
            #         break
            #     # clear the small contours
            #     contour_area = cv2.contourArea(contour)
            #     if c_ == giantin_channel:
            #         if contour_area <= h * w * 0.1:
            #             cv2.drawContours(task_img, contours, i, 0, -1)
            #             num_contours -= 1
            #             continue
            #     # # Other channel doesn't limit contour area
            #     # else:
            #     #     if contour_area <= contour_area_threshold:
            #     #         cv2.drawContours(task_img, contours, i, 0, -1)
            #     #         num_contours -= 1
            #     #         continue
            #
            #     # contour in the edge
            #     c_x = contour[:, :, 0].reshape(-1, )
            #     c_y = contour[:, :, 1].reshape(-1, )
            #     if h - 1 in c_x or w - 1 in c_y or 0 in c_x or 0 in c_y:
            #         if num_contours == 1:
            #             # do the background subtraction
            #             do_sub = True
            #             break
            #         elif i != len(contours) - 1:
            #             # clear the edge contours
            #             cv2.drawContours(task_img, contours, i, 0, -1)
            #             num_contours -= 1
            #             continue
            #         else:
            #             do_sub = True
            #             break
            #     # clear small contours
            #     if i != len(contours) - 1:
            #         # clear the edge contours
            #         cv2.drawContours(task_img, contours, i, 0, -1)
            #         num_contours -= 1
            #         continue
            #
            #     if c_ == giantin_channel:
            #         giantin_contour = contour
            # if do_sub:
            #     sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
            # else:
            #     if c_ == giantin_channel:
            #         # giantin channel. Must be one contour
            #         if num_contours == 1:
            #             # calculate giantin possibility
            #             giantin_mask = np.zeros_like(task_img, dtype=np.uint8)
            #             giantin_mask = cv2.drawContours(giantin_mask, [giantin_contour], 0, 1, -1)
            #
            #             total_possibility = pred_mask[np.where(giantin_mask > 0)]
            #             mean_possibility = total_possibility.mean()
            #             # print(mean_possibility)
            #
            #             giantin_mask = copy_golgi[:, :, giantin_channel] / (copy_golgi[:, :, giantin_channel] + 1) * 255
            #             giantin_mask = giantin_mask.astype(np.uint8)
            #             _, giantin_mask = cv2.threshold(giantin_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #             if mean_possibility < giantin_possibility_threshold:
            #                 print("low possibility: {}".format(mean_possibility))
            #                 ret_flag = False
            #                 return copy_golgi, _, _, ret_flag, sub_list
            #             else:
            #                 giantin_found = True
            #                 # to calculate contour perimeter / enclosing circle perimeter
            #                 # if ratio < 1 and only one contour found then do bgst further
            #                 center, radius = cv2.minEnclosingCircle(giantin_contour)
            #                 p_circle = 2 * radius * np.pi
            #                 p_cnt = cv2.arcLength(giantin_contour, True)
            #                 ratio = p_cnt / p_circle
            #                 if ratio < 1.05:
            #                     channel_mask = task_img / (task_img + 1) * 255
            #                     channel_mask = channel_mask.astype(np.uint8)
            #                     _, channel_mask = cv2.threshold(channel_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #                     contours, hierachy = cv2.findContours(channel_mask, cv2.RETR_TREE,
            #                                                           cv2.CHAIN_APPROX_NONE)
            #                     if len(contours) == 1:
            #                         if giantin_2_contours_found:
            #                             break
            #                         sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
            #                         continue
            #                     elif len(contours) == 2:
            #                         giantin_2_contours_found = True
            #                         if hierachy[0][0][2] != 1 or hierachy[0][1][3] != 0:
            #                             sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
            #                             continue
            #                         area_outer_cnt = cv2.contourArea(contours[0])
            #                         area_inner_cnt = cv2.contourArea(contours[1])
            #                         if area_inner_cnt / area_outer_cnt < 0.08:
            #                             sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
            #                             continue
            #                 else:
            #                     print("ratio larger than 1.05. {}".format(ratio))
            #                 # print("p_circle: {}, p_cnt:{}, ratio:{}.".format(p_circle, p_cnt, p_cnt / p_circle))
            #                 # a_circle = radius * radius * np.pi
            #                 # a_cnt = len(np.where(giantin_mask > 0)[0])
            #                 # print("a_circle: {}, a_cnt:{}, ratio:{}.".format(a_circle, a_cnt, a_cnt / a_circle))
            #                 break
            #         elif num_contours > 1:
            #             sub = max(np.where(task_img > 0, task_img, np.inf).min(), 20)
            #         else:
            #             # no contour in giantin channel
            #             print("no contour in giantin channel")
            #             ret_flag = False
            #             return copy_golgi, _, _, ret_flag, sub_list
            #     else:
            #         # Other channels. Having overlapping area with giantin channel.
            #         if np.multiply(task_img, copy_golgi[:, :, giantin_channel]).sum() > 0:
            #             ret_flag = True
            #             break
            #         else:
            #             # on overlapping with giantin channel
            #             print("on overlapping with giantin channel")
            #             ret_flag = False
            #             return copy_golgi, _, _, ret_flag
    return copy_golgi, giantin_mask, giantin_contour, ret_flag, sub_list


def cal_center_of_mass(image):
    """
    calculate center of mass.
    :param image: shape is (h,w,c) or (h,w)
    :return: [(mx, my)]
    """
    shape_len = len(image.shape)
    assert shape_len == 2, "Dimension of image shape is not 2."
    h, w = image.shape
    total_intensity = 0
    Qx = 0
    Qy = 0
    for i in range(h):
        for j in range(w):
            intensity = int(image[i][j])
            total_intensity += intensity
            Qy += i * intensity
            Qx += j * intensity
    mx = round(Qx / total_intensity, 4)
    my = round(Qy / total_intensity, 4)
    return mx, my


def cal_gyradius(image, mx, my):
    """
    Calculate gyradius.
    :param mx: centor of mass in x-axis
    :param my: centor of mass in y-axis
    :param image: shape is (h,w)
    :return: gyradius
    """
    assert len(image.shape) == 2, "Dimension of image shape is not 2."
    h, w = image.shape
    Q = 0
    total_intensity = 0
    for i in range(h):
        for j in range(w):
            intensity = image[i][j]
            total_intensity += intensity
            Q += ((i - my) ** 2 + (j - mx) ** 2) * intensity
    gyradius = round(np.sqrt(Q / total_intensity), 4)
    return gyradius


def shift_make_border(image, giantin_channel, border_size=(701, 701), center_coord=(350, 350), shift_to_imageJ=True):
    """

    :param image:
    :param giantin_channel:
    :param border_size:
    :param center_coord:
    :param shift_to_imageJ: Boolean, whether to shift 0.5 pixel.
            There is 0.5 pixel shift for the center of mass calculation in comparison to imageJ.
    :return:
    """
    assert len(image.shape) == 3, "Dimension of image shape is not 3."
    h, w, c = image.shape
    pad_image = np.pad(image, ((0, border_size[0] - w), (0, border_size[1] - h), (0, 0)))
    mx, my = cal_center_of_mass(pad_image[:, :, giantin_channel])
    if shift_to_imageJ:
        mx += 0.5
        my += 0.5
    cx, cy = center_coord

    x_shift = cx - mx
    y_shift = cy - my

    mat_translation = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    dst = np.copy(pad_image)
    for c_ in range(c):
        dst[:, :, c_] = cv2.warpAffine(pad_image[:, :, c_], mat_translation, pad_image.shape[:2])
    print(cal_center_of_mass(dst[:, :, giantin_channel]))
    return dst


def normalize_total_intensity(image, target_total_intensity):
    len_shape = len(image.shape)
    assert len_shape == 2 or 3, "Dimension of image shape is neither 2 nor 3."
    total_intensity = np.sum(np.sum(image, axis=0), axis=0)
    ratio = target_total_intensity / total_intensity
    normalized_image_tmp = np.multiply(image, ratio)
    normalized_image = np.uint16(np.round(normalized_image_tmp))
    return normalized_image
