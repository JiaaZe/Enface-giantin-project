import random

import cv2
import numpy as np
import pandas as pd
import tensorflow.keras.utils
from matplotlib import pyplot as plt
from patchify import patchify
from read_roi import read_roi_zip
import copy
import math

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
                   giantin_channel, rect_size=40, sub_list=None, show_plt=False, overlapping=True):
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
    :param overlapping: giantin channel have overlapping with other channel
    :return:
    """
    golgi_h, golgi_w, golgi_c = golgi_image.shape
    x, y, w, h = cv2.boundingRect(contour)
    max_size = max(w, h)
    if max_size >= rect_size:
        rect_size = (max_size // 10 + 1) * 10
    w_pad = rect_size - w
    h_pad = rect_size - h
    x0 = max(x - math.ceil(w_pad / 2), 0)
    y0 = max(y - math.ceil(h_pad / 2), 0)
    x1 = x0 + rect_size
    y1 = y0 + rect_size
    if x1 > golgi_w - 1:
        rect_size = golgi_w - x0
    if y1 > golgi_h - 1:
        rect_size = golgi_h - y0
    x1 = x0 + rect_size
    y1 = y0 + rect_size
    crop_golgi = np.copy(golgi_image[y0:y1, x0:x1, :])
    crop_mask = np.copy(pred_mask[y0:y1, x0:x1])
    print(x, y)
    if show_plt:
        plt.figure(figsize=(18, 10))
        plt.subplot(141)
        plt.title("channel 0")
        plt.imshow(crop_golgi[:, :, 0])
        plt.subplot(142)
        plt.title("channel 1")
        plt.imshow(crop_golgi[:, :, 1])
        if golgi_c == 3:
            plt.subplot(143)
            plt.title("channel 2")
            plt.imshow(crop_golgi[:, :, 2])
        plt.subplot(144)
        plt.title("crop_mask")
        plt.imshow(crop_mask)
        plt.show()
    clear_golgi, giantin_mask, giantin_contour, flag, sub_list, reject_msg = check_golgi_crop(crop_golgi, crop_mask,
                                                                                              giantin_channel=giantin_channel,
                                                                                              sub_list=sub_list,
                                                                                              min_giantin_area=min_giantin_area,
                                                                                              giantin_possibility_threshold=
                                                                                              giantin_possibility_threshold,
                                                                                              have_overlapping=overlapping)
    if flag:
        circularity = 4 * np.pi * cv2.contourArea(giantin_contour) / (
                cv2.arcLength(giantin_contour, True) ** 2)
        print("circularity is {}".format(circularity))
        # enclosing circle
        _, r_giantin = cv2.minEnclosingCircle(giantin_contour)
        area_circle = r_giantin ** 2 * np.pi
        # min area rect
        center_, (min_rect_h, min_rect_w), angle = cv2.minAreaRect(giantin_contour)
        # bounding rect
        x_, y_, rect_h, rect_w = cv2.boundingRect(giantin_contour)
        area_bound_rect = rect_h * rect_w

        area_contour = cv2.contourArea(giantin_contour)

        area_enclosing_circle_ratio = area_contour / area_circle
        area_min_rect_ratio = min(min_rect_h, min_rect_w) / max(min_rect_h, min_rect_w)
        area_bound_rect_ratio = area_contour / area_bound_rect
        print("ratio of minAreaRect :{}".format(area_min_rect_ratio))
        print("ratio of minEnclosingCircle :{}".format(area_enclosing_circle_ratio))
        print("ratio of boundingRect :{}".format(area_bound_rect_ratio))
        #
        circle = np.zeros(shape=(50, 50), dtype=np.uint8)
        cv2.circle(circle, (25, 25), radius=10, color=1, thickness=1)
        circle_contour, _ = cv2.findContours(circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        circle_ret = cv2.matchShapes(giantin_contour, circle_contour[0], 1, 0.0)
        print("circle: {}".format(circle_ret))

        ellipse = np.zeros(shape=(50, 50), dtype=np.uint8)
        cv2.ellipse(ellipse, (25, 25), axes=(5, 10), color=1, thickness=2, angle=0, startAngle=0, endAngle=360)
        ellipse_contour, _ = cv2.findContours(ellipse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ellipse_ret = cv2.matchShapes(giantin_contour, ellipse_contour[0], 1, 0.0)
        print("ellipse: {}".format(ellipse_ret))
    #
    #     hull = cv2.convexHull(giantin_contour, returnPoints=False)
    #     # 寻找凸缺陷
    #     defects = cv2.convexityDefects(giantin_contour, hull)
    #     d_list = []
    #     for i in range(defects.shape[0]):
    #         s, e, f, d = defects[i, 0]
    #         d_list.append(d)
    #
    #         start = tuple(giantin_contour[s, 0])
    #         end = tuple(giantin_contour[e, 0])
    #         far = tuple(giantin_contour[f, 0])
    #         cv2.line(crop_mask, start, end, 5, 1)
    #         cv2.circle(crop_mask, far, 1, 10, -1)
    #
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(crop_mask)
    #     plt.show()
    #     print(d_list)
    #     print(np.array(d_list).mean(), np.array(d_list).std())
    #     # area_ratio = area_contour / area_circle
    #     # if area_ratio < 0.5:
    #     #     print("ratio between minAreaRect area of giantin_contour and pred_contour: {} < 0.5".format(
    #     #         area_ratio))
    #     #     flag = False
    #     # else:
    #     #     print("ratio between minAreaRect area of giantin_contour and pred_contour: {}".format(
    #     #         area_ratio))
    else:
        print(reject_msg)
    if show_plt:
        if flag:
            cmap = "Purples_r"
        else:
            cmap = "PuRd_r"
        plt.figure(figsize=(18, 10))
        plt.subplot(141)
        plt.title("channel 0")
        plt.imshow(clear_golgi[:, :, 0], cmap=cmap)
        plt.subplot(142)
        plt.title("channel 1")
        plt.imshow(clear_golgi[:, :, 1], cmap=cmap)
        if golgi_c == 3:
            plt.subplot(143)
            plt.title("channel 2")
            plt.imshow(clear_golgi[:, :, 2], cmap=cmap)
        if flag:
            plt.subplot(144)
            plt.title("giantin mask")
            # hull = cv2.convexHull(giantin_contour, returnPoints=False)
            # # 检测凸凹陷
            # defects = cv2.convexityDefects(giantin_contour, hull)
            # for i in range(defects.shape[0]):
            #     s, e, f, d = defects[i, 0]
            #     start = tuple(giantin_contour[s][0])
            #     end = tuple(giantin_contour[e][0])
            #     far = tuple(giantin_contour[f][0])
            #     print(start,end, far)
            #     cv2.line(giantin_mask, start, end, 10, 1)
            #     cv2.circle(giantin_mask, far, 1, 5, -1)
            plt.imshow(giantin_mask, cmap=cmap)
        plt.show()
    return clear_golgi, giantin_contour, flag, sub_list


def check_golgi_crop(golgi, pred_mask, giantin_channel, sub_list=None, blank_channel=-1, min_giantin_area=200,
                     giantin_possibility_threshold=0.5, have_overlapping=True):
    """
    Check if selected giantin is availiable. Also do bgst.
    :param sub_list: Last time bgst value in each channel. None then first sub.
    :param golgi: golgi crop image [h,w,c]
    :param pred_mask:  crop model output
    :param giantin_channel:
    :param blank_channel:
    :param min_giantin_area: minimum area of giantin
    :param giantin_possibility_threshold: mean possibility of giantin threshold
    :param have_overlapping: giantin channel have overlapping with other channel
    :return: bgst_golgi, giantin_contour, boolean
    """
    circle_contour = np.array(
        [[[25, 15]], [[24, 16]], [[23, 16]], [[22, 16]], [[21, 16]], [[20, 17]], [[19, 17]], [[18, 18]], [[17, 19]],
         [[17, 20]], [[16, 21]], [[16, 22]], [[16, 23]], [[16, 24]], [[15, 25]], [[16, 26]], [[16, 27]], [[16, 28]],
         [[16, 29]], [[17, 30]], [[17, 31]], [[18, 32]], [[19, 33]], [[20, 33]], [[21, 34]], [[22, 34]], [[23, 34]],
         [[24, 34]], [[25, 35]], [[26, 34]], [[27, 34]], [[28, 34]], [[29, 34]], [[30, 33]], [[31, 33]], [[32, 32]],
         [[33, 31]], [[33, 30]], [[34, 29]], [[34, 28]], [[34, 27]], [[34, 26]], [[35, 25]], [[34, 24]], [[34, 23]],
         [[34, 22]], [[34, 21]], [[33, 20]], [[33, 19]], [[32, 18]], [[31, 17]], [[30, 17]], [[29, 16]], [[28, 16]],
         [[27, 16]], [[26, 16]]], dtype=np.int32)
    ellipse_contour = np.array(
        [[[23, 14]], [[22, 15]], [[21, 16]], [[21, 17]], [[20, 18]], [[20, 19]], [[20, 20]], [[19, 21]], [[19, 22]],
         [[19, 23]], [[19, 24]], [[19, 25]], [[19, 26]], [[19, 27]], [[19, 28]], [[20, 29]], [[20, 30]], [[20, 31]],
         [[20, 32]], [[21, 33]], [[21, 34]], [[22, 35]], [[23, 36]], [[24, 36]], [[25, 36]], [[26, 36]], [[27, 36]],
         [[28, 35]], [[29, 34]], [[29, 33]], [[30, 32]], [[30, 31]], [[30, 30]], [[30, 29]], [[31, 28]], [[31, 27]],
         [[31, 26]], [[31, 25]], [[31, 24]], [[31, 23]], [[31, 22]], [[31, 21]], [[30, 20]], [[30, 19]], [[30, 18]],
         [[29, 17]], [[29, 16]], [[28, 15]], [[27, 14]], [[26, 14]], [[25, 14]], [[24, 14]]], dtype=np.int32)
    had_convex = False
    min_sub = 50
    reject_msg = ""
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
            do_sub = False
            if c_ == giantin_channel:
                # sort by contour area
                contours = sorted(contours, key=lambda x: cv2.contourArea(x))
                for i, contour in enumerate(contours):
                    if giantin_found:
                        # bgst to find inner circle.
                        if i != len(contours) - 1:
                            # clear small contour
                            cv2.drawContours(task_img, contours, i, 0, -1)
                            num_contours -= 1
                            continue
                        else:
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
                        else:
                            # no. of contours >= 2, select the last one which is the largest.
                            if i != len(contours) - 1:
                                # clear small contour
                                cv2.drawContours(task_img, contours, i, 0, -1)
                                num_contours -= 1
                                continue
                            else:
                                giantin_contour = contour
                                break
                if do_sub:
                    sub = max(np.min(np.where(task_img > 0, task_img, np.inf)), min_sub)
                    continue
                if num_contours == 1:
                    if not giantin_found:
                        continue
                    # calculate giantin possibility
                    giantin_mask = np.zeros_like(task_img, dtype=np.uint8)
                    giantin_mask = cv2.drawContours(giantin_mask, [giantin_contour], 0, 1, -1)

                    total_possibility = pred_mask[np.where(giantin_mask > 0)]
                    mean_possibility = total_possibility.mean()
                    # print(mean_possibility)
                    if mean_possibility < giantin_possibility_threshold:
                        ret_flag = False
                        reject_msg = "low possibility: {}.".format(mean_possibility)
                        return copy_golgi, _, _, ret_flag, sub_list, reject_msg
                    giantin_found = True

                    # Calculate ratio: contour perimeter/enclosing circle perimeter
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
                            # one contour, further bgst
                            if giantin_2_contours_found:
                                # got only one contour due to subtract too large
                                break
                            sub = max(np.where(task_img > 0, task_img, np.inf).min(), min_sub)
                            continue
                        elif len(contours) == 2:
                            giantin_2_contours_found = True
                            # not hierachy contours
                            if hierachy[0][0][2] != 1 and hierachy[0][1][3] != 0:
                                sub = max(np.min(np.where(task_img > 0, task_img, np.inf)), min_sub)
                                continue
                            area_outer_cnt = cv2.contourArea(contours[0])
                            area_inner_cnt = cv2.contourArea(contours[1])
                            if area_inner_cnt / area_outer_cnt < 0.05:
                                # inner contour area too small, further bgst.
                                sub = max(np.where(task_img > 0, task_img, np.inf).min(), min_sub)
                                continue
                            circle_match = cv2.matchShapes(contours[0], circle_contour, 1, 0)
                            if circle_match > 0.1:
                                ellipse_match = cv2.matchShapes(contours[0], ellipse_contour, 1, 0)
                                if ellipse_match > 0.1:
                                    # contour shape is not satisified.
                                    reject_msg = "contour shape is not satisified."
                                    ret_flag = False
                                    return copy_golgi, _, _, ret_flag, sub_list, reject_msg
                                else:
                                    # find convexity defects to further bgst
                                    hull = cv2.convexHull(contours[0], returnPoints=False)
                                    # 寻找凸缺陷
                                    defects = cv2.convexityDefects(contours[0], hull)
                                    large_d_count = 0
                                    for i in range(defects.shape[0]):
                                        _, _, _, d = defects[i, 0]
                                        if d > 500:
                                            large_d_count += 1
                                    if large_d_count > 0:
                                        had_convex = True
                                        sub = max(np.where(task_img > 0, task_img, np.inf).min(), min_sub)
                                        continue
                                    else:
                                        print("Ellipse match: {}".format(ellipse_match))
                            else:
                                print("Circle match: {}".format(circle_match))
                            giantin_mask = channel_mask
                        else:
                            if cv2.contourArea(contours[-1]) > 50:
                                reject_msg = "no. of contours > 2"
                                ret_flag = False
                                return copy_golgi, _, _, ret_flag, sub_list, reject_msg
                            else:
                                # got many small contours
                                sub = max(np.where(task_img > 0, task_img, np.inf).min(), min_sub)
                                continue
                    else:
                        if had_convex:
                            reject_msg = "crop had convex, and not satisfied."
                            ret_flag = False
                            return copy_golgi, _, _, ret_flag, sub_list, reject_msg
                        # the U shape
                        print("ratio larger than 1.05. {}".format(ratio))
                    break
                else:
                    # no contour in giantin channel
                    ret_flag = False
                    reject_msg = "no contour in giantin channel."
                    return copy_golgi, _, _, ret_flag, sub_list, reject_msg
            else:
                # Other channel
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                max_area = 0
                for i, contour in enumerate(contours):
                    c_x = contour[:, :, 0].reshape(-1, )
                    c_y = contour[:, :, 1].reshape(-1, )
                    if h - 1 in c_x or w - 1 in c_y or 0 in c_x or 0 in c_y:
                        if i == 0:
                            # the largest one near the edge
                            do_sub = True
                            break
                        else:
                            cv2.drawContours(task_img, contours, i, 0, -1)
                            num_contours -= 1
                            continue
                    if i == 0:
                        max_area = cv2.contourArea(contour)
                    else:
                        # clear small contours
                        contour_area = cv2.contourArea(contour)
                        if contour_area <= max_area * 0.2:
                            # clear the small contours
                            cv2.drawContours(task_img, contours, i, 0, -1)
                            num_contours -= 1
                            continue
                if do_sub:
                    sub = max(np.min(np.where(task_img > 0, task_img, np.inf)), min_sub)
                    continue
                # Having overlapping area with giantin channel.
                if have_overlapping:
                    dilated_img = cv2.dilate(task_img, np.ones((3, 3)))
                    giantin_crop = copy_golgi[:, :, giantin_channel]
                    overlap = np.multiply(dilated_img, giantin_crop).sum()
                    if overlap > 0:
                        ret_flag = True
                        break
                    else:
                        # no overlapping with giantin channel
                        ret_flag = False
                        reject_msg = "no contour in giantin channel."
                        return copy_golgi, _, _, ret_flag, sub_list, reject_msg
                else:
                    ret_flag = True
                    break

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
    return copy_golgi, giantin_mask, giantin_contour, ret_flag, sub_list, reject_msg


def cal_center_of_mass(image, contour=None):
    """
    calculate center of mass.
    :param image: shape is  (h,w)
    :param contour: contour
    :return: [(mx, my)]
    """
    if contour is not None:
        mu = cv2.moments(contour)
    else:
        shape_len = len(image.shape)
        assert shape_len == 2, "Dimension of image shape is not 2."
        mu = cv2.moments(image)
    mx = round(mu['m10'] / mu['m00'], 4)
    my = round(mu['m01'] / mu['m00'], 4)

    # shape_len = len(image.shape)
    # assert shape_len == 2, "Dimension of image shape is not 2."
    # h, w = image.shape
    # total_intensity = 0
    # Qx = 0
    # Qy = 0
    # for i in range(h):
    #     for j in range(w):
    #         intensity = int(image[i][j])
    #         total_intensity += intensity
    #         Qy += i * intensity
    #         Qx += j * intensity
    # mx = round(Qx / total_intensity, 4)
    # my = round(Qy / total_intensity, 4)
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


def cal_radial_mean_intensity(golgi_image):
    len_shape = len(golgi_image.shape)
    assert len_shape == 3, "Dimension of image shape is not 3."
    mx, my = 349.5, 349.5
    h, w, c = golgi_image.shape
    df_list = []
    for c_ in range(c):
        df = pd.DataFrame(columns=["No. pixel", "total_intensity", "mean_intensity"], index=range(0, 499))
        no_pixel = [0 for _ in range(499)]
        total_intensity = [0 for _ in range(499)]
        for i in range(h):
            for j in range(w):
                distance_to_center = np.sqrt((i - my) ** 2 + (j - mx) ** 2)
                n = math.floor(distance_to_center)
                no_pixel[n] += 1
                total_intensity[n] += golgi_image[i, j, c_]
        df["No. pixel"] = no_pixel
        df["total_intensity"] = total_intensity
        df["mean_intensity"] = df["total_intensity"] / df["No. pixel"]
        df["normalized_mean_intensity"] = df["mean_intensity"] / df["mean_intensity"].max()
        df_list.append(df)
        print("chennel {} finished".format(c_))
    return df_list
