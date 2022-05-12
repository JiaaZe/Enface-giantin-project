import logging
import os.path

from tifffile import tifffile

from PyQt5.QtCore import QObject, pyqtSignal as Signal
from tensorflow.python.keras.models import load_model

from metrics import *
from image_functions import *

# valDice0.7042_valMeanIoU0.5532.h5
model_path = "./model/model.h5"


class Progress(QObject):
    append_text = Signal(str)
    update_progress = Signal(str)
    pipeline_finished = Signal(int)
    pipeline_error = Signal(int)

    def __init__(self, model, logger: logging.Logger, image_path_list,
                 param_pixel_threshold, param_giantin_threshold, param_giantin_area_threshold,
                 param_giantin_roi_size, param_giantin_channel, param_blank_channel, param_giantin_overlap,
                 pred_data, golgi_images, pred_flag=True):
        super().__init__()
        self.logger = logger
        self.image_path_list = image_path_list
        self.param_pixel_threshold = param_pixel_threshold
        self.param_giantin_threshold = param_giantin_threshold
        self.param_giantin_area_threshold = param_giantin_area_threshold
        self.param_giantin_roi_size = param_giantin_roi_size
        self.param_giantin_channel = param_giantin_channel
        self.param_blank_channel = param_blank_channel
        self.param_giantin_overlap = param_giantin_overlap
        self.pred_flag = pred_flag
        self.pred_data = pred_data
        self.golgi_images = golgi_images
        self.crop_golgi_list = []
        self.shifted_crop_golgi_list = []
        self.giantin_mask_list = []
        self.giantin_pred_crop_list = []

        self.model = None
        if model is not None:
            self.model = model
        else:
            if not os.path.exists(model_path):
                raise Exception("No such model file:{}".format(model_path))

    def pipeline(self):
        try:
            if self.pred_flag:
                # read images
                tif_path_list = []
                golgi_image_list = []
                giantin_image_list = []
                for path in self.image_path_list:
                    if os.path.isdir(path):
                        for curDir, dirs, files in os.walk(path):
                            for file in files:
                                if file.endswith(".tif"):
                                    tif_path = os.path.join(curDir, file)
                                    tif_path_list.append(tif_path)
                                    golgi_image = tifffile.imread(tif_path)
                                    golgi_image_list.append(golgi_image)
                                    giantin_image_list.append(golgi_image[self.param_giantin_channel])
                    elif path.endswith(".tif"):
                        golgi_image = tifffile.imread(path)
                        tif_path_list.append(path)
                        golgi_image_list.append(golgi_image)
                        giantin_image_list.append(golgi_image[self.param_giantin_channel])
                # print(tif_path_list)
                num_golgi_images = len(tif_path_list)
                self.logger.info("Read {} golgi images sucessfully.".format(num_golgi_images))
                self.append_text.emit("Read {} golgi images sucessfully.".format(num_golgi_images))

                # preprocessing images to model input
                padded_giantin_list, patches_giantin_list = padding_image(giantin_image_list, do_patchify=True,
                                                                          clear_edge_roi=False, patch_size=(256, 256),
                                                                          patch_step=206)
                model_input = make_model_input(patches_giantin_list,
                                               do_norm=True,
                                               data_shape=(-1, 256, 256, 1))
                self.logger.info("Preprocess images sucessfully.")
                self.append_text.emit("Preprocess images sucessfully.")

                # run model
                if self.model is None:
                    self.model = load_model(model_path, compile=False)
                    self.model.compile(loss=bce_dice_loss,
                                       metrics=["binary_crossentropy", mean_iou, dice_coef])
                model_pred = []
                for j, image_ in enumerate(model_input):

                    finished = "==" * (j + 1)
                    left = ".." * (num_golgi_images - j - 1)
                    progress_text = "Predicting giantin image: {}/{} [{}>{}] ".format(j + 1, num_golgi_images, finished,
                                                                                      left)
                    if j == 0:
                        self.append_text.emit(progress_text)
                    else:
                        self.update_progress.emit(progress_text)

                    pred_ = self.model.predict(image_, verbose=1)
                    model_pred.append(pred_)

                # convert model output to original shape
                pred_mask, pred_mask_patches = pred_to_mask(model_pred)
                self.pred_data = unpadding_image(pred_mask, giantin_image_list)
                self.golgi_images = golgi_image_list

                self.logger.info("Prediction finished.")
                self.append_text.emit("Prediction finished.")
            else:
                self.logger.info("Reuse the data from last time.")
                self.append_text.emit("Reuse the data from last time.")

            # analysis golgi
            self.logger.info("Analyzing predicted giantin masks.")
            num_pred_data = len(self.pred_data)
            for i, pred_mask in enumerate(self.pred_data):
                finished = "==" * (i + 1)
                left = ".." * (num_pred_data - i - 1)
                progress_text = "Analyzing predicted giantin masks: {}/{} [{}>{}] ".format(i + 1, num_pred_data,
                                                                                           finished,
                                                                                           left)
                if i == 0:
                    self.append_text.emit(progress_text)
                else:
                    self.update_progress.emit(progress_text)
                selected_golgi_list, shifted_golgi_list, giantin_mask_list, giantin_pred_list = self.analysis_golgi(
                    self.golgi_images[i],
                    pred_mask)
                # crop golgi original image
                self.crop_golgi_list.extend(selected_golgi_list)
                # shifted and resized crop golgi image
                self.shifted_crop_golgi_list.extend(shifted_golgi_list)
                # crop giantin mask
                self.giantin_mask_list.extend(giantin_mask_list)
                # crop giantin pred
                self.giantin_pred_crop_list.extend(giantin_pred_list)

            self.logger.info("Analyzing predicted giantin masks finished.")
            self.append_text.emit("Analyzing predicted giantin masks finished.")

        except Exception as e:
            self.logger.error("Error: {}".format(e), exc_info=True)
            self.append_text.emit("Error: {}".format(e))
            self.pred_flag = True
            self.pipeline_error.emit(0)
        else:
            self.pipeline_finished.emit(0)

    def analysis_golgi(self, golgi_image, pred_mask):
        golgi_image = golgi_image.transpose((1, 2, 0))
        thres_mask = np.array(pred_mask > self.param_pixel_threshold, dtype=np.uint8)
        contours, _ = cv2.findContours(thres_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        target_size = 701
        centroid = (350, 350)
        selected_golgi_list = []
        shifted_golgi_list = []
        giantin_mask_list = []
        giantin_pred_list = []
        for i, contour in enumerate(contours):
            sub_list = None
            rect_size = self.param_giantin_roi_size
            for _ in range(2):
                crop_golgi, giantin_contour, crop_pred, giantin_mask, flag, sub_list, rej_msg = check_contours(
                    golgi_image,
                    pred_mask, contour,
                    giantin_channel=self.param_giantin_channel,
                    blank_channel=self.param_blank_channel,
                    min_giantin_area=self.param_giantin_area_threshold,
                    sub_list=sub_list,
                    giantin_possibility_threshold=self.param_giantin_threshold,
                    rect_size=rect_size,
                    show_plt=False,
                    overlapping=self.param_giantin_overlap)
                if flag:
                    crop_giantin = crop_golgi[:, :, self.param_giantin_channel]
                    mx, my = cal_center_of_mass(crop_giantin, giantin_contour)
                    gyradius = cal_gyradius(crop_giantin, mx, my)
                    if rect_size > gyradius * target_size / 100:
                        rect_size = int(gyradius * target_size / 100)
                        # print("new rect_size: {}".format(rect_size))
                        continue
                    else:
                        new_size = [int(size * 100 / gyradius) for size in crop_giantin.shape]
                        resized_golgi = cv2.resize(crop_golgi, new_size, interpolation=cv2.INTER_LINEAR)
                        normalized_golgi = normalize_total_intensity(resized_golgi, target_total_intensity=200000000)
                        shifted_golgi = shift_make_border(normalized_golgi, giantin_channel=self.param_giantin_channel,
                                                          border_size=(target_size, target_size),
                                                          center_coord=centroid, shift_to_imageJ=True)
                        selected_golgi_list.append(crop_golgi)
                        shifted_golgi_list.append(shifted_golgi)
                        giantin_mask_list.append(giantin_mask)
                        giantin_pred_list.append(crop_pred)
                        break
                else:
                    self.logger.info(rej_msg)
                    break
        return selected_golgi_list, shifted_golgi_list, giantin_mask_list, giantin_pred_list

    def get_model(self):
        return self.model

    def get_pred_data(self):
        return self.pred_data

    def get_golgi_images(self):
        return self.golgi_images

    def get_crop_golgi(self):
        return self.crop_golgi_list, self.shifted_crop_golgi_list, self.giantin_mask_list, self.giantin_pred_crop_list

    def get_pred_flag(self):
        return self.pred_flag
