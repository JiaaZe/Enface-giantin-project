import logging
import os

import cv2
import numpy as np
import pandas as pd
import sys

from PyQt5.QtGui import QIntValidator, QMovie
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QApplication, QLabel, QFileDialog
from PyQt5.QtCore import pyqtSignal as Signal, QThread, QObject
from PyQt5 import QtCore
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from utils import get_logger
from qt_ui.golgi_details_widget import Ui_Golgi_details
from image_functions import check_golgi_crop, cal_center_of_mass, cal_gyradius, normalize_total_intensity, \
    shift_make_border, cal_radial_mean_intensity


class GolgiDetailWidget(QWidget):
    save_signal = Signal(int)
    signal_backwork = Signal()

    # mode: 1 for golgi details. 2 for dispaly averaged golgi
    def __init__(self, window_name, logger: logging.Logger, crop_golgi=None,
                 mode=1, giantin_mask=None, giantin_pred=None, param_dict=None, save_directory="", channel_name=None):
        super().__init__()
        self.setWindowTitle(window_name)
        self.ui = Ui_Golgi_details()
        self.ui.setupUi(self)
        self.mode = mode
        if logger is None:
            logger = get_logger()
        self.logger = logger
        self.save_directory = save_directory
        self.channel_name = channel_name

        self.radial_mean_intensity_df_list = None
        self.radius_list = None
        self.thread = None
        self.backwork = None

        self.crop_golgi = crop_golgi
        self.ui.browser_error.setVisible(False)
        if self.mode == 1:
            self.ui.btn_save.setText("Save")
            self.giantin_mask = giantin_mask
            # original mask
            self.crop_mask = crop_golgi / (crop_golgi + 1) * 255
            self.crop_mask = self.crop_mask.astype(np.bool_)
            self.giantin_pred = giantin_pred
            self.giantin_channel = param_dict["param_giantin_channel"]
            self.blank_channel = param_dict["param_blank_channel"]
            self.overlapping = param_dict["param_giantin_overlap"]
            self.giantin_possibility_threshold = param_dict["param_giantin_threshold"]
            self.min_giantin_area = param_dict["param_giantin_area_threshold"]

            self.new_shifted_golgi = None
            # subtracted crop golgi
            self.new_crop_golgi = np.copy(crop_golgi)
            # subtracted crop mask
            self.new_crop_mask = np.copy(self.crop_mask)
            self.new_giantin_mask = None
            self.new_giantin_pred = None

            self.ui.btn_export.setVisible(False)
            self.show_golgi_details(self.crop_golgi, self.crop_mask)

            # subtraction
            if crop_golgi.shape[-1] == 2:
                self.ui.sub_value_c3.setVisible(False)
                self.ui.btn_sub_c3.setVisible(False)
                self.ui.label_c3.setVisible(False)
            self.ui.sub_value_c1.setValidator(QIntValidator())
            self.ui.sub_value_c2.setValidator(QIntValidator())
            self.ui.sub_value_c3.setValidator(QIntValidator())
            self.ui.btn_sub_c1.clicked.connect(lambda: self.sub_handler(0, self.ui.sub_value_c1.text(),
                                                                        self.ui.btn_sub_c1))
            self.ui.btn_sub_c2.clicked.connect(lambda: self.sub_handler(1, self.ui.sub_value_c2.text(),
                                                                        self.ui.btn_sub_c2))
            self.ui.btn_sub_c3.clicked.connect(lambda: self.sub_handler(2, self.ui.sub_value_c3.text(),
                                                                        self.ui.btn_sub_c3))
            self.ui.btn_check.clicked.connect(self.check_handler)

            self.ui.btn_save.clicked.connect(lambda: self.save_signal.emit(0))

            self.ui.btn_check.setDisabled(True)
            self.ui.btn_save.setDisabled(True)
        elif self.mode == 2:
            self.giantin_channel = param_dict["param_giantin_channel"]
            self.ui.btn_save.setText("Save plots")
            self.hide_widget_for_averaged()
            self.ui.btn_save.setDisabled(True)
            self.ui.btn_export.setDisabled(True)
            self.ui.btn_save.clicked.connect(lambda: self.save_averaged_result())
            self.ui.btn_export.clicked.connect(self.export_averaged_result)
            self.show_loading()

    def update_message(self, text):
        self.ui.browser_error.setVisible(True)
        self.ui.browser_error.append(text)

    def show_loading(self):
        layout = QVBoxLayout(self.ui.golgi_content_widget)
        label = QLabel(self.ui.golgi_content_widget)
        label.setStyleSheet("background: white;border: 0px")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setObjectName("label")
        layout.addWidget(label)
        movie = QMovie("./loading.gif")
        label.setMovie(movie)
        movie.start()

    def sub_handler(self, channel, sub_value, btn_ui):
        try:
            if sub_value == "":
                return
            btn_ui.setDisabled(True)

            sub_value = int(sub_value)
            # get original crop golgi in certain channel
            golgi_crop = np.copy(self.crop_golgi[:, :, channel])
            # golgi_crop = new_crop[:, :, channel]
            golgi_crop = np.where(golgi_crop > sub_value, golgi_crop - sub_value, 0)
            # for i in range(h):
            #     for j in range(w):
            #         if golgi_crop[i][j] > sub_value:
            #             golgi_crop[i][j] = golgi_crop[i][j] - sub_value
            #         else:
            #             golgi_crop[i][j] = 0

            # calcualte mask
            mask = golgi_crop / (golgi_crop + 1) * 255
            mask = mask.astype(np.bool_)

            self.new_crop_golgi[:, :, channel] = golgi_crop
            self.new_crop_mask[:, :, channel] = mask
            self.show_golgi_details(self.new_crop_golgi, self.new_crop_mask)
            btn_ui.setEnabled(True)
            self.ui.btn_check.setEnabled(True)
        except Exception as e:
            self.logger.error("Error when do subtraction:{}".format(e), exc_info=True)

    def check_handler(self):
        try:
            target_size = 701
            centroid = (350, 350)
            sub_list = None
            rect_size = self.new_crop_golgi.shape[0]
            for _ in range(2):
                crop_golgi = self.new_crop_golgi[:rect_size, :rect_size]
                pred = self.giantin_pred[:rect_size, :rect_size]
                golgi, mask, contour, flag, sub_list, rej_msg = check_golgi_crop(crop_golgi,
                                                                                 pred,
                                                                                 edge_contour=[0, 0, 0, 0],
                                                                                 giantin_channel=self.giantin_channel,
                                                                                 blank_channel=self.blank_channel,
                                                                                 sub_list=sub_list,
                                                                                 min_giantin_area=self.min_giantin_area,
                                                                                 giantin_possibility_threshold=
                                                                                 self.giantin_possibility_threshold,
                                                                                 have_overlapping=self.overlapping)
                if flag:
                    crop_giantin = golgi[:, :, self.giantin_channel]
                    mx, my = cal_center_of_mass(crop_giantin, contour)
                    gyradius = cal_gyradius(crop_giantin, mx, my)
                    if rect_size > gyradius * target_size / 100:
                        rect_size = int(gyradius * target_size / 100)
                        print("new rect_size: {}".format(rect_size))
                        continue
                    else:
                        new_size = [int(size * 100 / gyradius) for size in crop_giantin.shape]
                        resized_golgi = cv2.resize(golgi, new_size, interpolation=cv2.INTER_LINEAR)
                        normalized_golgi = normalize_total_intensity(resized_golgi, target_total_intensity=200000000)
                        shifted_golgi = shift_make_border(normalized_golgi, giantin_channel=self.giantin_channel,
                                                          border_size=(target_size, target_size),
                                                          center_coord=centroid, shift_to_imageJ=True)
                        self.new_shifted_golgi = shifted_golgi
                        self.new_crop_golgi = golgi
                        self.new_giantin_mask = mask
                        self.new_giantin_pred = pred
                        self.ui.btn_save.setEnabled(True)

                        # show result after check
                        self.show_golgi_details(self.new_crop_golgi, None)
                        break
                else:
                    self.logger.info(rej_msg)
                    break
        except Exception as e:
            err_msg = "Error when check single golgi mini-stacks:{}".format(e)
            self.logger.error(err_msg, exc_info=True)
            self.update_message(err_msg)

    def get_new_data(self):
        return self.new_crop_golgi, self.new_shifted_golgi, self.new_giantin_mask, self.new_giantin_pred

    def show_golgi_details(self, crop_golgi, masks):
        try:
            if masks is None:
                masks = crop_golgi / (crop_golgi + 1) * 255
                masks = masks.astype(np.bool_)
            num_channel = crop_golgi.shape[-1]
            columns = num_channel
            rows = 2
            static_canvas = FigureCanvas(Figure(figsize=(2 * columns, 0.8 * rows)))
            subplot_axes = static_canvas.figure.subplots(rows, columns)
            static_canvas.figure.tight_layout(pad=0.6)
            # static_canvas.figure.subplots_adjust(wspace=0.4)
            font_size = 9
            for j in range(num_channel):
                for i in range(2):
                    if i == 0:
                        img = crop_golgi[:, :, j]
                        # title = "Channel"
                        title = self.channel_name[j]
                        cmap = None
                    else:
                        img = masks[:, :, j]
                        # title = "Mask"
                        title = self.channel_name[j] + " mask"
                        cmap = "binary_r"
                    axes = subplot_axes[i][j]
                    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                        label.set_fontsize(font_size)

                    img_ = axes.imshow(img, cmap=cmap)
                    cbar = static_canvas.figure.colorbar(img_, ax=axes)
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontsize(font_size)
                    if i == 1:
                        # mask color bar label -> [0,1]
                        # cbar.ax.set_yticklabels([0, 1])
                        cbar.set_ticks([0, 1])
                    # if j == self.giantin_channel:
                    #     axes.set_title("Giantin " + title)
                    # else:
                    #     axes.set_title("C{} ".format(j + 1) + title)
                    axes.set_title(title, fontdict={'fontsize': font_size})
            self.plot_widget(static_canvas)
        except Exception as e:
            err_msg = "Error when show golgi mini-stacks details:{}".format(e)
            self.logger.error(err_msg, exc_info=True)
            self.update_message(err_msg)

    # for averaged data
    def hide_widget_for_averaged(self):
        self.ui.btn_sub_c1.setVisible(False)
        self.ui.btn_sub_c2.setVisible(False)
        self.ui.btn_sub_c3.setVisible(False)
        self.ui.sub_value_c1.setVisible(False)
        self.ui.sub_value_c2.setVisible(False)
        self.ui.sub_value_c3.setVisible(False)
        self.ui.label_c1.setVisible(False)
        self.ui.label_c2.setVisible(False)
        self.ui.label_c3.setVisible(False)
        self.ui.btn_check.setVisible(False)

    def plot_widget(self, canvas):
        plotLayout = self.ui.golgi_content_widget.layout()
        if plotLayout is None:
            plotLayout = QVBoxLayout()
            plotLayout.setContentsMargins(2, 2, 2, 2)
            plotLayout.addWidget(canvas)
            self.ui.golgi_content_widget.setLayout(plotLayout)
        else:
            cur_item = plotLayout.itemAt(0)
            if cur_item is not None:
                cur_widget = cur_item.widget()
                if cur_widget is not None:
                    plotLayout.replaceWidget(cur_widget, canvas)

    def show_averaged_w_plot(self, averaged_golgi, num_ministacks):
        self.thread = QThread()

        self.backwork = Backwork(averaged_golgi)
        self.backwork.moveToThread(self.thread)
        self.backwork.finished_signal.connect(lambda: self.backwork_finished_handler(averaged_golgi, num_ministacks))
        self.signal_backwork.connect(self.backwork.cal)
        self.thread.start()
        self.signal_backwork.emit()

    def backwork_finished_handler(self, crop_data, num_ministacks):
        color_map = ["red", "green", "blue"]
        empty_channel_list = []
        try:
            num_channel = crop_data.shape[-1]
            columns = num_channel + 1
            rows = 2
            static_canvas = FigureCanvas(Figure(figsize=(2 * columns, 1 * rows)))
            subplot_axes = static_canvas.figure.subplots(rows, columns)
            for i, name in enumerate(self.channel_name):
                if name == "":
                    empty_channel_list.append(i)
                    subplot_axes[0][i].axis('off')
                    subplot_axes[1][i].axis('off')
            static_canvas.figure.tight_layout(pad=0.6)

            golgi_x_axis_labels = np.arange(0, 701, 350)
            golgi_y_axis_labels = np.arange(700, -1, -350)
            font_size = 10

            # get plot data
            self.radial_mean_intensity_df_list, self.radius_list = self.backwork.get_data()
            giantin_radius = self.radius_list[self.giantin_channel]
            normalized_radius = [i / giantin_radius for i in self.radius_list]
            for i, radius in enumerate(self.radius_list):
                self.radial_mean_intensity_df_list[i]["normalized_radius"] = self.radial_mean_intensity_df_list[
                                                                                 i].index / giantin_radius

            for j in range(num_channel):
                # hide empty channel name
                if self.channel_name[j] == "":
                    continue
                # show golgi
                golgi_axes = subplot_axes[0][j]
                # golgi_axes.set_title("Channel {}".format(j + 1))
                golgi_axes.set_title(self.channel_name[j], fontdict={'fontsize': font_size})
                img_ = golgi_axes.imshow(crop_data[:, :, j])
                cbar = static_canvas.figure.colorbar(img_, ax=golgi_axes)

                golgi_axes.set_xlim(np.min(golgi_x_axis_labels), np.max(golgi_x_axis_labels))
                golgi_axes.set_ylim(np.max(golgi_y_axis_labels), np.min(golgi_y_axis_labels))
                golgi_axes.set_xticks(golgi_x_axis_labels)
                golgi_axes.set_yticks(golgi_y_axis_labels)

                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(font_size)

                for label in (golgi_axes.get_xticklabels() + golgi_axes.get_yticklabels()):
                    label.set_fontsize(font_size)

                # show plot
                plot_axes = subplot_axes[1][j]
                plot_axes.plot(self.radial_mean_intensity_df_list[j]["normalized_radius"],
                               self.radial_mean_intensity_df_list[j]["normalized_mean_intensity"], c=color_map[j])
                plot_axes.set_title("normalized radius={:.2f}".format(normalized_radius[j]),
                                    fontdict={'fontsize': font_size})
                for label in (plot_axes.get_xticklabels() + plot_axes.get_yticklabels()):
                    label.set_fontsize(font_size)

            subplot_axes[0][-1].set_title("merge (n={})".format(num_ministacks),
                                          fontdict={'fontsize': font_size})
            merge_img = crop_data / np.amax(crop_data, axis=(0, 1))
            if len(empty_channel_list) > 0:
                empty_shape = merge_img.shape[:2]
                empty_img = np.zeros(shape=empty_shape)
                for empty_channel in empty_channel_list:
                    merge_img[:, :, empty_channel] = empty_img
            if num_channel < 3:
                empty_shape = merge_img.shape[:2] + (3 - num_channel,)
                empty_img = np.zeros(shape=empty_shape)
                merge_img = np.dstack([merge_img, empty_img])
            subplot_axes[0][-1].imshow(merge_img)
            handles = [Rectangle((0, 0), 0, 0, label=i) for i in self.channel_name]

            leg = subplot_axes[0][-1].legend(handles=handles, handlelength=0, handletextpad=0, loc='upper left',
                                             bbox_to_anchor=(0.8, 1), fontsize='x-small')
            for i, text in enumerate(leg.get_texts()):
                text.set_color(color_map[i])

            for k in range(num_channel):
                # hide empty channel name
                if self.channel_name[k] == "":
                    continue
                subplot_axes[1][-1].plot(self.radial_mean_intensity_df_list[k]["normalized_radius"],
                                         self.radial_mean_intensity_df_list[k]["normalized_mean_intensity"],
                                         c=color_map[k], label=self.channel_name[k])
            subplot_axes[1][-1].legend(labelcolor='linecolor', fontsize='small')

            for axes in subplot_axes[1]:
                axes.set_xlabel("Distance from center")

            subplot_axes[1][0].set_ylabel("Radial mean intensity")

            self.plot_widget(static_canvas)
            self.ui.btn_save.setEnabled(True)
            self.ui.btn_export.setEnabled(True)
        except Exception as e:
            err_msg = "Error when plot the averaged plot:{}".format(e)
            self.logger.error(err_msg, exc_info=True)
            self.update_message(err_msg)

    def save_averaged_result(self):
        try:
            plotLayout = self.ui.golgi_content_widget.layout()
            canvas = plotLayout.itemAt(0).widget()
            save_path, save_type = QFileDialog.getSaveFileName(self, "Save File",
                                                               directory=os.path.join(self.save_directory,
                                                                                      "averaged_plot"),
                                                               filter='pdf (*.pdf);; png (*.png);;jpg (*.jpg)')
            if save_type == "" and save_path == "":
                return
            if save_type == "pdf (*.pdf)":
                with PdfPages(save_path) as pdf:
                    pdf.savefig(canvas.figure, dpi=120)
            else:
                canvas.figure.savefig(save_path)
        except Exception as e:
            err_msg = "Error when save averaged plot: {}".format(e)
            self.logger.error(err_msg, exc_info=True)
            self.update_message(err_msg)
        else:
            os.startfile(os.path.split(save_path)[0])
            success_msg = "Save averaged plot sucessfully."
            self.logger.info(success_msg)
            self.update_message(success_msg)

    def export_averaged_result(self):
        try:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save File",
                                                       directory=os.path.join(self.save_directory,
                                                                              "radial mean intensity"),
                                                       filter='xlsx (*.xlsx)')
            if save_path == "":
                return
            excel_writer = pd.ExcelWriter(save_path)
            for i, df in enumerate(self.radial_mean_intensity_df_list):
                df.to_excel(excel_writer, sheet_name="C{}".format(i + 1))
            excel_writer.save()
        except Exception as e:
            err_msg = "Error when export averaged results: {}".format(e)
            self.logger.error(err_msg, exc_info=True)
            self.update_message(err_msg)
        else:
            os.startfile(os.path.split(save_path)[0])
            success_msg = "Export averaged results sucessfully."
            self.logger.info(success_msg)
            self.update_message(success_msg)


class Backwork(QObject):
    finished_signal = Signal()

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.radial_mean_intensity_df_list = []
        self.radius_list = []

    def cal(self):
        self.radial_mean_intensity_df_list, self.radius_list = cal_radial_mean_intensity(self.data)
        self.finished_signal.emit()

    def get_data(self):
        return self.radial_mean_intensity_df_list, self.radius_list


if __name__ == '__main__':
    param_dict = {"param_giantin_area_threshold": 150, "param_giantin_threshold": 0.6, "param_giantin_overlap": True,
                  "param_blank_channel": -1, "param_giantin_channel": 0}
    data = pd.read_csv("../try/try.csv")
    app = QApplication(sys.argv)
    window = GolgiDetailWidget("Averaged golgi mini-stacks", None, mode=2, param_dict=param_dict)
    window.show_averaged_w_plot(np.dstack([data, data]))
    # window = GolgiDetailWidget("Golgi Details", logger=None, mode=1,
    #                            crop_golgi=np.dstack([data, data, data]),
    #                            param_dict=param_dict)
    window.show()
    sys.exit(app.exec())
