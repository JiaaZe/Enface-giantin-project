import cv2
import numpy as np
import pandas as pd
import sys

from PyQt5.QtGui import QIntValidator, QMovie
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal as Signal, QThread, QObject
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure

from qt_ui.golgi_details_widget import Ui_Golgi_details
from image_functions import check_golgi_crop, cal_center_of_mass, cal_gyradius, normalize_total_intensity, \
    shift_make_border, cal_radial_mean_intensity


class GolgiDetailWidget(QWidget):
    save_signal = Signal(int)
    signal_backwork = Signal()

    # mode: 1 for golgi details. 2 for dispaly averaged golgi
    def __init__(self, window_name, crop_golgi=None, mode=1, giantin_mask=None, giantin_pred=None, param_dict=None):
        super().__init__()
        self.setObjectName(window_name)
        self.ui = Ui_Golgi_details()
        self.ui.setupUi(self)
        self.mode = mode

        self.radial_mean_intensity_df_list = None
        self.radius_list = None
        self.thread = None
        self.backwork = None

        self.crop_golgi = crop_golgi
        if self.mode == 1:
            self.ui.btn_save.setText("Save")
            self.giantin_mask = giantin_mask
            self.giantin_pred = giantin_pred
            self.giantin_channel = param_dict["param_giantin_channel"]
            self.blank_channel = param_dict["param_blank_channel"]
            self.overlapping = param_dict["param_giantin_overlap"]
            self.giantin_possibility_threshold = param_dict["param_giantin_threshold"]
            self.min_giantin_area = param_dict["param_giantin_area_threshold"]

            self.new_shifted_golgi = None
            self.new_crop_golgi = None
            self.new_giantin_mask = None
            self.new_giantin_pred = None

            self.ui.btn_export.setVisible(False)
            if self.giantin_mask is None:
                self.giantin_mask = self.crop_golgi[:, :, self.giantin_channel] / (
                        self.crop_golgi[:, :, self.giantin_channel] + 1) * 255
            self.show_img(self.crop_golgi, self.giantin_mask)

            # subtraction
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
            self.ui.btn_save.setText("Save plots")
            self.hide_widget_for_averaged()
            self.ui.btn_save.clicked.connect(lambda: self.save_averaged_result())
            self.ui.btn_export.clicked.connect(lambda: self.export_averaged_result())
            self.show_loading()

    def show_loading(self):
        layout = QVBoxLayout(self.ui.golgi_content_widget)
        label = QLabel(self.ui.golgi_content_widget)
        label.setStyleSheet("background: white;border: 0px")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setObjectName("label")
        layout.addWidget(label)
        movie = QMovie("../qt_ui/loading.gif")
        label.setMovie(movie)
        movie.start()

    def sub_handler(self, channel, sub_value, btn_ui):
        if sub_value == "":
            return
        btn_ui.setDisabled(True)

        sub_value = int(sub_value)
        new_crop = np.copy(self.crop_golgi)
        golgi_crop = new_crop[:, :, channel]
        h, w = golgi_crop.shape
        for i in range(h):
            for j in range(w):
                if golgi_crop[i][j] > sub_value:
                    golgi_crop[i][j] = golgi_crop[i][j] - sub_value
                else:
                    golgi_crop[i][j] = 0

        mask = golgi_crop / (golgi_crop + 1) * 255
        mask = mask.astype(np.bool_)

        self.new_crop_golgi = new_crop
        if channel == self.giantin_channel:
            title = "Giantin Mask"
        else:
            title = "C{} Mask".format(channel + 1)
        self.show_img(new_crop, mask, title)
        btn_ui.setEnabled(True)
        self.ui.btn_check.setEnabled(True)

    def check_handler(self):
        target_size = 701
        centroid = (350, 350)
        sub_list = None
        rect_size = self.new_crop_golgi.shape[0]
        for _ in range(2):
            crop_golgi = self.new_crop_golgi[:rect_size, :rect_size]
            pred = self.giantin_pred[:rect_size, :rect_size]
            golgi, mask, contour, flag, sub_list, rej_msg = check_golgi_crop(crop_golgi,
                                                                             pred,
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
                    break
            else:
                self.logger.info(rej_msg)
                break

    def get_new_data(self):
        return self.new_crop_golgi, self.new_shifted_golgi, self.new_giantin_mask, self.new_giantin_pred

    def show_img(self, crop_golgi, mask, mask_title="Giantin Mask"):
        num_channel = crop_golgi.shape[-1]
        columns = num_channel + 1
        rows = 1
        static_canvas = FigureCanvas(Figure(figsize=(2 * columns, 0.8 * rows)))
        subplot_axes = static_canvas.figure.subplots(1, (num_channel + 1))
        static_canvas.figure.tight_layout(h_pad=0, w_pad=0.5)
        # static_canvas.figure.subplots_adjust(wspace=0.4)
        font_size = 9
        for i, axes in enumerate(subplot_axes.reshape(-1)):
            for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                label.set_fontsize(font_size)

            if i == num_channel:
                # show mask
                axes.set_title(mask_title)
                img_ = axes.imshow(mask)
                cbar = static_canvas.figure.colorbar(img_, ax=axes)
            else:
                img = crop_golgi[:, :, i]
                axes.set_title(mask_title)
                img_ = axes.imshow(img)
                cbar = static_canvas.figure.colorbar(img_, ax=axes)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(font_size)
        plotLayout = self.ui.golgi_content_widget.layout()
        if plotLayout is None:
            plotLayout = QVBoxLayout()
            plotLayout.setContentsMargins(2, 2, 2, 2)
            plotLayout.addWidget(static_canvas)
            self.ui.golgi_content_widget.setLayout(plotLayout)
        else:
            cur_item = plotLayout.itemAt(0)
            cur_widget = cur_item.widget()
            if cur_widget is not None:
                plotLayout.replaceWidget(cur_widget, static_canvas)

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

    def show_averaged_w_plot(self, averaged_golgi):
        self.thread = QThread()

        self.backwork = Backwork(averaged_golgi)
        self.backwork.moveToThread(self.thread)
        self.backwork.finished_signal.connect(lambda: self.backwork_finished_handler(averaged_golgi))
        self.signal_backwork.connect(self.backwork.cal)
        self.thread.start()
        self.signal_backwork.emit()

    def backwork_finished_handler(self, crop_data):
        num_channel = crop_data.shape[-1]
        columns = num_channel + 1
        rows = 2
        static_canvas = FigureCanvas(Figure(figsize=(3 * columns, 1 * rows)))
        subplot_axes = static_canvas.figure.subplots(rows, columns)
        static_canvas.figure.tight_layout(pad=0.6)

        golgi_x_axis_labels = np.arange(0, 701, 350)
        golgi_y_axis_labels = np.arange(700, -1, -350)
        font_size = 9

        # get plot data
        self.radial_mean_intensity_df_list, self.radius_list = self.backwork.get_data()

        for j in range(columns - 1):
            # show golgi
            golgi_axes = subplot_axes[0][j]
            golgi_axes.set_title("Channel {}".format(j + 1))
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
            plot_axes.plot(self.radial_mean_intensity_df_list[j]["normalized_mean_intensity"])
            for label in (plot_axes.get_xticklabels() + plot_axes.get_yticklabels()):
                label.set_fontsize(font_size)

        subplot_axes[0][-1].set_title("Merge")
        merge_img = crop_data / np.amax(crop_data, axis=(0, 1))
        subplot_axes[0][-1].imshow(merge_img)

        subplot_axes[1][-1].plot(self.radial_mean_intensity_df_list[0]["normalized_mean_intensity"])
        subplot_axes[1][-1].plot(self.radial_mean_intensity_df_list[1]["normalized_mean_intensity"])
        subplot_axes[1][-1].plot(self.radial_mean_intensity_df_list[2]["normalized_mean_intensity"])

        for axes in subplot_axes[1]:
            axes.set_xlabel("Distance from center")
            axes.set_ylabel("Radial mean intensity")

        plotLayout = QVBoxLayout()
        plotLayout.setContentsMargins(2, 2, 2, 2)
        plotLayout.addWidget(static_canvas)
        self.ui.golgi_content_widget.setLayout(plotLayout)

    def save_averaged_result(self):
        ...

    def export_averaged_result(self):
        ...


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
    window = GolgiDetailWidget("Averaged golgi mini-stacks", mode=2)
    # window = GolgiDetailWidget("Averaged golgi mini-stacks", mode=1, crop_golgi=np.dstack([data, data, data]),
    #                            param_dict=param_dict)
    window.show()
    sys.exit(app.exec())
