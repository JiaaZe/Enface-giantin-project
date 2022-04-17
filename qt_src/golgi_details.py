import cv2
import numpy as np
import pandas as pd
import sys

from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QApplication
from PyQt5.QtCore import pyqtSignal as Signal
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure

from qt_ui.golgi_details_widget import Ui_Golgi_details
from image_functions import check_golgi_crop, cal_center_of_mass, cal_gyradius, normalize_total_intensity, \
    shift_make_border


class GolgiDetailWidget(QWidget):
    save_signal = Signal(int)

    def __init__(self, crop_golgi, giantin_mask, giantin_pred, param_dict):
        super().__init__()
        self.ui = Ui_Golgi_details()
        self.ui.setupUi(self)

        self.crop_golgi = crop_golgi
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

        self.show_img(self.crop_golgi, self.giantin_mask)

        # subtraction
        self.ui.sub_value.setValidator(QIntValidator())
        self.ui.btn_sub.clicked.connect(self.sub_handler)
        self.ui.btn_check.clicked.connect(self.check_handler)
        self.ui.btn_save.clicked.connect(lambda: self.save_signal.emit(0))

        self.ui.btn_check.setDisabled(True)
        self.ui.btn_save.setDisabled(True)

    def sub_handler(self):
        if self.ui.sub_value.text() == "":
            return
        self.ui.btn_sub.setDisabled(True)

        sub_value = int(self.ui.sub_value.text())
        new_crop = np.copy(self.crop_golgi)
        giantin_crop = new_crop[:, :, self.giantin_channel]
        h, w = giantin_crop.shape
        for i in range(h):
            for j in range(w):
                if giantin_crop[i][j] > sub_value:
                    giantin_crop[i][j] = giantin_crop[i][j] - sub_value
                else:
                    giantin_crop[i][j] = 0

        giantin_mask = giantin_crop / (giantin_crop + 1) * 255
        giantin_mask = giantin_mask.astype(np.bool_)

        self.new_crop_golgi = new_crop

        self.show_img(new_crop, giantin_mask)
        self.ui.btn_sub.setEnabled(True)
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

    def show_img(self, crop_golgi, giantin_mask):
        num_channel = crop_golgi.shape[-1]
        columns = num_channel + 1
        rows = 1
        static_canvas = FigureCanvas(Figure(figsize=(2 * columns, 0.8 * rows)))
        subplot_axes = static_canvas.figure.subplots(1, (num_channel + 1))
        static_canvas.figure.tight_layout()
        static_canvas.figure.subplots_adjust(wspace=0.4)
        font_size = 9
        for i, axes in enumerate(subplot_axes.reshape(-1)):
            for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                label.set_fontsize(font_size)

            if i == num_channel:
                # show mask
                axes.set_title("Giantin mask")
                img_ = axes.imshow(giantin_mask)
                cbar = static_canvas.figure.colorbar(img_, ax=axes)
            else:
                img = crop_golgi[:, :, i]
                axes.set_title("Channel {}".format(i + 1))
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

# if __name__ == '__main__':
#     data = pd.read_csv("../try/try.csv")
#     app = QApplication(sys.argv)
#     window = GolgiDetailWidget(np.array([data, data, data]).transpose((1, 2, 0)), data, giantin_channel=0)
#     window.show()
#     sys.exit(app.exec())
