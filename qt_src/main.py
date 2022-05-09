import configparser
import os.path
import sys
import re

import numpy as np
from PyQt5.QtCore import QRegularExpression, QThread, pyqtSignal as Signal
from PyQt5.QtGui import QRegularExpressionValidator, QIntValidator
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from processing import Progress
from utils import *

from qt_ui.mainUI import Ui_MainWindow
from golgi_details import GolgiDetailWidget
from dialog_save import DialogSave

config_file = "config.ini"


class MainWindow(QMainWindow):
    start_backgroung_work = Signal()
    last_path_str = ""
    last_giantin_channel = ""

    def __init__(self):
        super().__init__()

        self.logger = get_logger()
        self.logger.info("\n==============================START==============================")

        self.cfg = None
        self.param_dict = {}

        self.thread = None
        self.progress = None
        self.model = None
        self.pred_data = None
        self.golgi_images = None
        self.pred_flag = True
        self.save_directory = ""
        self.exp_name = ""

        self.setWindowTitle("Enface Average Tool")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.tabWidget.setCurrentIndex(0)

        # tab 1
        # image folder path list view
        self.listview_image_path = ListBoxWidget(self.ui.image_group)
        self.listview_image_path.setDragEnabled(True)
        self.listview_image_path.setObjectName("listview_image_path")
        self.listview_image_path.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.ui.horizontalLayout_2.insertWidget(1, self.listview_image_path)

        self.ui.btn_image_clear.clicked.connect(lambda: self.listview_image_path.clear())
        self.ui.btn_image_remove.clicked.connect(lambda: self.listview_image_path.remove_items())

        def btn_browse_handler():
            path_list = open_file_dialog(mode=1)
            self.listview_image_path.addItems(path_list)

        self.ui.btn_image_browse.clicked.connect(lambda: btn_browse_handler())

        try:
            self.load_param_from_config()
        except Exception as e:
            err_msg = "Error:{} when load parameters from {}. Using default parameters.".format(e, config_file)
            self.update_message(err_msg)
            self.logger.error(err_msg, exc_info=True)
            self.param_default()

        # param line edit RegExp
        float_re = QRegularExpression("([0-9]*\.?[0-9]+|[0-9]+\.?[0-9]*)$")
        self.ui.param_pixel_threshold.setValidator(QRegularExpressionValidator(float_re))
        self.ui.param_giantin_threshold.setValidator(QRegularExpressionValidator(float_re))
        self.ui.param_giantin_area_threshold.setValidator(QRegularExpressionValidator(float_re))
        self.ui.param_giantin_roi_size.setValidator(QIntValidator())
        self.ui.param_giantin_channel.setValidator(QIntValidator(1, 10))
        self.ui.param_blank_channel.setValidator(QIntValidator(-1, 10))

        self.ui.btn_default_param.clicked.connect(lambda: self.param_default())

        # channel name
        self.ui.btn_extract_channel_name.clicked.connect(lambda: self.extract_file_name())

        # start
        self.ui.btn_start.clicked.connect(lambda: self.start())

        # tab 2
        self.scroll_golgi_content = None
        self.axes_id = None

        self.axes_dict = {}
        self.selected_list = []
        self.crop_golgi_list = []
        self.shifted_crop_golgi_list = []
        self.giantin_mask_list = []
        self.giantin_pred_list = []

        self.popup_golgi_widget = None
        self.popup_averaged = None
        self.save_golgi_dialog = None

        self.ui.btn_show_avergaed.setDisabled(True)
        self.ui.btn_show_avergaed.clicked.connect(lambda: self.show_averaged())
        self.ui.btn_save.clicked.connect(lambda: self.save_golgi_stacks())

    def extract_file_name(self, first_path=None):
        if first_path is None:
            first_item = self.listview_image_path.item(0)
            if first_item is None:
                return
            first_path = first_item.text()
        file_name = os.path.split(first_path)[1]
        if not os.path.isfile(first_path):
            for _, _, files in os.walk(first_path):
                for file in files:
                    if file.endswith(".tif"):
                        file_name = file
                        file_name_split_list = re.split('[-_]', file_name)
                        self.ui.comboBox_c1.clear()
                        self.ui.comboBox_c1.addItems(file_name_split_list)
                        self.ui.comboBox_c2.clear()
                        self.ui.comboBox_c2.addItems(file_name_split_list)
                        self.ui.comboBox_c3.clear()
                        self.ui.comboBox_c3.addItems(file_name_split_list)
                        return
        else:
            file_name_split_list = re.split('[-_]', file_name)
            self.ui.comboBox_c1.clear()
            self.ui.comboBox_c1.addItems(file_name_split_list)
            self.ui.comboBox_c2.clear()
            self.ui.comboBox_c2.addItems(file_name_split_list)
            self.ui.comboBox_c3.clear()
            self.ui.comboBox_c3.addItems(file_name_split_list)
            return

    def get_cur_channel_name(self):
        c1_name = self.ui.comboBox_c1.currentText()
        c2_name = self.ui.comboBox_c2.currentText()
        c3_name = self.ui.comboBox_c3.currentText()
        return [c1_name, c2_name, c3_name]

    def param_default(self):
        self.ui.param_pixel_threshold.setText("0.5")
        self.ui.param_giantin_threshold.setText("0.6")
        self.ui.param_giantin_area_threshold.setText("150")
        self.ui.param_giantin_roi_size.setText("50")
        self.ui.param_giantin_channel.setText("1")
        self.ui.param_blank_channel.setText("-1")
        self.ui.param_giantin_overlap.setChecked(True)

    def load_param_from_config(self):
        self.cfg = configparser.ConfigParser()
        if os.path.exists(config_file):
            self.cfg.read(config_file)
            param_pixel_threshold = self.cfg.getfloat("params", "param_pixel_threshold")
            self.param_dict["param_pixel_threshold"] = param_pixel_threshold
            self.ui.param_pixel_threshold.setText(str(param_pixel_threshold))

            param_giantin_threshold = self.cfg.getfloat("params", "param_giantin_threshold")
            self.param_dict["param_pixel_threshold"] = param_giantin_threshold
            self.ui.param_giantin_threshold.setText(str(param_giantin_threshold))

            param_giantin_area_threshold = self.cfg.getfloat("params", "param_giantin_area_threshold")
            self.param_dict["param_pixel_threshold"] = param_giantin_area_threshold
            self.ui.param_giantin_area_threshold.setText(str(param_giantin_area_threshold))

            param_giantin_roi_size = self.cfg.getint("params", "param_giantin_roi_size")
            self.param_dict["param_pixel_threshold"] = param_giantin_roi_size
            self.ui.param_giantin_roi_size.setText(str(param_giantin_roi_size))

            param_giantin_channel = self.cfg.getint("params", "param_giantin_channel")
            self.param_dict["param_giantin_channel"] = param_giantin_channel - 1
            self.ui.param_giantin_channel.setText(str(param_giantin_channel))

            param_blank_channel = self.cfg.getint("params", "param_blank_channel")
            self.param_dict["param_blank_channel"] = param_blank_channel - 1
            self.ui.param_blank_channel.setText(str(param_blank_channel))

            param_giantin_overlap = self.cfg.getboolean("params", "param_giantin_overlap")
            self.param_dict["param_giantin_overlap"] = param_giantin_overlap
            self.ui.param_giantin_overlap.setChecked(param_giantin_overlap)

            path_list = self.cfg.get("path", "path").split(";")
            self.param_dict["path_list"] = path_list
            self.listview_image_path.addItems(path_list)
            self.extract_file_name(path_list[0])

        else:
            self.cfg['path'] = {}
            self.cfg['params'] = {}

    def get_write_param(self):
        err_msg = ""
        path_list = self.listview_image_path.get_all()
        if len(path_list) > 0:
            path_str = ";".join(path_list)
            # to control whether predict again.
            self.pred_flag = not (path_str == self.last_path_str)
            self.cfg['path']['path'] = path_str
            self.param_dict["path_list"] = path_list
            if os.path.isfile(path_list[0]):
                self.exp_name = os.path.split(path_list[0])[1].split(".")[0]
                self.save_directory = os.path.split(path_list[0])[0]
            else:
                self.exp_name = os.path.split(path_list[0])[1]
                self.save_directory = path_list[0]
        else:
            path_str = ""
            err_msg += "{} is empty.\n".format(self.ui.label_image_folder.text())

        param_pixel_threshold = self.ui.param_pixel_threshold.text()
        if len(param_pixel_threshold) > 0:
            self.cfg['params']['param_pixel_threshold'] = param_pixel_threshold
            self.param_dict["param_pixel_threshold"] = float(param_pixel_threshold)
        else:
            err_msg += "{} is empty.\n".format(self.ui.label_pixel_threshold.text())

        param_giantin_threshold = self.ui.param_giantin_threshold.text()
        if len(param_giantin_threshold) > 0:
            self.cfg['params']['param_giantin_threshold'] = param_giantin_threshold
            self.param_dict["param_giantin_threshold"] = float(param_giantin_threshold)
        else:
            err_msg += "{} is empty.\n".format(self.ui.label_giantin_threshold.text())

        param_giantin_area_threshold = self.ui.param_giantin_area_threshold.text()
        if len(param_giantin_area_threshold) > 0:
            self.cfg['params']['param_giantin_area_threshold'] = param_giantin_area_threshold
            self.param_dict["param_giantin_area_threshold"] = float(param_giantin_area_threshold)
        else:
            err_msg += "{} is empty.\n".format(self.ui.label_giantin_threshold.text())

        param_giantin_roi_size = self.ui.param_giantin_roi_size.text()
        if len(param_giantin_roi_size) > 0:
            self.cfg['params']['param_giantin_roi_size'] = param_giantin_roi_size
            self.param_dict["param_giantin_roi_size"] = int(param_giantin_roi_size)
        else:
            err_msg += "{} is empty.\n".format(self.ui.label_giantin_size.text())

        param_giantin_channel = self.ui.param_giantin_channel.text()
        if len(param_giantin_channel) > 0:
            self.pred_flag = self.pred_flag or not (param_giantin_channel == self.last_giantin_channel)
            self.cfg['params']['param_giantin_channel'] = param_giantin_channel
            self.param_dict["param_giantin_channel"] = int(param_giantin_channel) - 1
        else:
            err_msg += "{} is empty.\n".format(self.ui.label_giantin_channel.text())

        param_blank_channel = self.ui.param_blank_channel.text()
        if len(param_blank_channel) > 0:
            self.cfg['params']['param_blank_channel'] = param_blank_channel
            self.param_dict["param_blank_channel"] = int(param_blank_channel) - 1
        else:
            err_msg += "{} is empty.\n".format(self.ui.label_blank_channel.text())

        param_giantin_overlap = self.ui.param_giantin_overlap.isChecked()
        self.cfg['params']['param_giantin_overlap'] = str(param_giantin_overlap)
        self.param_dict["param_giantin_overlap"] = param_giantin_overlap

        if len(err_msg) > 0:
            err_msg += "Please check all parameters and start again."
            self.ui.progress_text.setText(err_msg)
            return False
        else:
            self.last_path_str = path_str
            self.last_giantin_channel = param_giantin_channel
            with open(config_file, 'w') as configfile:
                self.cfg.write(configfile)
            return True

    def update_message(self, text):
        self.ui.progress_text.append(text)

    def update_process(self, text):
        cur_text = self.ui.progress_text.toPlainText()
        cur_text_list = cur_text.split("\n")
        cur_text_list[-1] = text
        self.ui.progress_text.setText("\n".join(cur_text_list))

    def process_pipeline_finished_handler(self):
        self.ui.btn_start.setEnabled(True)
        self.pred_flag = self.progress.get_pred_flag()
        self.model = self.progress.get_model()
        self.pred_data = self.progress.get_pred_data()
        self.golgi_images = self.progress.get_golgi_images()
        self.crop_golgi_list, self.shifted_crop_golgi_list, \
        self.giantin_mask_list, self.giantin_pred_list = self.progress.get_crop_golgi()
        if len(self.crop_golgi_list) == 0:
            self.update_message("No satisfied giantin found. Try to use ImageJ manually.")
            return
        try:
            # show result in tab2
            self.show_golgi()
            # go to tab2
            self.ui.tabWidget.setCurrentIndex(1)
            self.ui.btn_show_avergaed.setEnabled(True)
        except Exception as e:
            self.pred_flag = True
            self.ui.progress_text.append("{}".format(e))
            self.logger.error("{}".format(e), exc_info=True)
            self.ui.btn_start.setEnabled(True)

    def process_pipeline_error_handler(self):
        self.ui.btn_start.setEnabled(True)
        self.pred_flag = self.progress.get_pred_flag()
        self.golgi_images = self.progress.get_golgi_images()

    def start(self):
        self.selected_list = []
        try:
            self.ui.progress_text.clear()
            param_flag = self.get_write_param()
            if not param_flag:
                return
            if self.pred_data is None:
                self.pred_flag = True
            self.ui.btn_start.setDisabled(True)
            self.ui.progress_text.clear()
            self.logger.info("start")

            if self.thread is not None:
                self.thread.terminate()
            self.thread = QThread(self)
            self.logger.info('start doing stuff in: {}'.format(QThread.currentThread()))
            self.progress = Progress(model=self.model, logger=self.logger, image_path_list=self.param_dict["path_list"],
                                     param_pixel_threshold=self.param_dict["param_pixel_threshold"],
                                     param_giantin_threshold=self.param_dict["param_giantin_threshold"],
                                     param_giantin_area_threshold=self.param_dict["param_giantin_area_threshold"],
                                     param_giantin_roi_size=self.param_dict["param_giantin_roi_size"],
                                     param_giantin_channel=self.param_dict["param_giantin_channel"],
                                     param_blank_channel=self.param_dict["param_blank_channel"],
                                     param_giantin_overlap=self.param_dict["param_giantin_overlap"],
                                     pred_flag=self.pred_flag, pred_data=self.pred_data, golgi_images=self.golgi_images)

            self.progress.moveToThread(self.thread)
            self.start_backgroung_work.connect(self.progress.pipeline)
            self.progress.append_text.connect(self.update_message)
            self.progress.update_progress.connect(self.update_process)
            self.progress.pipeline_finished.connect(self.process_pipeline_finished_handler)
            self.progress.pipeline_error.connect(self.process_pipeline_error_handler)

            self.thread.start()
            self.start_backgroung_work.emit()
        except Exception as e:
            self.pred_flag = True
            self.ui.progress_text.append("{}".format(e))
            self.logger.error("{}".format(e), exc_info=True)
            self.ui.btn_start.setEnabled(True)

    def subplot_left_click(self, event):
        axes = event.inaxes
        if axes is None:
            return
        # axes_id = self.axes_dict[hash(id(axes))]
        if id(axes) not in self.axes_dict.keys():
            return
        if len(axes.patches) > 0:
            axes.patches.pop()
            axes.patches.pop()
            event.canvas.draw()
        else:
            ax_h, ax_w = 701, 701
            axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, facecolor="white", alpha=0.3))
            axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, fill=False, edgecolor="red", linewidth=5))
            event.canvas.draw()

        axes_id = self.axes_dict[id(axes)]
        if axes_id in self.selected_list:
            self.selected_list.remove(axes_id)
        else:
            self.selected_list.append(axes_id)

    def subplot_right_click(self, event):
        axes = event.inaxes
        if axes is None:
            return
        # axes_id = self.axes_dict[hash(id(axes))]
        if id(axes) not in self.axes_dict.keys():
            return
        self.axes_id = self.axes_dict[id(axes)]

        self.popup_golgi_widget = GolgiDetailWidget("Golgi details", logger=self.logger,
                                                    crop_golgi=self.crop_golgi_list[self.axes_id],
                                                    giantin_mask=self.giantin_mask_list[self.axes_id],
                                                    giantin_pred=self.giantin_pred_list[self.axes_id],
                                                    param_dict=self.param_dict,
                                                    save_directory=self.save_directory,
                                                    channel_name=self.get_cur_channel_name())
        self.popup_golgi_widget.show()

        self.popup_golgi_widget.save_signal.connect(self.update_sub_data)

    def update_sub_data(self):
        new_crop, new_shifted_golgi, new_mask, new_pred = self.popup_golgi_widget.get_new_data()
        if new_shifted_golgi is not None:
            self.crop_golgi_list[self.axes_id] = new_crop
            self.giantin_mask_list[self.axes_id] = new_mask
            self.shifted_crop_golgi_list[self.axes_id] = new_shifted_golgi
            self.giantin_pred_list[self.axes_id] = new_pred
        self.popup_golgi_widget.close()
        self.show_golgi()

    def subplot_onclick_handler(self, event):
        print('you pressed', event.button, event.xdata, event.ydata)
        # MouseButton.Left: 1
        # MouseButton.Right: 3
        if event.button == 3:
            # mouse right click to open giantin details in new window
            self.subplot_right_click(event)
            ...
        else:
            self.subplot_left_click(event)
        print(self.selected_list)

    def show_golgi(self):
        # clear old plot
        self.ui.scroll_golgi_content = QWidget()
        self.ui.scroll_golgi.setWidget(self.ui.scroll_golgi_content)

        for i, crop_golgi in enumerate(self.shifted_crop_golgi_list):
            if crop_golgi.shape[-1] == 2:
                empty_shape = crop_golgi.shape[:2] + (1,)
                empty_img = np.zeros(shape=empty_shape)
                self.shifted_crop_golgi_list[i] = np.dstack([crop_golgi, empty_img])
        giantin_list = np.array(self.shifted_crop_golgi_list)[:, :, :, self.param_dict["param_giantin_channel"]]
        num_giantin = giantin_list.shape[0]
        qScrollLayout = QVBoxLayout()
        qfigWidget = QWidget()

        columns = 4
        rows = int(num_giantin / columns + 1)
        static_canvas = FigureCanvas(Figure(figsize=(1.7 * columns, 1.7 * rows)))
        static_canvas.mpl_connect('button_press_event', self.subplot_onclick_handler)

        axes_dict = {}
        subplot_axes = static_canvas.figure.subplots(rows, columns)
        static_canvas.figure.tight_layout()
        static_canvas.figure.subplots_adjust(hspace=0.3)

        x_axis_labels = np.arange(0, 701, 350)
        y_axis_labels = np.arange(700, -1, -350)
        font_size = 9
        for i, axes in enumerate(subplot_axes.reshape(-1)):
            if i >= num_giantin:
                axes.axis("off")
            else:
                for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                    label.set_fontsize(font_size)
                # key = hash(id(axes))
                key = id(axes)
                axes_dict[key] = i
                img = giantin_list[i]
                axes.set_xlim(np.min(x_axis_labels), np.max(x_axis_labels))
                axes.set_ylim(np.max(y_axis_labels), np.min(y_axis_labels))
                axes.set_xticks(x_axis_labels)
                axes.set_yticks(y_axis_labels)
                img_ = axes.imshow(img)
                cbar = static_canvas.figure.colorbar(img_, ax=axes)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(font_size)
                if i in self.selected_list:
                    # keep original selected
                    ax_h, ax_w = 701, 701
                    axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, facecolor="white", alpha=0.3))
                    axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, fill=False, edgecolor="red", linewidth=5))
        self.axes_dict = axes_dict

        static_canvas.draw()
        plotLayout = QVBoxLayout()
        plotLayout.addWidget(static_canvas)
        qfigWidget.setLayout(plotLayout)
        static_canvas.setMinimumSize(static_canvas.size())
        qScrollLayout.addWidget(qfigWidget)
        self.ui.scroll_golgi_content.setLayout(qScrollLayout)
        self.ui.scroll_golgi_content.show()

    def show_averaged(self):
        if self.ui.btn_pick.isChecked():
            # pick_select
            selected_shifted_golgi = np.array(self.shifted_crop_golgi_list)[self.selected_list]
        else:
            # drop_select
            selected_shifted_golgi = np.delete(np.array(self.shifted_crop_golgi_list), self.selected_list, axis=0)
        averaged_golgi = np.mean(selected_shifted_golgi, axis=0)
        num_selected = selected_shifted_golgi.shape[0]
        self.popup_averaged = GolgiDetailWidget("Averaged golgi mini-stacks", logger=self.logger, mode=2,
                                                save_directory=self.save_directory,
                                                param_dict=
                                                {"param_giantin_channel": self.param_dict["param_giantin_channel"]},
                                                channel_name=self.get_cur_channel_name())
        self.popup_averaged.show()
        self.popup_averaged.show_averaged_w_plot(averaged_golgi=averaged_golgi, num_ministacks=num_selected)

    def save_golgi_stacks(self):
        # save all golgi mini stacks
        if self.ui.btn_pick.isChecked():
            # pick_select
            selected_shifted_golgi = np.array(self.shifted_crop_golgi_list)[self.selected_list]
        else:
            # drop_select
            selected_shifted_golgi = np.delete(np.array(self.shifted_crop_golgi_list), self.selected_list, axis=0)
        self.save_golgi_dialog = DialogSave(selected_shifted_golgi, exp_name=self.exp_name,
                                            save_directory=self.save_directory)
        self.save_golgi_dialog.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
