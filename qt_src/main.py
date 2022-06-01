import configparser
import os.path
import sys
import re

import numpy as np
from PyQt5.QtCore import QRegularExpression, QThread, pyqtSignal as Signal, QObject
from PyQt5.QtGui import QRegularExpressionValidator, QIntValidator
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QGridLayout, QGroupBox

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
    font_size = 8

    def __init__(self):
        super().__init__()

        self.logger = get_logger()
        self.logger.info("\n==============================START==============================")

        self.cfg = None
        self.param_dict = {}

        self.thread = None
        self.progress = None
        self.roi_worker = None
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
        self.result_golgi_content_list = [self.ui.scroll_golgi_content1,
                                          self.ui.scroll_golgi_content2,
                                          self.ui.scroll_golgi_content3]
        self.result_golgi_tab_list = [self.ui.tab_scroll_golgi1, self.ui.tab_scroll_golgi2, self.ui.tab_scroll_golgi3]

        # tab 1
        # image folder path list view
        self.listview_image_path = ListBoxWidget(self.ui.image_group)
        self.listview_image_path.setDragEnabled(True)
        self.listview_image_path.setObjectName("listview_image_path")
        self.listview_image_path.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.ui.horizontalLayout_2.insertWidget(1, self.listview_image_path)

        self.ui.btn_image_clear.clicked.connect(lambda: self.listview_image_path.clear())
        self.ui.btn_image_remove.clicked.connect(lambda: self.listview_image_path.remove_items())

        # each tab's group list
        self.tab_group_list = []
        self.tif_folder_list = []
        self.tif_name_list = []

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
        self.axes_index = None

        self.selected_dict = {}
        self.crop_golgi_list = []
        self.shifted_crop_golgi_list = []
        self.ministacks_roi_list = []
        self.giantin_mask_list = []
        self.giantin_pred_list = []

        self.popup_golgi_widget = None
        self.popup_averaged = None
        self.save_golgi_dialog = None

        self.ui.btn_show_avergaed.setDisabled(True)
        self.ui.btn_show_avergaed.clicked.connect(lambda: self.show_averaged())
        self.ui.btn_save.clicked.connect(lambda: self.save_golgi_stacks())
        self.ui.btn_save_roi.clicked.connect(lambda: self.save_stacks_roi())

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
        tmp_tif_name_list = self.progress.get_tif_name_list()
        tmp_tif_folder_list = self.progress.get_tif_folder_list()
        if len(tmp_tif_name_list) > 0:
            self.tif_name_list = tmp_tif_name_list
            self.tif_folder_list = tmp_tif_folder_list
        self.crop_golgi_list, self.shifted_crop_golgi_list, self.giantin_mask_list, self.giantin_pred_list \
            = self.progress.get_crop_golgi()
        self.ministacks_roi_list = self.progress.get_roi_list()
        if len(self.crop_golgi_list) == 0:
            self.update_message("No satisfied giantin found. Try to use ImageJ manually.")
            return
        try:
            # show result in tab2
            c1_name, c2_name, c3_name = self.get_cur_channel_name()
            self.tab_group_list = []
            self.show_golgi(c1_name, 0)
            self.show_golgi(c2_name, 1)
            self.show_golgi(c3_name, 2)
            # go to tab2
            self.ui.tabWidget.setCurrentIndex(1)
            self.ui.tabWidget_2.setCurrentIndex(self.param_dict["param_giantin_channel"])
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
        self.selected_dict = {}
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
        if len(axes.patches) > 0:
            axes.patches.pop()
            axes.patches.pop()
            event.canvas.draw()
        else:
            ax_h, ax_w = 701, 701
            axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, facecolor="white", alpha=0.3))
            axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, fill=False, edgecolor="red", linewidth=5))
            event.canvas.draw()
        event.canvas.clicked = not event.canvas.clicked

        n, i = self.axes_index
        if n in self.selected_dict.keys():
            selected_list = self.selected_dict[n]
            if i in selected_list:
                selected_list.remove(i)
            else:
                selected_list.append(i)
            if len(selected_list) > 0:
                self.selected_dict[n] = selected_list
            else:
                self.selected_dict.pop(n)
        else:
            self.selected_dict[n] = [i]

        cur_tab_index = self.ui.tabWidget_2.currentIndex()
        for tab_index, tab_content in enumerate(self.result_golgi_content_list):
            if tab_index != cur_tab_index and tab_content.isEnabled():
                group_box = self.tab_group_list[tab_index][n]
                canvas = group_box.layout().itemAt(i).widget().layout().itemAt(0).widget()
                canvas.clicked = not canvas.clicked
                axes = canvas.figure.axes[0]
                if len(axes.patches) > 0:
                    axes.patches.pop()
                    axes.patches.pop()
                else:
                    ax_h, ax_w = 701, 701
                    axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, facecolor="white", alpha=0.3))
                    axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, fill=False, edgecolor="red", linewidth=5))
                canvas.draw()

    def subplot_right_click(self, event):
        n, i = self.axes_index
        axes = event.inaxes
        if axes is None:
            return
        self.popup_golgi_widget = GolgiDetailWidget("Golgi details", logger=self.logger,
                                                    crop_golgi=self.crop_golgi_list[n][i],
                                                    giantin_mask=self.giantin_mask_list[n][i],
                                                    giantin_pred=self.giantin_pred_list[n][i],
                                                    param_dict=self.param_dict,
                                                    save_directory=self.save_directory,
                                                    channel_name=self.get_cur_channel_name())
        self.popup_golgi_widget.show()

        self.popup_golgi_widget.save_signal.connect(self.update_sub_data)

    def update_sub_data(self):
        new_crop, new_shifted_golgi, new_mask, new_pred = self.popup_golgi_widget.get_new_data()
        n, i = self.axes_index
        if new_shifted_golgi is not None:
            self.crop_golgi_list[n][i] = new_crop
            self.giantin_mask_list[n][i] = new_mask
            self.shifted_crop_golgi_list[n][i] = new_shifted_golgi
            self.giantin_pred_list[n][i] = new_pred
        # not sure
        self.popup_golgi_widget.setVisible(False)
        self.popup_golgi_widget.close()
        self.update_all_tab_widget()

    def update_all_tab_widget(self):
        # widget_index = self.axes_index_num
        n, i = self.axes_index
        for tab_index, tab_content in enumerate(self.result_golgi_content_list):
            if tab_content.isEnabled():
                group_box = self.tab_group_list[tab_index][n]
                group_box_gridlayout = group_box.layout()
                square_widget_old = group_box_gridlayout.itemAt(i).widget()
                square_widget_layout = square_widget_old.layout()
                canvas_old = square_widget_layout.itemAt(0).widget()
                index = canvas_old.index
                row = canvas_old.row
                col = canvas_old.col
                old_clicked = canvas_old.clicked

                canvas_new = MyCanvas(Figure(figsize=(3, 5)), index=index, row=row, col=col)
                canvas_new.figure.tight_layout()
                canvas_new.mpl_connect('button_press_event', lambda event: self.subplot_onclick_handler(event))

                ax = canvas_new.figure.subplots()
                img_ = ax.imshow(self.shifted_crop_golgi_list[n][i][:, :, tab_index])
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(self.font_size)
                cbar = canvas_new.figure.colorbar(img_, ax=ax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(self.font_size)

                if old_clicked:
                    canvas_new.clicked = True
                    axes = canvas_new.figure.axes[0]
                    ax_h, ax_w = 701, 701
                    axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, facecolor="white", alpha=0.3))
                    axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, fill=False, edgecolor="red", linewidth=5))
                    canvas_new.draw()

                square_widget_layout.removeWidget(canvas_old)
                canvas_old.deleteLater()
                square_widget_layout.addWidget(canvas_new)

    def subplot_onclick_handler(self, event):
        self.axes_index = event.canvas.index
        # print('you pressed', event.button, event.xdata, event.ydata)
        # MouseButton.Left: 1
        # MouseButton.Right: 3
        if event.button == 3:
            # mouse right click to open giantin details in new window
            self.subplot_right_click(event)
        else:
            self.subplot_left_click(event)

    def show_golgi(self, c_name, tab_index=0):
        # clear old plot
        cur_tab = self.result_golgi_tab_list[tab_index]
        self.ui.tabWidget_2.setTabText(tab_index, c_name)
        if c_name == "":
            self.ui.tabWidget_2.setTabEnabled(tab_index, False)
        else:
            self.ui.tabWidget_2.setTabEnabled(tab_index, True)

        whole_figure_widget = QWidget()
        # print("============{}==========".format(tab_index))
        # print("whole_figure_widget {}".format(id(whole_figure_widget)))

        content_vboxlayout = QVBoxLayout(whole_figure_widget)
        # print("content_gridlayout {}".format(id(content_gridlayout)))
        columns_per_row = 4
        group_list = []
        for n, shifted_crop_golgi_list in enumerate(self.shifted_crop_golgi_list):
            widget_count = 0
            group_box = QGroupBox()
            group_box.setTitle(self.tif_name_list[n])
            group_box.setStyleSheet("background:white;border: 0.5px solid white;")
            group_gridlayout = QGridLayout(group_box)
            for i, crop_golgi in enumerate(shifted_crop_golgi_list):
                if crop_golgi.shape[-1] == 2:
                    empty_shape = crop_golgi.shape[:2] + (1,)
                    empty_img = np.zeros(shape=empty_shape)
                    self.shifted_crop_golgi_list[n][i] = np.dstack([crop_golgi, empty_img])

                row = int(widget_count / columns_per_row)
                column = widget_count % columns_per_row
                square_widget = SquareWidget(whole_figure_widget)

                index = [n, i]
                canvas = MyCanvas(Figure(figsize=(3, 5)), index=index, row=row, col=column)
                widget_count += 1
                canvas.mpl_connect('button_press_event',
                                   lambda event: self.subplot_onclick_handler(event))
                ax = canvas.figure.subplots()
                canvas.figure.tight_layout()
                img_ = ax.imshow(crop_golgi[:, :, tab_index])
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(self.font_size)
                cbar = canvas.figure.colorbar(img_, ax=ax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(self.font_size)

                layout = QVBoxLayout()
                layout.addWidget(canvas)

                square_widget.setLayout(layout)

                group_gridlayout.addWidget(square_widget, row, column, 1, 1)
            group_list.append(group_box)
            content_vboxlayout.addWidget(group_box)
        self.tab_group_list.append(group_list)

        whole_figure_widget.setLayout(content_vboxlayout)

        temp_layout = self.result_golgi_content_list[tab_index].layout()
        if temp_layout is not None:
            # print("content layout {}".format(id(temp_layout)))
            item_widget = temp_layout.itemAt(0).widget()
            # print("item_widget {}".format(id(item_widget)))
            temp_layout.removeWidget(item_widget)
            temp_layout.addWidget(whole_figure_widget)
            item_widget.deleteLater()
        else:
            scroll_layout = QVBoxLayout(self.result_golgi_content_list[tab_index])
            # print("scroll_layout {}".format(id(scroll_layout)))
            scroll_layout.addWidget(whole_figure_widget)
            self.result_golgi_content_list[tab_index].setLayout(scroll_layout)
        self.result_golgi_content_list[tab_index].show()

    def show_averaged(self):
        try:
            selected_shifted_golgi = None
            if self.ui.btn_pick.isChecked():
                # pick_select
                for n in self.selected_dict.keys():
                    selected_index = self.selected_dict[n]
                    if selected_shifted_golgi is None:
                        selected_shifted_golgi = np.array(self.shifted_crop_golgi_list[n])[selected_index]
                    else:
                        selected_shifted_golgi = np.append(selected_shifted_golgi,
                                                           np.array(self.shifted_crop_golgi_list[n])[selected_index],
                                                           axis=0)
            else:
                # drop_select
                if not bool(self.selected_dict):
                    # selected_dict is empty
                    for golgi_list in self.shifted_crop_golgi_list:
                        if len(golgi_list) == 0:
                            continue
                        if selected_shifted_golgi is None:
                            selected_shifted_golgi = golgi_list
                        else:
                            selected_shifted_golgi = np.append(selected_shifted_golgi, golgi_list, axis=0)
                else:
                    selected_shifted_golgi = []
                    for n, golgi_list in enumerate(self.shifted_crop_golgi_list):
                        if n in self.selected_dict.keys():
                            selected_index = self.selected_dict[n]
                            aft_del_np = np.delete(np.array(self.shifted_crop_golgi_list[n]), selected_index, axis=0)
                            if len(aft_del_np) > 0:
                                selected_shifted_golgi.extend(aft_del_np)
                        elif len(golgi_list) > 0:
                            selected_shifted_golgi.extend(np.array(golgi_list))
            averaged_golgi = np.mean(selected_shifted_golgi, axis=0)
            num_selected = len(selected_shifted_golgi)
            self.popup_averaged = GolgiDetailWidget("Averaged golgi mini-stacks", logger=self.logger, mode=2,
                                                    save_directory=self.save_directory,
                                                    param_dict=
                                                    {"param_giantin_channel": self.param_dict["param_giantin_channel"]},
                                                    channel_name=self.get_cur_channel_name())
            self.popup_averaged.show()
            self.popup_averaged.show_averaged_w_plot(averaged_golgi=averaged_golgi, num_ministacks=num_selected)
        except Exception as e:
            err_msg = "Averaging mini-stacks Error: {}".format(e)
            self.ui.progress_text.append(err_msg)
            # go back to tab 1
            self.ui.tabWidget.setCurrentIndex(0)
            self.logger.error(err_msg, exc_info=True)

    def save_golgi_stacks(self):
        try:
            # save all golgi mini stacks
            selected_shifted_golgi = None
            if self.ui.btn_pick.isChecked():
                # pick_select
                for n in self.selected_dict.keys():
                    selected_index = self.selected_dict[n]
                    if selected_shifted_golgi is None:
                        selected_shifted_golgi = np.array(self.shifted_crop_golgi_list[n])[selected_index]
                    else:
                        selected_shifted_golgi = np.append(selected_shifted_golgi,
                                                           np.array(self.shifted_crop_golgi_list[n])[selected_index],
                                                           axis=0)
            else:
                # drop_select
                if not bool(self.selected_dict):
                    # selected_dict is empty
                    for golgi_list in self.shifted_crop_golgi_list:
                        if selected_shifted_golgi is None:
                            selected_shifted_golgi = golgi_list
                        else:
                            selected_shifted_golgi = np.append(selected_shifted_golgi, golgi_list, axis=0)
                else:
                    for n in self.selected_dict.keys():
                        selected_index = self.selected_dict[n]
                        delete_np = np.delete(np.array(self.shifted_crop_golgi_list[n]), selected_index, axis=0)
                        if selected_shifted_golgi is None:
                            selected_shifted_golgi = delete_np
                        else:
                            selected_shifted_golgi = np.append(selected_shifted_golgi, delete_np, axis=0)
            self.save_golgi_dialog = DialogSave(selected_shifted_golgi, exp_name=self.exp_name,
                                                save_directory=self.save_directory)
            self.save_golgi_dialog.show()
        except Exception as e:
            err_msg = "Saving mini-stacks Error: {}".format(e)
            self.ui.progress_text.append(err_msg)
            # go back to tab 1
            self.ui.tabWidget.setCurrentIndex(0)
            self.logger.error(err_msg, exc_info=True)

    class RoiWorder(QObject):
        append_text = Signal(str)
        worker_finished = Signal(int)

        def __init__(self, selected_roi_list, folder_list, name_list, giantin_channel, logger):
            super().__init__()
            self.selected_roi_list = selected_roi_list
            self.folder_list = folder_list
            self.name_list = name_list
            self.giantin_channel = giantin_channel
            self.logger = logger

        def save_roi(self):
            try:
                if len(self.selected_roi_list) == 0:
                    raise Exception("Selected 0 ministack.")
                for n, roi_coords in enumerate(self.selected_roi_list):
                    if len(roi_coords) > 0:
                        msg = coord2roi(roi_coords, self.folder_list[n], "{}_ROI".format(self.name_list[n]),
                                        self.giantin_channel)
                        self.append_text.emit(msg)
                self.worker_finished.emit(0)
            except Exception as e:
                err_msg = "Saving mini-stacks ROI Error: {}".format(e)
                self.append_text.emit(err_msg)
                self.logger.error(err_msg, exc_info=True)
                self.worker_finished.emit(0)

    def save_stacks_roi(self):
        selected_roi_list = self.get_used_stacks(data_used=self.ministacks_roi_list)
        self.ui.tabWidget.setCurrentIndex(0)

        if self.thread is not None:
            self.thread.terminate()
        self.thread = QThread(self)
        self.roi_worker = self.RoiWorder(selected_roi_list=selected_roi_list, folder_list=self.tif_folder_list,
                                         name_list=self.tif_name_list,
                                         logger=self.logger,
                                         giantin_channel=self.param_dict["param_giantin_channel"] + 1)

        self.roi_worker.moveToThread(self.thread)
        self.thread.started.connect(self.roi_worker.save_roi)
        self.roi_worker.append_text.connect(self.update_message)
        self.roi_worker.worker_finished.connect(self.thread.quit)
        self.roi_worker.worker_finished.connect(self.roi_worker.deleteLater)
        self.thread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
