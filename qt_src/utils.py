import logging
import os
import platform
import subprocess
import zipfile
import time
from logging.handlers import TimedRotatingFileHandler

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QContextMenuEvent, QCursor
from PyQt5.QtWidgets import QListWidget, QFileDialog, QListView, QAbstractItemView, QTreeView, QWidget, QMenu, QAction

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
import roifile


def coord2list(coord):
    x, y, w, h = coord
    coord_list = [[x, y], [x + h, y], [x + h, y + w], [x, y + w]]
    return coord_list


def get_zip(files, zip_name):
    zp = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for file in files:
        zp.write(file)
    zp.close()
    time.sleep(1)


def coord2roi(coords, output_folder, zip_name, giantin_channel):
    # temp_path = os.path.join(output_folder, "roi_temp")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    roi_path_list = []
    for i, coord in enumerate(coords):
        coord_list = coord2list(coord)
        roi = roifile.ImagejRoi.frompoints(coord_list)
        c_x = int(coord_list[0][0] / 2 + coord_list[1][0] / 2)
        c_y = int(coord_list[0][1] / 2 + coord_list[3][1] / 2)
        roi_path = './000{}-{:04d}-{:04d}.roi'.format(giantin_channel, c_x, c_y)
        roi_path_list.append(roi_path)
        roi.tofile(roi_path)
    zip_file = os.path.join(output_folder, zip_name + ".zip")
    replicate_time = 0
    while os.path.exists(zip_file):
        replicate_time += 1
        zip_name_tmp = zip_name + "({}).zip".format(replicate_time)
        zip_file = os.path.join(output_folder, zip_name_tmp)
    get_zip(roi_path_list, zip_file)
    for roi_path in roi_path_list:
        os.remove(roi_path)
    return "Create " + zip_file


def open_file_dialog(mode=1, filetype_list=[], folder=""):
    """
    :param mode: 1. multiple directories, 2.single file, 3. multiple fils, 4 single directory
    :param filetype_list:
    :param folder: default open folder.
    :return:
    """
    fileDialog = QFileDialog()
    if len(folder) > 0:
        fileDialog.setDirectory(folder)
    path = ""
    path_list = []
    if mode == 1:
        # multiple directories
        fileDialog.setFileMode(QFileDialog.Directory)
        # path = fileDialog.getExistingDirectory()
        fileDialog.setOption(QFileDialog.DontUseNativeDialog, True)
        # fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
        file_view = fileDialog.findChild(QListView)

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        f_tree_view = fileDialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if fileDialog.exec():
            path_list = fileDialog.selectedFiles()
            path = ';'.join(path_list)
    elif mode == 4:
        fileDialog.setFileMode(QFileDialog.Directory)
        path = fileDialog.getExistingDirectory()
        return path
    else:
        # single file
        fileDialog.setFileMode(QFileDialog.ExistingFile)
        name_filter = ""
        if len(filetype_list) > 0:
            for filetype in filetype_list:
                if len(name_filter) > 0:
                    name_filter += ";;"
                name_filter += "{} files (*.{} *.{})".format(filetype, filetype, filetype.upper())
            path = fileDialog.getOpenFileName(filter=name_filter)[0]
        else:
            path = fileDialog.getOpenFileName()[0]
        path_list.append(path)
    return path_list


def open_folder_func(folder_path):
    # open saved folder.
    systemType = platform.platform()
    if "mac" in systemType:
        subprocess.call(["open", folder_path])
    else:
        os.startfile(folder_path)


class ListBoxWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.resize(600, 600)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()

            links = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    links.append(str(url.toLocalFile()))
                else:
                    links.append(str(url.toString()))
            self.addItems(links)
        else:
            event.ignore()

    def remove_items(self):
        selected = self.selectedItems()
        for i in selected:
            row = self.row(i)
            self.takeItem(row)

    def get_all(self):
        self.selectAll()
        items = []
        for i in self.selectedItems():
            items.append(i.text())
        return items


class MaskWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setAttribute(Qt.WA_StyledBackground)
        self.setStyleSheet('background:rgba(0,0,0,200);')
        self.setAttribute(Qt.WA_DeleteOnClose)

    def show(self):
        if self.parent() is None:
            return

        parent_rect = self.parent().geometry()
        self.setGeometry(0, 0, parent_rect.width(), parent_rect.height())
        super().show()


class MyWidget(QWidget):
    def __init__(self, menu_action_func):
        super().__init__()
        self.menu = None
        self.menu_action_func = menu_action_func

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        print(event)
        self.menu = QMenu()
        openAction = QAction('See Details', self)
        openAction.triggered.connect(self.menu_action_func)
        self.menu.addAction(openAction)
        self.menu.popup(QCursor.pos())


class SquareWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        h_size, w_size = 290, 230
        self.setFixedSize(QtCore.QSize(h_size, w_size))
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))


class MyCanvas(FigureCanvas):
    def __init__(self, figure=None, index=None, row=None, col=None, clicked=False):
        super().__init__(figure=figure)
        self.index = index
        self.row = row
        self.col = col
        self.clicked = clicked


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_path = './Logs/'
    log_name = log_path + "logFile" + '.log'
    logfile = log_name
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    file_handler = TimedRotatingFileHandler(logfile, when='H', interval=3, backupCount=4, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)

    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO)
    console_logger.setFormatter(formatter)

    logger.addHandler(console_logger)
    logger.addHandler(file_handler)

    return logger
