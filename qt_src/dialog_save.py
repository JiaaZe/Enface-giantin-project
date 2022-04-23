import os.path
import sys

import numpy as np
import tifffile.tifffile

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import pyqtSignal as Signal

from qt_ui.dialog_save import Ui_Dialog_save
from utils import open_file_dialog


class DialogSave(QWidget):
    save_signal = Signal(int)

    def __init__(self, crop_golgi_data, save_directory=None, exp_name=None):
        # crop_golgi_data: shape:[n,701,701,c]
        super().__init__()
        self.ui = Ui_Dialog_save()
        self.ui.setupUi(self)

        if exp_name is not None:
            self.exp_name = exp_name
            self.ui.exp_name_text.setText(exp_name)
        else:
            self.exp_name = ""
        if save_directory is not None:
            self.path = save_directory
            self.ui.path_text.setText(save_directory)
        else:
            self.path = ""
        self.data = crop_golgi_data

        self.ui.btn_save.setDisabled(True)
        if len(self.exp_name) > 0 and len(self.path) > 0:
            self.ui.btn_save.setEnabled(True)
        self.ui.exp_name_text.textChanged.connect(lambda: self.enable_save_btn())
        self.ui.path_text.textChanged.connect(lambda: self.enable_save_btn())

        self.ui.btn_browse.clicked.connect(lambda: self.btn_browse_handler())
        self.ui.btn_save.clicked.connect(lambda: self.save_handler())
        self.ui.btn_cancel.clicked.connect(lambda: self.close())

    def enable_save_btn(self):
        self.exp_name = self.ui.exp_name_text.text()
        self.path = self.ui.path_text.text()

        if len(self.exp_name) > 0 and len(self.path) > 0:
            self.ui.btn_save.setEnabled(True)

    def btn_browse_handler(self):
        self.path = open_file_dialog(mode=4)
        self.ui.path_text.setText(self.path)

    def save_handler(self):
        data_shape = self.data.shape
        c = data_shape[-1]
        for c_ in range(c):
            data_in_channel = self.data[:, :, :, c_]
            filename = os.path.join(self.path, "averaged_{}_C{}.tif".format(self.exp_name, c_ + 1))
            tifffile.imsave(file=filename, data=data_in_channel)
        os.startfile(self.path)
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    data = tifffile.imread("../jupyter_data/C1-giantin-647.tif")
    input_data = np.expand_dims(data, axis=-1)
    window = DialogSave(crop_golgi_data=input_data)
    window.show()
    sys.exit(app.exec())
