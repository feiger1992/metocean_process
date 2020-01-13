# encoding = utf-8
import Metocean
from Metocean.PYQT import M_window
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = M_window()

    ex.show()
    sys.exit(app.exec_())
