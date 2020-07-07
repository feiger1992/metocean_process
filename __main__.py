# encoding = utf-8
import Metocean
from Metocean.PYQT import M_window
from PyQt5.QtWidgets import QApplication
import sys
from Metocean.tide import Process_Tide
from Metocean.current import Single_Tide_Point, Current_pre_process, Read_Report, small_diff_dir
from Metocean.Wind_and_Wave import Wind_Wave

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = M_window()

    ex.show()
    sys.exit(app.exec_())
