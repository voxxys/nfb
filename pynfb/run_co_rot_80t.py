import os
import sys
from PyQt5 import QtGui

full_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__))+'/..')
sys.path.insert(0, full_path)

from pynfb.experiment import Experiment
from pynfb.io.xml_ import xml_file_to_params

def main():

    app = QtGui.QApplication(sys.argv)
    experiment = Experiment(app, xml_file_to_params('set_co_80trials_rot.xml'))
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()