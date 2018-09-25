

import sys
from PyQt5 import QtGui
from pynfb.experiment import Experiment
from pynfb.io.xml_ import xml_file_to_params

def main():

    app = QtGui.QApplication(sys.argv)
    experiment = Experiment(app, xml_file_to_params('set_center_out_rot_10s_gen_test.xml'))
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()