from PyQt4 import QtGui
from PyQt4 import QtCore

class BCIFitWidget(QtGui.QWidget):
    fit_clicked = QtCore.pyqtSignal()
    fit_2states_clicked = QtCore.pyqtSignal()
    def __init__(self, bci_signal, *args):
        super(BCIFitWidget, self).__init__(*args)
        label = QtGui.QLabel(bci_signal.name)

        # fit button
        fit_button = QtGui.QPushButton('Fit model')
        fit_button.clicked.connect(lambda: self.fit_clicked.emit())

        # fit button
        fit2_button = QtGui.QPushButton('Update stats')
        fit2_button.clicked.connect(lambda: self.fit_2states_clicked.emit())

        # set layout
        layout = QtGui.QHBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignRight)
        layout.addWidget(label)
        layout.addWidget(fit_button)
        layout.addWidget(fit2_button)


if __name__ == '__main__':
    app = QtGui.QApplication([])


    class BCISignalMock:
        def __init__(self):
            self.name = 'bci'

    w = BCIFitWidget(BCISignalMock())
    w.show()
    app.exec_()