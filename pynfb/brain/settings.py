import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import opengl as gl
app = QtGui.QApplication([])
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from pyqtgraph.parametertree.parameterTypes import ListParameter

class SourceSpaceWidgetPainterSetting(pTypes.GroupParameter):
    cmap_children = [
        {'name': 'Type', 'type': 'list', 'values': ['local', 'global', 'manual'], 'value': 'global'},
        {'name': 'Lock current limits', 'type': 'bool', 'value': False, },
        {'name': 'Lower limit', 'type': 'float', 'readonly': True},
        {'name': 'Upper limit', 'type': 'float', 'readonly': True},
        {'name': 'Threshold', 'type': 'float'},
    ]

    def __init__(self):
        self = Parameter.create(name='Colormap', type='group', children=self.cmap_children)


cmap_param_group = Parameter.create(name='Colormap', type='group', children=cmap_children)
list_type, chb_lock, lower_limit, upper_limit, threshold = cmap_param_group.childs

def type_changed():
    def limits_setReadonly(readonly):
        lower_limit.setReadonly(readonly)
        upper_limit.setReadonly(readonly)

    if list_type.value() == 'local':
        limits_setReadonly(True)
        chb_lock.setReadonly(True)
    if list_type.value() == 'local':
        limits_setReadonly(True)
        chb_lock.setReadonly(True)
    if list_type.value() == 'manual':
        limits_setReadonly(False)
        chb_lock.setReadonly(True)
list_type.sigValueChanged.connect(type_changed)



# Draw:


cmap_setting_widget = ParameterTree(showHeader=False)
cmap_setting_widget.setParameters(cmap_param_group, showTop=True)
cmap_setting_widget.setWindowTitle('pyqtgraph example: Parameter Tree')

win = QtGui.QWidget()
layout = QtGui.QVBoxLayout()
win.setLayout(layout)
layout.addWidget(cmap_setting_widget)
win.show()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()