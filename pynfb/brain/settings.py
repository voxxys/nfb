from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree


class SourceSpaceWidgetPainterSettings(pTypes.GroupParameter):
    def __init__(self):
        opts = {'name': 'Visualization settings', 'type': 'group', 'value': 'true'}
        pTypes.GroupParameter.__init__(self, **opts)

        # Colormap options
        cmap_children = [
            {'name': 'Type', 'type': 'list', 'values': ['local', 'global', 'manual'], 'value': 'global'},
            {'name': 'Lock current limits', 'type': 'bool', 'value': False, },
            {'name': 'Lower limit', 'type': 'float', 'readonly': True},
            {'name': 'Upper limit', 'type': 'float', 'readonly': True},
            {'name': 'Threshold', 'type': 'float', 'readonly': True},
        ]
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

        self.addChild(cmap_param_group)

    def create_widget(self):
        self.widget = ParameterTree(showHeader=False)
        self.widget.setParameters(self, showTop=True)
        PREFERRED = QtGui.QSizePolicy.Preferred
        self.widget.setSizePolicy(PREFERRED, PREFERRED)
        return self.widget



if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        settings = SourceSpaceWidgetPainterSettings()

        # Draw:


        settings_widget = settings.create_widget()

        win = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        win.setLayout(layout)
        layout.addWidget(settings_widget)
        win.show()