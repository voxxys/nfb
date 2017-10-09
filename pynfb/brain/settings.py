from pyqtgraph.Qt import QtGui
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import ParameterTree

class MyGroupParameter(pTypes.GroupParameter):
    def addChild(self, child, *args, **kwargs):
        # child can be a Parameter instance or it can be a dict
        try:
            child_name = child.name()
        except AttributeError:
            child_name = child['name']

        attr_name = '_'.join(child_name.lower().split(' '))
        # Can't use hasattr because Parameter.__getattr__ looks for attr in self.names which is a dict of children's
        # names
        if attr_name in self.__dict__:
            msg = "Can't add child '{}' to '{}' because it already has an attribute named '{}'".format(
                child_name, self.name(), attr_name)
            raise ValueError(msg)
        else:
            child = super().addChild(child, *args, **kwargs)
            setattr(self, attr_name, child)

    def create_widget(self):
        self.widget = ParameterTree(showHeader=False)
        self.widget.setParameters(self, showTop=True)
        PREFERRED = QtGui.QSizePolicy.Preferred
        self.widget.setSizePolicy(PREFERRED, PREFERRED)
        return self.widget


class SourceSpaceWidgetPainterSettings(MyGroupParameter):
    COLORMAP_LIMITS_GLOBAL = 'global'
    COLORMAP_LIMITS_LOCAL = 'local'
    COLORMAP_LIMITS_MANUAL = 'manual'
    COLORMAP_LIMITS_MODES = [COLORMAP_LIMITS_GLOBAL, COLORMAP_LIMITS_MANUAL, COLORMAP_LIMITS_LOCAL]

    def __init__(self):
        opts = {'name': 'Visualization settings', 'type': 'group', 'value': 'true'}
        pTypes.GroupParameter.__init__(self, **opts)

        # Colormap options
        cmap_children = [
            {'name': 'Mode', 'type': 'list',
             'values': self.COLORMAP_LIMITS_MODES, 'value': 'global'},
            {'name': 'Lock current limits', 'type': 'bool', 'value': False, },
            {'name': 'Lower limit', 'type': 'float', 'readonly': True},
            {'name': 'Upper limit', 'type': 'float', 'readonly': True},
            {'name': 'Threshold pct', 'type': 'int', 'readonly': True, 'value': 90},
        ]
        colormap = MyGroupParameter(name='Colormap', children=cmap_children)

        def type_changed():
            def limits_setReadonly(readonly):
                colormap.lower_limit.setReadonly(readonly)
                colormap.upper_limit.setReadonly(readonly)

            if colormap.mode.value() == self.COLORMAP_LIMITS_GLOBAL:
                limits_setReadonly(True)
                colormap.lock_current_limits.setReadonly(False)
            if colormap.mode.value() == self.COLORMAP_LIMITS_LOCAL:
                limits_setReadonly(True)
                colormap.lock_current_limits.setReadonly(True)
            if colormap.mode.value() == self.COLORMAP_LIMITS_MANUAL:
                limits_setReadonly(False)
                colormap.lock_current_limits.setReadonly(True)

        colormap.mode.sigValueChanged.connect(type_changed)

        self.addChild(colormap)

