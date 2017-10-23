from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import ParameterTree, parameterTypes as pTypes

from .slider import SliderParameter # Importing will register 'slider' type


class MyGroupParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        super().__init__(**opts)

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
    COLORMAP_BUFFER_LENGTH_MAX = 40000.0
    COLORMAP_BUFFER_LENGTH_DEFAULT = 6000.0

    def __init__(self, fs):
        opts = {'name': 'Visualization settings', 'type': 'group', 'value': 'true'}
        super().__init__(**opts)
        self.fs = fs

        # Colormap settings
        cmap_children = [
            {'name': 'Mode', 'type': 'list', 'values': self.COLORMAP_LIMITS_MODES, 'value': 'global'},
            {'name': 'Lock current limits', 'type': 'bool', 'value': False, },
            {'name': 'Buffer length', 'type': 'slider', 'value': self.COLORMAP_BUFFER_LENGTH_DEFAULT / self.fs,
                                      'limits': (0, self.COLORMAP_BUFFER_LENGTH_MAX / self.fs), 'prec': 3},
            {'name': 'Upper limit', 'type': 'float', 'readonly': True, 'decimals': 3},
            {'name': 'Threshold pct', 'type': 'slider', 'suffix': '%', 'readonly': False, 'limits': (0, 100),
                                      'value': 50, 'prec': 0},

        ]
        colormap = MyGroupParameter(name='Colormap', children=cmap_children)

        def mode_changed(param, mode):
            if mode == self.COLORMAP_LIMITS_LOCAL:
                colormap.upper_limit.setReadonly(True)
                colormap.lock_current_limits.setReadonly(True)
            elif mode == self.COLORMAP_LIMITS_GLOBAL:
                colormap.upper_limit.setReadonly(True)
                colormap.lock_current_limits.setReadonly(False)
            elif mode == self.COLORMAP_LIMITS_MANUAL:
                colormap.upper_limit.setReadonly(False)
                colormap.lock_current_limits.setReadonly(True)
        colormap.mode.sigValueChanged.connect(mode_changed)

        self.addChild(colormap)


class SourceSpaceReconstructorSettings(MyGroupParameter):
    def __init__(self, fs):
        opts = {'name': 'Source space reconstruction settings', 'type': 'group', 'value': 'true'}
        super().__init__(**opts)
        self.fs = fs

        # Transformation settings
        trans_children = [
            {'name': 'Apply', 'type': 'bool', 'value': False},
            {'name': 'Lower cutoff', 'type': 'float', 'decimals': 1, 'suffix': 'Hz', 'limits': (0, None), 'value': 0.1},
            {'name': 'Upper cutoff', 'type': 'int', 'suffix': 'Hz', 'limits': (0, 100), 'value': 40}
        ]
        transformation = MyGroupParameter(name='Linear filter', children=trans_children)
        self.addChild(transformation)

        # Envelope extraction
        envelope = {'name': 'Extract envelope', 'type': 'bool', 'value': False}
        self.addChild(envelope)

        # Local desynchronisation
        desync_children = [
            {'name': 'Apply', 'type': 'bool', 'value': False},
            {'name': 'Window width', 'type': 'slider', 'suffix': '', 'readonly': False,
             'limits': (0.0 / self.fs, 1000.0 / self.fs), 'value': 0.050, 'prec': 3},
            {'name': 'Lag', 'type': 'slider', 'suffix': ' s', 'readonly': False,
             'limits': (0.0 / self.fs, 10000.0 / self.fs), 'value': 1.000, 'prec': 3},
        ]
        desync = MyGroupParameter(name='Linear desynchronisation', children=desync_children)
        self.addChild(desync)

class SourceSpaceSettings(MyGroupParameter):
    def __init__(self, painter_settings=None, reconstructor_settings=None, fs=None):
        opts = {'name': 'Source space settings', 'type': 'group', 'value': 'true'}
        super().__init__(**opts)
        self.fs = fs

        self.addChildren([
            painter_settings or SourceSpaceWidgetPainterSettings(fs=self.fs),
            reconstructor_settings or SourceSpaceReconstructorSettings(fs=self.fs),
        ])