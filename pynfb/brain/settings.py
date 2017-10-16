from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import ParameterTree, parameterTypes as pTypes, Parameter, registerParameterType


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
    COLORMAP_BUFFER_LENGTH_MAX = 40000
    COLORMAP_BUFFER_LENGTH_DEFAULT = 6000

    def __init__(self):
        opts = {'name': 'Visualization settings', 'type': 'group', 'value': 'true'}
        super().__init__(**opts)

        # Colormap settings
        cmap_children = [
            {'name': 'Mode', 'type': 'list', 'values': self.COLORMAP_LIMITS_MODES, 'value': 'global'},
            {'name': 'Lock current limits', 'type': 'bool', 'value': False, },
            {'name': 'Buffer length', 'type': 'slider', 'value': self.COLORMAP_BUFFER_LENGTH_DEFAULT,
                                      'limits': (0, self.COLORMAP_BUFFER_LENGTH_MAX), 'prec': 0},
            {'name': 'Upper limit', 'type': 'float', 'readonly': True},
            {'name': 'Threshold pct', 'type': 'slider', 'suffix': '%', 'readonly': False, 'limits': (0, 100),
                                      'value': 50, 'prec': 0},

        ]
        colormap = MyGroupParameter(name='Colormap', children=cmap_children)

        def type_changed():

            if colormap.mode.value() == self.COLORMAP_LIMITS_GLOBAL:
                colormap.upper_limit.setReadonly(True)
                colormap.lock_current_limits.setReadonly(False)

            if colormap.mode.value() == self.COLORMAP_LIMITS_LOCAL:
                colormap.upper_limit.setReadonly(True)
                colormap.lock_current_limits.setReadonly(True)

            if colormap.mode.value() == self.COLORMAP_LIMITS_MANUAL:
                colormap.upper_limit.setReadonly(False)
                colormap.lock_current_limits.setReadonly(True)

        colormap.mode.sigValueChanged.connect(type_changed)

        self.addChild(colormap)


class SourceSpaceReconstructorSettings(MyGroupParameter):
    def __init__(self):
        opts = {'name': 'Source space reconstruction settings', 'type': 'group', 'value': 'true'}
        super().__init__(**opts)

        # Transformation settings
        trans_children = [
            {'name': 'Apply linear filter', 'type': 'bool', 'value': False},
            {'name': 'Lower cutoff', 'type': 'float', 'dec': 1, 'suffix': 'Hz'},
            {'name': 'Upper cutoff', 'type': 'int', 'suffix': 'Hz'}
        ]
        transformation = MyGroupParameter(name='Transformation', children=trans_children)
        self.addChild(transformation)


class SourceSpaceSettings(MyGroupParameter):
    def __init__(self):
        opts = {'name': 'Source space settings', 'type': 'group', 'value': 'true'}
        super().__init__(**opts)

        self.addChildren([
            SourceSpaceWidgetPainterSettings(),
            SourceSpaceReconstructorSettings(),
        ])

# Adapted from https://stackoverflow.com/a/42011414/3042770
class Slider(QtGui.QWidget):
    def __init__(self, minimum, maximum, value=None, suffix='', prec=1, parent=None, **kwargs):
        super().__init__(parent=parent)
        self.suffix = suffix
        self.prec = prec

        self.outerLayout = QtGui.QHBoxLayout(self)
        self.outerLayout.setContentsMargins(0, 0, 0, 0)
        self.outerLayout.setSpacing(0)

        self.label = QtGui.QLabel(self)
        self.outerLayout.addWidget(self.label)

        # Start of innerLayout - slider with spacer items on its sides
        self.innerLayout = QtGui.QHBoxLayout()

        spacerItem = QtGui.QSpacerItem(0, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.innerLayout.addItem(spacerItem)

        self.slider = QtGui.QSlider(self)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.innerLayout.addWidget(self.slider)

        spacerItem1 = QtGui.QSpacerItem(0, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.innerLayout.addItem(spacerItem1)
        # End of innerLayout

        self.outerLayout.addLayout(self.innerLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self._setLabelValue_based_on_slider_value)
        if value:
            self.slider.setValue(self._value_to_slider_value(value))
            self.x = value
        else:
            self.x = self.minimum
        self._setLabelValue_based_on_slider_value(self.slider.value())

    def _setLabelValue_based_on_slider_value(self, slider_value):
        self.x = self._slider_value_to_value(slider_value)

        if self.prec == 0:
            label_text = "{0:d}{1}".format(int(self.x), self.suffix)
        else:
            label_text = "{0:.{2}g}{1}".format(self.x, self.suffix, self.prec)

        max_len = (
            max(len(str(int(limit))) for limit in (self.minimum, self.maximum))
            + (0 if self.prec ==0 else self.prec+1) + len(self.suffix)
        )

        self.label.setText("{:{width}}".format(label_text, width=max_len))

    def value(self):
        return self._slider_value_to_value(self.slider.value())

    def setValue(self, value):
        self.slider.setValue(self._value_to_slider_value(value))

    def _slider_value_to_value(self, slider_value):
        return self.minimum + (slider_value / (self.slider.maximum() - self.slider.minimum())) * (
            self.maximum - self.minimum)

    def _value_to_slider_value(self, value):
        return self.slider.minimum() + value / (self.maximum - self.minimum) * (
            self.slider.maximum() - self.slider.minimum())


class SliderParameterItem(pTypes.WidgetParameterItem):
    def __init__(self, param, depth):
        super().__init__(param, depth)
        self.hideWidget = False

    def makeWidget(self):

        opts = self.param.opts.copy()
        if 'limits' in opts:
            opts['minimum'], opts['maximum'] = opts['limits']
        else:
            raise ValueError("You have to provide 'limits' for this parameter")
        self.slider_widget = Slider(**opts)

        self.slider_widget.sigChanged = self.slider_widget.slider.valueChanged
        self.slider_widget.value = self.slider_widget.value
        self.slider_widget.setValue = self.slider_widget.setValue
        return self.slider_widget


class SliderParameter(Parameter):
    """Used for displaying a slider within the tree."""
    itemClass = SliderParameterItem


registerParameterType('slider', SliderParameter, override=True)
