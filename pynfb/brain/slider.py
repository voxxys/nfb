from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import parameterTypes as pTypes, Parameter, registerParameterType


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

        self.slider_widget.sigChanged = self.slider_widget.slider.sliderReleased
        self.slider_widget.value = self.slider_widget.value
        self.slider_widget.setValue = self.slider_widget.setValue
        return self.slider_widget


class SliderParameter(Parameter):
    """Used for displaying a slider within the tree."""
    itemClass = SliderParameterItem


registerParameterType('slider', SliderParameter, override=True)