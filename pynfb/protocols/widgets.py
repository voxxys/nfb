import time

import numpy as np
import pyqtgraph as pg

from scipy.misc import imread

from PyQt5.QtGui import QFont

import os
fin_img_dir_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/fin_img')


class ProtocolWidget(pg.PlotWidget):
    def __init__(self, **kwargs):
        super(ProtocolWidget, self).__init__(**kwargs)
        width = 5
        self.setYRange(-width, width)
        self.setXRange(-width, width)
        size = 500
        self.setMaximumWidth(size)
        self.setMaximumHeight(size)
        self.setMinimumWidth(size)
        self.setMinimumHeight(size)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setBackgroundBrush(pg.mkBrush('#252120'))
        self.reward_str = '<font size="4" color="#B48375">Reward: </font><font size="5" color="#91C7A9">{}</font>'
        self.reward = pg.TextItem(html=self.reward_str.format(0))
        self.reward.setPos(-4.7, 4.7)
        self.reward.setTextWidth(300)
        self.addItem(self.reward)
        self.clear_all()

    def clear_all(self):
        for item in self.items():
            self.removeItem(item)
        self.addItem(self.reward)

    def update_reward(self, reward):
        self.reward.setHtml(self.reward_str.format(reward))

    def show_reward(self, flag):
        if flag:
            self.reward.show()
        else:
            self.reward.hide()


class Painter:
    def __init__(self, show_reward=False):
        self.show_reward = show_reward

    def prepare_widget(self, widget):
        widget.show_reward(self.show_reward)


class CircleFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, noise_scaler=2, show_reward=False, radius = 3, circle_border=0, m_threshold=1):
        super(CircleFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.noise_scaler = noise_scaler
        self.x = np.linspace(-np.pi/2, np.pi/2, 100)
        np.random.seed(42)
        self.noise = np.sin(15*self.x)*0.5-0.5 if not circle_border else np.random.uniform(-0.5, 0.5, 100)-0.5
        self.widget = None
        self.radius = radius
        self.m_threshold = m_threshold

    def prepare_widget(self, widget):
        super(CircleFeedbackProtocolWidgetPainter, self).prepare_widget(widget)
        self.p1 = widget.plot(np.sin(self.x), np.cos(self.x), pen=pg.mkPen(229, 223, 213)).curve
        self.p2 = widget.plot(np.sin(self.x), -np.cos(self.x), pen=pg.mkPen(229, 223, 213)).curve
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(229, 223, 213, 25))
        self.fill = fill
        widget.addItem(fill)

    def set_red_state(self, flag):
        if flag:
            self.p1.setPen(pg.mkPen(176, 35, 48))
            self.p2.setPen(pg.mkPen(176, 35, 48))
            self.fill.setBrush(176, 35, 48, 25)
        else:
            self.p1.setPen(pg.mkPen(229, 223, 213))
            self.p2.setPen(pg.mkPen(229, 223, 213))
            self.fill.setBrush(229, 223, 213, 25)

    def redraw_state(self, sample, m_sample):
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)
        noise_ampl = -np.tanh(sample + self.noise_scaler) + 1
        noise = self.noise*noise_ampl
        self.p1.setData(self.radius * np.sin(self.x)*(1+noise), self.radius * np.cos(self.x)*(1+noise))
        self.p2.setData(self.radius * np.sin(self.x)*(1+noise), -self.radius * np.cos(self.x)*(1+noise))
        pass


class BarFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, noise_scaler=2, show_reward=False, radius = 3, circle_border=0, m_threshold=1):
        super(BarFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.x = np.linspace(-1, 1, 100)
        self.widget = None
        self.m_threshold = m_threshold

    def prepare_widget(self, widget):
        super(BarFeedbackProtocolWidgetPainter, self).prepare_widget(widget)
        self.p1 = widget.plot(self.x, np.zeros_like(self.x), pen=pg.mkPen(229, 223, 213)).curve
        self.p2 = widget.plot(self.x, np.zeros_like(self.x)-5, pen=pg.mkPen(229, 223, 213)).curve
        fill = pg.FillBetweenItem(self.p1, self.p2, brush=(229, 223, 213, 25))
        self.fill = fill
        widget.addItem(fill)

    def set_red_state(self, flag):
        if flag:
            self.p1.setPen(pg.mkPen(176, 35, 48))
            self.p2.setPen(pg.mkPen(176, 35, 48))
            self.fill.setBrush(176, 35, 48, 25)
        else:
            self.p1.setPen(pg.mkPen(229, 223, 213))
            self.p2.setPen(pg.mkPen(229, 223, 213))
            self.fill.setBrush(229, 223, 213, 25)

    def redraw_state(self, sample, m_sample):
        if m_sample is not None:
            self.set_red_state(m_sample > self.m_threshold)
        if np.ndim(sample)>0:
            sample = np.sum(sample)
        self.p1.setData(self.x, np.zeros_like(self.x)+max(min(sample, 5), -5))
        self.p2.setData(self.x, np.zeros_like(self.x)-5)
        pass


class BaselineProtocolWidgetPainter(Painter):
    def __init__(self, text='Relax', show_reward=False):
        super(BaselineProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.text = text

    def prepare_widget(self, widget):
        super(BaselineProtocolWidgetPainter, self).prepare_widget(widget)
        self.text_item = pg.TextItem(html='<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text),
                                anchor=(0.5, 0.5))
        self.text_item.setTextWidth(500)
        widget.addItem(self.text_item)
        self.plotItem = widget.plotItem

    def redraw_state(self, sample, m_sample):
        pass

    def set_message(self, text):
        self.text = text
        self.text_item.setHtml('<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text))

class ThresholdBlinkFeedbackProtocolWidgetPainter(Painter):
    def __init__(self, threshold=2000, time_ms=50, show_reward=False):
        super(ThresholdBlinkFeedbackProtocolWidgetPainter, self).__init__(show_reward=show_reward)
        self.threshold = threshold
        self.time_ms = time_ms
        self.blink_start_time = -1
        self.widget = None
        self.x = np.linspace(-10, 10, 2)
        self.previous_sample = -np.inf

    def prepare_widget(self, widget):
        super(ThresholdBlinkFeedbackProtocolWidgetPainter, self).prepare_widget(widget)
        self.p1 = widget.plot([-10, 10], [10, 10], pen=pg.mkPen(77, 144, 254)).curve
        self.p2 = widget.plot([-10, 10], [-10, -10], pen=pg.mkPen(77, 144, 254)).curve
        self.fill = pg.FillBetweenItem(self.p1, self.p2, brush=(255, 255, 255, 25))
        widget.addItem(self.fill)

    def redraw_state(self, samples, m_sample):
        samples = np.abs(samples)
        if np.ndim(samples)==0:
            samples = samples.reshape((1, ))

        previous_sample = self.previous_sample
        do_blink = False
        for sample in samples:
            if (sample >= self.threshold >= previous_sample) and (self.blink_start_time < 0):
                do_blink = True
            previous_sample = sample

        if do_blink:
            self.blink_start_time = time.time()

        if ((time.time() - self.blink_start_time < self.time_ms * 0.001) and (self.blink_start_time > 0)):
            self.fill.setBrush((255, 255, 255, 255))
        else:
            self.blink_start_time = -1
            self.fill.setBrush((255, 255, 255, 10))


        self.previous_sample = previous_sample
        pass


class VideoProtocolWidgetPainter(Painter):
    def __init__(self, video_file_path):
        super(VideoProtocolWidgetPainter, self).__init__()
        self.widget = None
        self.video = None
        self.timer = time.time()
        self.timer_period = 1 / 30
        self.frame_counter = 0
        self.n_frames = None
        self.err_msg = "Could't open video file. "
        import os.path
        if os.path.isfile(video_file_path):
            try:
                import imageio as imageio
                self.video = imageio.get_reader(video_file_path,  'ffmpeg')
                self.n_frames = self.video.get_length() - 1
            except ImportError as e:
                print(e.msg)
                self.err_msg += e.msg
        else:
            self.err_msg = "No file {}".format(video_file_path)


    def prepare_widget(self, widget):
        super(VideoProtocolWidgetPainter, self).prepare_widget(widget)
        if self.video is not None:
            self.img = pg.ImageItem()
            self.img.setScale(10 / self.video.get_data(0).shape[1])
            self.img.rotate(-90)
            self.img.setX(-5)
            self.img.setY(5/self.video.get_data(0).shape[1]*self.video.get_data(0).shape[0])
            widget.addItem(self.img)

        else:
            text_item = pg.TextItem(html='<center><font size="6" color="#a92f41">{}'
                                         '</font></center>'.format(self.err_msg),
                                    anchor=(0.5, 0.5))
            text_item.setTextWidth(500)
            widget.addItem(text_item)

    def redraw_state(self, sample, m_sample):
        if self.video is not None:
            timer = time.time()
            if timer - self.timer > self.timer_period:
                self.timer = timer
                self.frame_counter = (self.frame_counter + 1) % self.n_frames
                self.img.setImage(self.video.get_data(self.frame_counter))
            pass


class FingersProtocolWidgetPainter(Painter):

    def prepare_widget(self, widget):

        newx = 1920
        newy = 1080

        self.widget = widget

        self.widget.setYRange(-newy / 2, newy / 2)
        self.widget.setXRange(-newx / 2, newx / 2)
        self.widget.setMaximumWidth(newx)
        self.widget.setMaximumHeight(newy)

        self.plotItem = widget.plotItem

        self.images = [imread(fin_img_dir_path + '/' + str(image_num) + '.png') for image_num in np.arange(22)]

        self.img = pg.ImageItem(anchor=(0, 0))

        self.img.rotate(-90)
        self.img.setX(-newx / 2)
        self.img.setY(newy / 2)

        self.img.setImage(self.images[0])

        widget.addItem(self.img)

        self.widget = widget

        widget.setBackgroundBrush(pg.mkBrush('#606060'))

    def change_pic(self, num_pic):

        self.img.setImage(self.images[num_pic])

    def redraw_state(self):
        pass

    def set_message(self, text):
        pass

    def goFullScreen(self):
        self.widget.parentWidget().parentWidget().showFullScreen()

        newx = 1920
        newy = 1080
        self.widget.setYRange(-newy / 2, newy / 2)
        self.widget.setXRange(-newx / 2, newx / 2)
        self.widget.setMaximumWidth(newx)
        self.widget.setMaximumHeight(newy)
        self.widget.setMinimumWidth(newx)
        self.widget.setMinimumHeight(newy)


class CenterOutProtocolWidgetPainter(Painter):
    def __init__(self, if_4_targets, if_vanilla_co):
        self.if_4_targets = if_4_targets
        self.if_vanilla_co = if_vanilla_co

        # super(CenterOutProtocolWidgetPainter, self).__init__(show_reward=show_reward)

    def prepare_widget(self, widget):

        self.use_photo_trigger = True

        self.widget = widget
        self.cursor = pg.QtGui.QCursor

        widget.setMouseEnabled(x=False, y=False)

        # center of the outer circles ring, its radius, outer circle radius, center circle radius, segment radius and segment spacing
        self.centerX = 0
        self.centerY = 0
        self.bigR = 150
        self.smallR = 22
        self.centerR = 22

        self.arcOuterR = 45
        self.arcInnerR = 35
        self.arcSpacing = 0
        self.arcArrowWidth = 20
        self.arcPolyPoints = 30

        self.arcSuppDent = np.pi / 360 * 15
        self.arcSuppWidth = 2

        # colors of fixation cross, inactive outer, center, active outer and right guessed outer
        # # # # # COLORS # # # # #
        fullblack = pg.mkBrush('#000000')
        self.darkgray = pg.mkBrush('#333333')
        self.mediumgray = pg.mkBrush('#505050')
        self.lightgray = pg.mkBrush('#999999')
        self.red = pg.mkBrush('#993333')
        self.green = pg.mkBrush('#339933')
        self.bgcolor = fullblack # pg.mkBrush('#303030')

        # =============================================================================
        #         if(self.if_4_targets):
        #             rangelist = 2*np.arange(4)
        #         else:
        #             rangelist = range(8)
        # =============================================================================

        rangelist = range(8)

        self.outerCircles = [
            pg.QtGui.QGraphicsEllipseItem(self.centerX + self.bigR * np.cos(2 * np.pi / 8 * i) - self.smallR,
                                          self.centerY + self.bigR * np.sin(2 * np.pi / 8 * i) - self.smallR,
                                          2 * self.smallR, 2 * self.smallR) for i in rangelist]
        self.cx = [self.centerX + self.bigR * np.cos(2 * np.pi / 8 * i) - self.smallR for i in rangelist]
        self.cy = [self.centerY + self.bigR * np.sin(2 * np.pi / 8 * i) - self.smallR for i in rangelist]

        ArcSpacing = [{'x': self.arcSpacing * np.cos(2 * np.pi / 8 * i + 2 * np.pi / 16),
                       'y': self.arcSpacing * np.sin(2 * np.pi / 8 * i + 2 * np.pi / 16)} for i in range(8)]
        ArcAngle = [np.linspace(2 * np.pi / 8 * i, 2 * np.pi / 8 * (i + 1), self.arcPolyPoints).tolist() for i in
                    range(8)]
        ArcPoints = [[pg.QtCore.QPointF(self.arcOuterR * np.cos(outerArc) + ArcSpacing[i]['x'],
                                        self.arcOuterR * np.sin(outerArc) + ArcSpacing[i]['y']) for outerArc in
                      ArcAngle[i]] +
                     [pg.QtCore.QPointF(self.arcInnerR * np.cos(innerArc) + ArcSpacing[i]['x'],
                                        self.arcInnerR * np.sin(innerArc) + ArcSpacing[i]['y']) for innerArc in
                      ArcAngle[i][::-1]]
                     for i in range(8)]
        ArcPoly = [pg.QtGui.QPolygonF(ArcPoints[i]) for i in range(8)]
        self.arcSegments = [pg.QtGui.QGraphicsPolygonItem(ArcPoly[i]) for i in range(8)]

        arcCentralR = (self.arcOuterR + self.arcInnerR) / 2
        SupportArcAngle = [np.linspace(2 * np.pi / 8 * (i) + self.arcSuppDent, 2 * np.pi - self.arcSuppDent,
                                       self.arcPolyPoints * 8 // (i + 1)).tolist() for i in range(8)] + \
                          [np.linspace(2 * np.pi / 8 * (i) - self.arcSuppDent, 0 + self.arcSuppDent,
                                       self.arcPolyPoints * (i + 1)).tolist() for i in range(8)]
        SupportArcAngle[0] = [0]
        SupportArcAngle[-8] = [0]

        # for i in SupportArcAngle:
        #    print('suplen is',len(i))

        SupportArcPoints = [[pg.QtCore.QPointF((arcCentralR - self.arcSuppWidth) * np.cos(outerArc),
                                               (arcCentralR - self.arcSuppWidth) * np.sin(outerArc)) for outerArc in
                             SupportArcAngle[i]] +
                            [pg.QtCore.QPointF((arcCentralR + self.arcSuppWidth) * np.cos(innerArc),
                                               (arcCentralR + self.arcSuppWidth) * np.sin(innerArc)) for innerArc in
                             SupportArcAngle[i][::-1]]
                            for i in range(len(SupportArcAngle))]

        self.supportArcs = [pg.QtGui.QGraphicsPolygonItem(pg.QtGui.QPolygonF(i)) for i in SupportArcPoints]

        deltagrad = 2 * np.pi * self.arcSpacing / ((self.arcOuterR + self.arcInnerR) / 2 + self.arcSpacing) / 8

        ArcCenter = round(len(ArcAngle[0]) / 2)
        ArcArrowUpPoints = [[pg.QtCore.QPointF(self.arcOuterR * np.cos(outerArc) + +ArcSpacing[i]['x'],
                                               self.arcOuterR * np.sin(outerArc) + ArcSpacing[i]['y']) for outerArc in
                             ArcAngle[i][:ArcCenter:1]] + \
                            [pg.QtCore.QPointF(
                                ((self.arcOuterR + self.arcInnerR) / 2 + self.arcArrowWidth / 2) * np.cos(
                                    ArcAngle[i][ArcCenter]) + ArcSpacing[i]['x'],
                                ((self.arcOuterR + self.arcInnerR) / 2 + self.arcArrowWidth / 2) * np.sin(
                                    ArcAngle[i][ArcCenter]) + ArcSpacing[i]['y']),
                             pg.QtCore.QPointF(
                                 (self.arcOuterR + self.arcInnerR) / 2 * np.cos(ArcAngle[i][-1] + deltagrad) +
                                 ArcSpacing[i]['x'],
                                 (self.arcOuterR + self.arcInnerR) / 2 * np.sin(ArcAngle[i][-1] + deltagrad) +
                                 ArcSpacing[i]['y']),
                             pg.QtCore.QPointF(
                                 ((self.arcOuterR + self.arcInnerR) / 2 - self.arcArrowWidth / 2) * np.cos(
                                     ArcAngle[i][ArcCenter]) + ArcSpacing[i]['x'],
                                 ((self.arcOuterR + self.arcInnerR) / 2 - self.arcArrowWidth / 2) * np.sin(
                                     ArcAngle[i][ArcCenter]) + ArcSpacing[i]['y'])] + \
                            [pg.QtCore.QPointF(self.arcInnerR * np.cos(innerArc) + ArcSpacing[i]['x'],
                                               self.arcInnerR * np.sin(innerArc) + ArcSpacing[i]['y']) for innerArc in
                             ArcAngle[i][ArcCenter - 1::-1]] for i in range(8)]
        ArcArrowUpPoly = [pg.QtGui.QPolygonF(ap) for ap in ArcArrowUpPoints]
        self.arcUpArrow = [pg.QtGui.QGraphicsPolygonItem(ap) for ap in ArcArrowUpPoly]

        ArcArrowDownPoints = [[pg.QtCore.QPointF(arr.x(), -arr.y()) for arr in ap] for ap in ArcArrowUpPoints]
        ArcArrowDownPoly = [pg.QtGui.QPolygonF(ap) for ap in ArcArrowDownPoints]
        self.arcDownArrow = [pg.QtGui.QGraphicsPolygonItem(ap) for ap in ArcArrowDownPoly]

        self.circle = pg.QtGui.QGraphicsEllipseItem(self.centerX - self.centerR, self.centerY - self.centerR,
                                                    2 * self.centerR, 2 * self.centerR)

        self.fixCrossX = pg.QtGui.QGraphicsRectItem(-10, -2, 20, 4)
        self.fixCrossY = pg.QtGui.QGraphicsRectItem(-2, -10, 4, 20)


        self.txt = pg.TextItem(anchor=(0.5, 0.5))

        self.circle.setBrush(self.mediumgray)
        self.circle.setPen(pg.mkPen(None))
        widget.addItem(self.circle)

        for i in rangelist:
            self.outerCircles[i].setBrush(self.darkgray)
            self.outerCircles[i].setPen(pg.mkPen(None))
            widget.addItem(self.outerCircles[i])

            self.arcSegments[i].setBrush(self.lightgray)
            self.arcSegments[i].setPen(pg.mkPen(None))
            widget.addItem(self.arcSegments[i])
            self.arcSegments[i].hide()

            self.arcUpArrow[i].setBrush(self.lightgray)
            self.arcUpArrow[i].setPen(pg.mkPen(None))
            widget.addItem(self.arcUpArrow[i])
            self.arcUpArrow[i].hide()

            self.arcDownArrow[i].setBrush(self.lightgray)
            self.arcDownArrow[i].setPen(pg.mkPen(None))
            widget.addItem(self.arcDownArrow[i])
            self.arcDownArrow[i].hide()

        for i in self.supportArcs:
            i.setBrush(self.lightgray)
            i.setPen(pg.mkPen(None))
            widget.addItem(i)
            i.hide()

        self.fixCrossX.setBrush(fullblack)
        self.fixCrossY.setBrush(fullblack)
        widget.addItem(self.fixCrossX)
        widget.addItem(self.fixCrossY)

        widget.addItem(self.txt)

        self.whiterect = pg.QtGui.QGraphicsRectItem(1920 / 2 - 35-50, 1080 / 2 - 50-50, 200, 200)
        self.whiterect.setBrush(pg.mkBrush('w'))
        self.whiterect.setPen(pg.mkPen(None))
        widget.addItem(self.whiterect)
        self.whiterect.hide()

        newfont = QFont("Calibri", 16, QFont.Bold)
        self.txt.setFont(newfont)
        self.txt.setX(0)
        self.txt.setY(0)
        self.txt.setText('    ')
        widget.addItem(self.txt)

        widget.setBackgroundBrush(self.bgcolor)

        self.prev_par = 0
        self.prev_state = 0

        if (self.if_4_targets):

            for i in 2 * np.arange(4):
                self.outerCircles[i].setBrush(self.darkgray)
                self.outerCircles[i + 1].hide()

        else:

            for i in range(8):
                self.outerCircles[i].setBrush(self.darkgray)

        self.circle.show()

    def goFullScreen(self):
        self.widget.parentWidget().parentWidget().showFullScreen()

        newx = 1920
        newy = 1080
        self.widget.setYRange(-newy / 2, newy / 2)
        self.widget.setXRange(-newx / 2, newx / 2)
        self.widget.setMaximumWidth(newx)
        self.widget.setMaximumHeight(newy)
        self.widget.setMinimumWidth(newx)
        self.widget.setMinimumHeight(newy)

    def getMousePos(self):
        p = self.cursor.pos();
        p = self.widget.scene().views()[0].mapFromGlobal(p)
        self.trueX = (p.x() - 1920 / 2) / 0.917
        self.trueY = (1080 / 2 - p.y()) / 0.917
        # self.txt.setText(str(par))
        # self.txt.setText(str(round(self.trueX))+' '+str(round(self.trueY)))

        return [self.trueX, self.trueY]

    def checkHover(self, x, y):
        angle = int(round((np.arctan2(y, x)) / (2 * np.pi) * 8)) % 8
        if np.square(self.cx[angle] + self.smallR - x) + np.square(self.cy[angle] + self.smallR - y) < np.square(
                self.smallR):
            sw = 1
        else:
            sw = 0

        # self.txt.setText(str(par))

        return angle, sw

    def checkCenterHover(self, x, y):

        if np.square(x) + np.square(y) < np.square(self.centerR):
            sw = 1
        else:
            sw = 0

        # self.txt.setText(str(par))

        return sw

    def showCorrect(self, par, current, switch):
        if switch == 1:
            if par == current:
                self.outerCircles[current].setBrush(self.green)
            else:
                self.outerCircles[current].setBrush(self.red)
        else:
            if not self.if_vanilla_co:
                self.outerCircles[current].setBrush(self.darkgray)

    def doStuff(self, state, par):

        if state == 0:

            if (self.prev_state == 1):

                self.circle.show()

            else:

                if (self.if_4_targets):

                    for i in 2 * np.arange(4):
                        self.outerCircles[i].setBrush(self.darkgray)
                        self.outerCircles[i + 1].hide()

                else:

                    for i in range(8):
                        self.outerCircles[i].setBrush(self.darkgray)

                self.circle.show()

        if self.prev_state == 2:
            if self.prev_par > 0:
                for i in range(self.prev_par - 1):
                    self.arcSegments[i].hide()
                self.arcUpArrow[self.prev_par - 1].hide()
            elif self.prev_par < 0:
                for i in range(9 + self.prev_par, 8):
                    self.arcSegments[i].hide()
                self.arcDownArrow[abs(self.prev_par + 1)].hide()
            else:
                for i in range(7):
                    self.arcSegments[i].hide()
                self.arcUpArrow[7].hide()
            self.whiterect.hide()

            self.supportArcs[self.prev_par].hide()

            self.fixCrossX.show()
            self.fixCrossY.show()

        elif self.prev_state == 1:
            if self.if_vanilla_co:
                if (state == 3):
                    self.circle.hide()
            # else:
            # self.outerCircles[self.prev_par].setBrush(self.darkgray)
            self.whiterect.hide()

            # self.txt.setText(str(self.prev_par))

        elif state == 1:
            self.outerCircles[par].setBrush(self.lightgray)
            if (self.use_photo_trigger):
                self.whiterect.show()
            # self.txt.setText(str(par))

        if state == 2:
            self.fixCrossX.hide()
            self.fixCrossY.hide()
            if (par > 0):
                for i in range(par - 1):
                    self.arcSegments[i].show()
                    self.txt.setText(' ' + str(par * 45) + '°')
                self.arcUpArrow[par - 1].show()

                # self.txt.setText(str(par))
            elif par < 0:
                for i in range(9 + par, 8):
                    self.arcSegments[i].show()
                    self.txt.setText(str(par * 45) + '°')
                self.arcDownArrow[abs(par + 1)].show()

                # self.txt.setText(str(par))
            else:
                for i in range(7):
                    self.arcSegments[i].show()
                    # self.txt.setText(str(par * 45)+'°')
                    self.txt.setText(' ' + str(360) + '°')
                self.arcUpArrow[7].show()

                # self.txt.setText(str(par))
            if (self.use_photo_trigger):
                self.whiterect.show()

            self.supportArcs[par].show()
            # self.txt.setText(str(par))

        elif state == 3:
            self.circle.hide()
            self.txt.setText('   ')
            for i in range(8):
                # if not self.if_vanilla_co:
                # print('TEST')
                # self.outerCircles[i].setBrush(self.darkgray)

                self.arcSegments[i].hide()
                self.arcUpArrow[i].hide()
                self.arcDownArrow[i].hide()

            self.supportArcs[self.prev_par].hide()

            self.whiterect.hide()

            # self.txt.setText(str(par))

    def redraw_state(self):
        pass

    def set_message(self, text):
        # self.text = text
        # self.text_item.setHtml('<center><font size="7" color="#e5dfc5">{}</font></center>'.format(self.text))
        pass



if __name__ == '__main__':
    from PyQt5 import QtGui, QtWidgets
    from PyQt5 import QtCore, QtWidgets
    a = QtWidgets.QApplication([])
    w = ProtocolWidget()
    w.show()
    b = BarFeedbackProtocolWidgetPainter()
    b.prepare_widget(w)
    timer = QtCore.QTimer()
    timer.start(1000/30)
    timer.timeout.connect(lambda: b.redraw_state(np.random.normal(scale=3), np.random.normal(scale=0.1)))
    a.exec_()
    #for k in range(10000):
    #    sleep(1/30)
    #    b.redraw_state(np.random.normal(size=1))


