import time

class ExpyrimentWindow:
    def __init__(self):
        import expyriment
        experiment = expyriment.design.Experiment()
        self.stimuli = expyriment.stimuli
        self.control = expyriment.control
        self.control.defaults.initialize_delay = 0
        self.control.defaults.window_mode = False
        self.control.defaults.auto_create_subject_id = True
        self.control.initialize(experiment)

    def open(self):
        self.control.start(skip_ready_screen=True)

    def close(self):
        self.control.end()

class BinaryBlinkBlock:
    def __init__(self, exp: ExpyrimentWindow):
        self.rect = exp.stimuli.Rectangle((700, 700), colour=(255, 255, 255))
        self.blank = exp.stimuli.BlankScreen()
        self.rect.preload()
        self.blank.preload()

    def update(self, sample):
        if sample:
            self.rect.present()
        else:
            self.blank.present()


class SoundBlinkBlock:
    def __init__(self, exp: ExpyrimentWindow):
        self.rect = exp.stimuli.Rectangle((1000, 1000), colour=(255, 255, 255))
        self.blank = exp.stimuli.BlankScreen()
        self.tone = exp.stimuli.Tone(10)
        self.rect.preload()
        self.blank.preload()
        self.tone.preload()

    def update(self, sample):
        self.tone.present()
        if sample:
            self.rect.present()
        else:
            self.blank.present()

class TextBlock:
    def __init__(self, exp: ExpyrimentWindow, text: str):
        self.text = exp.stimuli.TextLine(text)
        self.text_not_presented = True

    def update(self, sample):
        if self.text_not_presented:
            self.text.present()
            self.text_not_presented = False

if __name__ == '__main__':
    exp = ExpyrimentWindow()
    block = SoundBlinkBlock(exp)
    text = TextBlock(exp, 'KEK')
    exp.open()
    for k in range(1000):
        time.sleep(0.01)
        text.update(0)
    for k in range(1000):
        time.sleep(0.01)
        block.update(k % 2)
    exp.close()
