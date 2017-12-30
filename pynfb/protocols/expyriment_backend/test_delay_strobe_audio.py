from pynfb.inlets.channels_selector import ChannelsSelector
from pynfb.inlets.lsl_inlet import LSLInlet
from pylsl import local_clock

import numpy as np
from time import time, sleep
import pylab as plt
from pynfb.protocols.expyriment_backend.window import ExpyrimentWindow, BinaryBlinkBlock

events_stream = LSLInlet('NVX136_Events')
data_stream = LSLInlet('NVX136_Data')

stream = ChannelsSelector(data_stream, events_inlet=events_stream)

fs = stream.get_frequency()
n_samples = int(60 * fs)
buffer = np.zeros((n_samples + 100, 5))

exp = ExpyrimentWindow()
tone = exp.stimuli.Tone(5000)
tone.preload()


counter = 0
t = time()
n = 0
clocks = []
tone.present()
while counter < n_samples:
    #sleep(0.01)
    chunk, _, timestamps = stream.get_next_chunk()
    if chunk is not None and len(chunk)>0:
        n = len(chunk)
        counter += n
        buffer[counter - n:counter, 0] = timestamps
        buffer[counter - n:counter, 1] = chunk[:, -1]
        buffer[counter - n:counter, 2] = 0
        buffer[counter - n:counter, 3] = 0
        buffer[counter - n:counter, 4] = chunk[:, 0]
        buffer[counter - n, 3] = 1
        buffer[counter-1, 3] = 2

#        if time() - t > 5:
#            buffer[counter - 1, 2] = 1
#
        #    clocks.append(local_clock() + stream.inlet.inlet.time_correction())
        #    t = time()
        #    block.update(True)
        if max(chunk[:, 0]) < 0.0002 :
            tone.present()
            t = time()

        #if time() - t > 1:
            #block.update(False)

exp.close()
plt.plot(buffer[:counter, 0], buffer[:counter, 1])
plt.plot(buffer[:counter, 0], buffer[:counter, 2],)
plt.plot(buffer[:counter, 0], buffer[:counter, 4],)
plt.plot(buffer[:counter, 0], buffer[:counter, 3], 'o')
#plt.plot(clocks, np.zeros_like(clocks), 'o')
plt.show()