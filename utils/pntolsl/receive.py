from pylsl import StreamInlet, resolve_stream
from collections import deque

print("looking for a stream...")
streams = resolve_stream('type', 'BVH')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
timestamp_vec = []

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    timestamp_vec.append(timestamp)
    if len(timestamp_vec) == 100:
        print(1/((timestamp_vec[99] - timestamp_vec[0])/100))
        timestamp_vec.clear()
    #print(timestamp, sample)