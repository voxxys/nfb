from PyQt5 import QtGui, QtCore, QtWidgets
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet, local_clock
import sys, socket, threading, time
import time
import numpy as np
from collections import deque

from form2 import Ui_Form


class App(Ui_Form):
    def __init__(self, dialog):
        Ui_Form.__init__(self)
        self.setupUi(dialog)

        self.pushButton.clicked.connect(self.connect_ip)
        self.pushButton_2.clicked.connect(self.disconnect)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.clicked.connect(self.refresh)

        self.TCP_IP = '127.0.0.1'
        self.TCP_PORT = 7010
        self.BUFFER_SIZE = 1800
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.t_1 = time.time()

        self.info = StreamInfo('AxisNeuron', 'BVH', 240, 60, 'float32', 'myuid2424')
        self.info.desc().append_child_value("manufacturer", "AxisNeuron")
        channels = self.info.desc().append_child("channels")
        
        chan_names = []

        #for j in np.arange(59):

        # ONLY STREAM HANDS DATA
            # right hand = from 16
            # right fingers = 17-35

            # left hand = from 39
            # left fingers = 40-58

            # seq = 16 #sequence_in_data_block
            # num_ch_in_354 = np.arange(6*seq,6*seq+6) # numch in 354 channel data

        self.ch_idxs = np.hstack((np.arange(96,216),np.arange(234,354)))

        for j in self.ch_idxs: # left and right hand 
            for coord in ['Xpos','Ypos','Zpos','Xrot','Yrot','Zrot']:
                chan_names.append(str(j)+'_'+coord)
    
        for c in chan_names:
            channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "angle") \
                .append_child_value("type", "BVH")


        self.chan_names = chan_names

        self.outlet = StreamOutlet(self.info)
        #self.buff = []
        #self.count = 0


        self.t = QtCore.QTime()
        self.graph_data_1 = deque(maxlen=1000)
        self.graph_data_2 = deque(maxlen=1000)
        self.graph_data_3 = deque(maxlen=1000)
        self.graph_data_4 = deque(maxlen=1000)

        self.thread_1 = threading.Thread(target=self.get_data, args=())
        self.thread_1.daemon = True
        #self.thread_2 = threading.Thread(target=self.show_data, args=())
        #self.thread_2.daemon = True
        #self.thread_3 = threading.Thread(target=self.get_lsl, args=())
        #self.thread_3.daemon = True

        self.counter = 0.0

        self.connect_ip()

    def refresh(self):
        x_vec = [0,0,0,0,0]
        y_vec = [0,0,0,0,0]
        self.p1.setData(x = x_vec, y = y_vec)

        #self.p3.repaint()
        #self.p4.repaint()

    def disconnect(self):
        #self.thread._stop()
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.s.close()

    '''def get_lsl(self):
        streams = resolve_stream('name', 'NVX136_Data')
        inlet = StreamInlet(streams[0])

        while 1:
            chunk, timestamp = inlet.pull_chunk()
            np_ar_chunk = np.asarray(chunk);
            chunk_size = np_ar_chunk.shape[0];

            if chunk_size > 0:
                a = float(np_ar_chunk[-1:,-2])
                self.graph_data_4.append({'x': self.t.elapsed(), 'y': a})
            if len(self.graph_data_4) > 950:
                self.graph_data_4.popleft()

            #print("Timespamp: ", timestamp)
            #print("Chunk: ", np_ar_chunk[-1:,-2])'''

    def connect_ip(self):
        self.pushButton.setEnabled(False)
        #self.pushButton_2.setEnabled(True)
        if self.lineEdit.isModified() & self.lineEdit_2.isModified():
            self.TCP_IP = self.lineEdit.text()
            self.TCP_PORT = self.lineEdit_2.text()
        self.s.connect((self.TCP_IP, self.TCP_PORT))
        print(' ')
        print(" * * * Connected to Perception Neuron data stream * * * ")
        self.thread_1.start()
        #self.thread_2.start()
        #self.thread_3.start()
        self.t.start()

    def get_data(self):
        res = []
        data_full = True
        while 1:
            data_in = self.s.recv(self.BUFFER_SIZE) #Recieve a string of data from Axis Neuron
            if data_full: #If the previous string was full
                if (data_in.decode("utf-8")).find("C") > 0: #If there is a sample beginning
                    index = (data_in.decode("utf-8")).find("C") #Find index for data beginning
                    index2 = (data_in.decode("utf-8")).find("|") #Find indef for data ending
                    if index2 > 0 & index2 > index: #If indexes are correct
                        res = data_in[index + 7:index2 - 2].decode("utf-8") #Store the data
                        res = res.split(" ") #Split the string
                        numbers = [float(i) for i in res] #Create an array with the recieved data
                        self.send_data(numbers) #send_data pushes the array into lsl
                    else: #If the recieved string doesn't contain the complete data sample
                        res = (data_in[index + 7:]).decode("utf-8") #Store the incomplete data
                        data_full = False #Flag that the data is incomplete
            else: #If previously didn't recieve a complete data sample
                if (data_in.decode("utf-8")).find("C") > 0:
                    index = (data_in.decode("utf-8")).find("C") #Find indexes
                    index2 = (data_in.decode("utf-8")).find("|")
                    res1 = data_in[:index - 5].decode("utf-8") #Get the ending for the previous sample
                    res = res + res1 #Get a complete sample from two parts
                    res = res.split(" ")
                    numbers = [float(i) for i in res] #Create an array
                    self.send_data(numbers) #Send the array through lsl
                    if index2 > 0 & index2 > index: #If there is a complete sample
                        res = data_in[index + 7:index2 - 2].decode("utf-8")
                        res = res.split(" ")
                        numbers = [float(i) for i in res]
                        self.send_data(numbers)
                        data_full = True
                    else: #If there isn't
                        res = (data_in[(index + 7):]).decode("utf-8") #Store the beginning of a new sample
                else: #If there is no sample beginning
                    index2 = (data_in.decode("utf-8")).find("|") #Find the ending
                    if index2 > 0: #If it is there
                        res1 = data_in[:index2 - 2].decode("utf-8")
                        res = res + res1
                        res = res.split(" ")
                        numbers = [float(i) for i in res]
                        self.send_data(numbers)
                        data_full = True
                    else:
                        data_full = True

    def send_data(self, numbers):
        joint = self.comboBox.currentIndex()

        self.graph_data_1.append({'x': self.t.elapsed(), 'y': numbers[joint * 6 + 3]})
        self.graph_data_2.append({'x': self.t.elapsed(), 'y': numbers[joint * 6 + 4]})
        self.graph_data_3.append({'x': self.t.elapsed(), 'y': numbers[joint * 6 + 5]})

        if len(self.graph_data_1) > 950:
            self.graph_data_1.popleft()
            self.graph_data_2.popleft()
            self.graph_data_3.popleft()

        #, numbers[joint * 6 + 4], numbers[joint * 6 + 5]
        
        #mysample = numbers[237:240]

        # STREAM DATA FROM HAND ONLY

        # np.hstack((np.arange(96,216),np.arange(234,354)))

        mysample = [numbers[chidx] for chidx in self.ch_idxs]

        #if self.count < 10:
        #    self.buff.append(mysample)
        #    self.count = self.count+1
        #else:
        #    self.outlet.push_chunk(self.buff)
        #    self.count = 0
        #    self.buff.clear()

        self.outlet.push_sample(mysample, self.counter/60)
        self.counter += 1
        #print(self.outlet.channel_count)
        #time.sleep(0.00833)

        t_2 = time.time()
        if t_2 - self.t_1 > 5:
            print(' ')
            print(time.asctime(time.localtime(t_2)))
            self.t_1 = t_2
            print('Stream alive, last sample (left hand): ')
            print(numbers[234:240])


    ''' def show_data(self):
     time.sleep(2)
     i = 0
     while 1:
         time.sleep(0.017)
         x4 = [item['x'] for item in list(self.graph_data_4)]
         y4 = [item['y'] for item in list(self.graph_data_4)]
         self.p1.setData(x=[item['x'] for item in list(self.graph_data_1)],
                             y=[item['y'] for item in list(self.graph_data_1)])
         self.p2.setData(x=[item['x'] for item in list(self.graph_data_2)],
                             y=[item['y'] for item in list(self.graph_data_2)])
         self.p3.setData(x=[item['x'] for item in list(self.graph_data_3)],
                             y=[item['y'] for item in list(self.graph_data_3)])
          self.p4.setData(x = x4, y = y4)'''


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog = QtWidgets.QDialog()

    prog = App(dialog)

    #dialog.show()
    sys.exit(app.exec_())

'''
                    if res.find("a") > 0:
                        print("hello")
                    if res.find("r") > 0:
                        print("hello")
                    if res.find("h") > 0:
                        print("hello")'''