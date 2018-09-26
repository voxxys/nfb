from PyQt5 import QtGui, QtCore, QtWidgets
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet, local_clock
import sys, socket, threading, time
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

        self.info = StreamInfo('AxisNeuron', 'BVH', 354, 60, 'float32', 'myuid2424')
        self.info.desc().append_child_value("manufacturer", "AxisNeuron")
        channels = self.info.desc().append_child("channels")
        
        chan_names = []
        for j in np.arange(59):
            for coord in ['Xpos','Ypos','Zpos','Xrot','Yrot','Zrot']:
                chan_names.append(str(j)+'_'+coord)
    
        for c in chan_names:
            channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "angle") \
                .append_child_value("type", "BVH")

        self.outlet = StreamOutlet(self.info)

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
        data_full = True
        while 1:
            data_in = self.s.recv(self.BUFFER_SIZE)
            
            if data_full:
                if (data_in.decode("utf-8")).find("C") > 0:
                    index = (data_in.decode("utf-8")).find("C")
                    index2 = (data_in.decode("utf-8")).find("|")
                    if index2 > 0 & index2 > index:
                        res = data_in[index + 7:index2 - 2].decode("utf-8")
                        res = res.split(" ")
                        numbers = [float(i) for i in res]
                        self.calc_data(numbers)
                    else:
                        res = (data_in[index + 7:]).decode("utf-8")
                        data_full = False
            else:
                if (data_in.decode("utf-8")).find("C") > 0:
                    index = (data_in.decode("utf-8")).find("C")
                    index2 = (data_in.decode("utf-8")).find("|")
                    res1 = data_in[:index - 5].decode("utf-8")
                    res = res + res1
                    if res.find("a") > 0:
                        print("hello")
                    if res.find("r") > 0:
                        print("hello")
                    if res.find("h") > 0:
                        print("hello")
                    res = res.split(" ")
                    numbers = [float(i) for i in res]
                    self.calc_data(numbers)
                    if index2 > 0 & index2 > index:
                        res = data_in[index + 7:index2 - 2].decode("utf-8")
                        res = res.split(" ")
                        numbers = [float(i) for i in res]
                        self.calc_data(numbers)
                        data_full = True
                    else:
                        res = (data_in[(index + 7):]).decode("utf-8")
                else:
                    index2 = (data_in.decode("utf-8")).find("|")
                    if index2 > 0:
                        res1 = data_in[:index2 - 2].decode("utf-8")
                        res = res + res1
                        res = res.split(" ")
                        numbers = [float(i) for i in res]
                        self.calc_data(numbers)
                        data_full = True
                    else:
                        data_full = True
                        #res1 = data_in.decode("utf-8")
                        #res = res + res1
            #data_old = data_in

    def calc_data(self, numbers):
        joint = self.comboBox.currentIndex()
		#print(numbers[joint*6+3:joint*6+6])

        self.graph_data_1.append({'x': self.t.elapsed(), 'y': numbers[joint * 6 + 3]})
        self.graph_data_2.append({'x': self.t.elapsed(), 'y': numbers[joint * 6 + 4]})
        self.graph_data_3.append({'x': self.t.elapsed(), 'y': numbers[joint * 6 + 5]})

        if len(self.graph_data_1) > 950:
            self.graph_data_1.popleft()
            self.graph_data_2.popleft()
            self.graph_data_3.popleft()

        #, numbers[joint * 6 + 4], numbers[joint * 6 + 5]
        
        #print(len(numbers))
        #mysample = numbers[78:216]
        mysample = numbers[:]
        
        #print(mysample)
        self.outlet.push_sample(mysample)
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