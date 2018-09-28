
import matplotlib.pyplot as plt

from pylsl import StreamInlet, resolve_stream
import socket

from IPython import display

TCP_IP = '127.0.0.1'
TCP_PORT = 7001
BUFFER_SIZE = 4096

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP,TCP_PORT))

import time
import numpy as np

start = time.time()
print("hello")
end = time.time()
print(end - start)

srate = 120
dt = 1/srate

srate_2 = 500
dt_2 = 1/srate_2

inlet = []

# first resolve an EEG stream on the lab network

streams = resolve_stream('name', 'NVX136_Data')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
print("stream found")

exp_time = 20
buffer = np.zeros(10*exp_time*srate*2)
pos = 0
win = 5
win_samples = win*srate

buffer_2 = np.zeros(10*exp_time*srate_2*2)
pos_2 = 0
win_samples_2 = win*srate_2

buffer_3 = np.zeros(10*exp_time*srate_2*2)


m = 0
#while 1:

num = 6*39+4

fig = plt.figure()


tic = time.time()
toc = tic
while (toc-tic<15):#m<1000:    
    
    chunk, timestamp = inlet.pull_chunk()
    np_ar_chunk = np.asarray(chunk);
    chunk_size = np_ar_chunk.shape[0];
    
    if chunk_size > 0:
        
        data = s.recv(BUFFER_SIZE)
        #print ("data recieved: ", data)
        search = True
        i = 9
        k = 0
        while search:
            if data[i] == 32:
                k = k+1
            if k == num:
                l = i
                k = k+1
            if k == (num+2):
                search = False
            i = i+1
        Pinky3 = float(data[l:i])
        
        
        buffer_2[pos_2:(pos_2+chunk_size)] = np_ar_chunk[:,34].T;
        buffer_3[pos_2:(pos_2+chunk_size)] = Pinky3*np.ones(np_ar_chunk[:,34].T.shape);
        pos_2 = pos_2 + chunk_size
        
    
        
##        if((pos > win_samples)&(pos_2 > win_samples_2)):
#        if(pos_2 > win_samples_2):
#
##             xax = np.arange((pos - win_samples)/srate,pos/srate,dt);
##             plt.clf() 
##             plt.plot(xax,buffer[(pos - win_samples):pos]);
#
#            xax_2 = np.arange((pos_2 - win_samples_2)/srate_2,pos_2/srate_2,dt_2);
#            xax_2 = xax_2[:win_samples_2]
#            plt.clf() 
#            plt.plot(xax_2,buffer_2[(pos_2 - win_samples_2):pos_2]);
#            plt.plot(xax_2,buffer_3[(pos_2 - win_samples_2):pos_2]);
#            
##            display.display(plt.gcf())
##            display.clear_output(wait=True)
#        
#             # plt.xlim((pos - win_samples,pos))
#            
#            plt.show(False)
#            plt.draw()
#            plt.pause(0.005)
#            fig.canvas.draw()
#            #time.sleep(0.01)
            
            
    #time.sleep(0.01)
    toc = time.time()


fig = plt.figure()

xax_2 = np.arange((pos_2 - win_samples_2)/srate_2,pos_2/srate_2,dt_2);
xax_2 = xax_2[:win_samples_2]
    
plt.clf() 
plt.plot(xax_2,buffer_2[(pos_2 - win_samples_2):pos_2]*100);
plt.plot(xax_2,buffer_3[(pos_2 - win_samples_2):pos_2]);


plt.show(False)
plt.draw()
fig.canvas.draw()
    
    
#        
#        
#    data = s.recv(BUFFER_SIZE)
#    #print ("data recieved: ", data)
#    search = True
#    i = 9
#    k = 0
#    while search:
#        if data[i] == 32:
#            k = k+1
#        if k == num:
#            l = i
#            k = k+1
#        if k == (num+2):
#            search = False
#        i = i+1
#    Pinky3 = float(data[l:i])
#    #print("Pinky data: ", Pinky3)
#    
#    buffer[pos] = Pinky3
#    pos = pos + 1
#    m = m + 1
#
#
#    if(pos > win_samples):
#        xax = np.arange((pos - win_samples),pos);
#        plt.clf() 
#        plt.plot(xax,buffer[(pos - win_samples):pos]);
#        plt.xlim((pos - win_samples,pos))
#        plt.draw();
#        plt.show()
#        
#        
#
#s.close()

print(toc-tic)
    

    
    


