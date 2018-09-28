
import socket

TCP_IP = '127.0.0.1'
TCP_PORT = 7010
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

tic = time.time()

exp_time = 20
buffer = np.zeros(10*srate*2)
pos = 0
win = 2
win_samples = 5*srate

m = 0
#while 1:

num = 10

import matplotlib.pyplot as plt
fig = plt.figure()

while m<1000:    
    
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
    print(data)
    #Pinky3 = float(data[l:i])
    #print("Pinky data: ", Pinky3)
    
    #buffer[pos] = Pinky3
    #pos = pos + 1
    #m = m + 1
    
    #if(pos > win_samples):
    #    xax = np.arange((pos - win_samples),pos);
    #    plt.clf() 
    #    plt.plot(xax,buffer[(pos - win_samples):pos]);
    #    plt.xlim((pos - win_samples,pos))
    #    plt.draw();
    #    plt.show()

    toc = time.time()

s.close()

print(toc-tic)
    

    
    


