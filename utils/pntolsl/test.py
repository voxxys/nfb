
import socket

TCP_IP = '127.0.0.1'
TCP_PORT = 7001
BUFFER_SIZE = 2000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP,TCP_PORT))

data_full = True


while 1:
    #data_new = s.recv(BUFFER_SIZE)
    data_in = s.recv(BUFFER_SIZE)
    if len(data_in) < BUFFER_SIZE:
        print(data_in)
    data_old = data_in
    '''if data_full:
        index = (data_in.decode("utf-8")).find("Char")
        res = ((data_in[(index + 7):]).decode("utf-8")).split(" ")
        if res[len(res) - 1] == "||":
            numbers = [float(i) for i in res[:len(res) - 1]]
        else:
            data_full = False
            numbers_first = [float(i) for i in res[:len(res)]]
    else:
        index = (data_in.decode("utf-8")).find("Char")
        res1 = ((data_in[:(index)]).decode("utf-8")).split(" ")
        numbers_second = [float(i) for i in res1[:len(res1) - 2]]
        numbers = numbers_first + numbers_second
        res2 = ((data_in[(index + 7):]).decode("utf-8")).split(" ")
        numbers_first = [float(i) for i in res2[:len(res2)]]

    print ("data recieved: ", numbers)'''



s.close()



