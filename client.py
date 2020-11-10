# Echo client program
import socket
import time

HOST = ''    # The remote host
PORT = 50008              # The same port as used by the server


count = 0

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
	count += 1
	d = bytes("hello" + str(count), 'utf-8')
	s.sendall(d)
	data = s.recv(1024)
	print('Received', repr(data))


	time.sleep(0.1)
