import os

file = open('list_test_seq_all.txt','w')
num = 0
for line in open('list_mars_test.txt'):
	line = line.split()
	line[0] = line[0][0:-8]
	num = num + 1
	print num, line[0], line[1]
	line = '%s %s\n'%(line[0], line[1])
	file.write(line)
file.close()