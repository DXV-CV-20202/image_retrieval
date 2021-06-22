with open('confusion_matrix.txt') as f:
	lines = f.read()
	x = []
	for i in range(len(lines) - 1):
		if (lines[i] == ' ') and (lines[i+1] != ' '):
			x.append(',')
		else:
			x.append(lines[i])
	x.append(lines[-1])
	print(''.join(x))