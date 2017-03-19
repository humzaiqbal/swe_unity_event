import os
def get_iterations(filename):
	no_extension = filename.split('.')[0]
	num_iterations = no_extension.split('model')[1]
	return num_iterations + '.txt'

def get_file():
	for filename in os.listdir(os.getcwd()):
		if filename.endswith('.h5'):
			return get_iterations(filename)
