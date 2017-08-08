
import numpy as np
import os
import math


def convertToDist():
	paths = '/Users/avanisharma/Masters/2nd_Sem/Research Project/HCP/HCP_untransformed_corrmat/'

	for filename in os.listdir(paths):
		f = os.path.join(paths, filename)
		file = open(f, 'r').readlines()
		print(len(file))
		file2 = open('/Users/avanisharma/Masters/2nd_Sem/Research Project/HCP/HCP_untransformed_distmat/'+filename[:-4]+'dist.txt', 'w')
		matrix = []
		for line in file:
			line = line.split(',')
			vec = []
			for num in line:
				#print num
				num_ = math.sqrt(1-float(num))
				vec = vec + [num_]
			#print(len(vec))
			#exit()
			matrix = matrix + [vec]
		di = np.diag_indices(361)
		#print di
		matrix = np.array(matrix)
		matrix[di] = 0.0
		#print matrix
		matrix = matrix.tolist()
		#print len(matrix[0])
		#exit()

		for row in matrix:
			file2.write(','.join([str(val) for val in row]) + '\n')

		print "Done Writing"

convertToDist()

