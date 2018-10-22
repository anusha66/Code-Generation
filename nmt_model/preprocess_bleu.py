import utils

data, fail = utils.read_corpus('/home/anushap/Code-Generation-Old/data/code_dev.txt','tgt')
fout = open("/home/anushap/Code-Generation-Old/data/code_dev_bleu.txt", "w")

for i in range(len(data)):

	each = ' '.join(data[i][1:-1])
	fout.write(each)
	fout.write('\n')

fout.close()
