import utils

data, fail = utils.read_corpus('/home/anushap/Code-Generation/nmt_model/data/nl2code/code_test.txt','tgt')
fout = open("/home/anushap/Code-Generation/nmt_model/data/nl2code/code_test_bleu.txt", "w")

for i in range(len(data)):

	each = ' '.join(data[i][1:-1])
	fout.write(each)
	fout.write('\n')

fout.close()
