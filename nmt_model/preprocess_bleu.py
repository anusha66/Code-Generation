import utils

data, fail1 = utils.read_corpus('/home/anushap/Code-Generation/nmt_model/data/code2code/code_test.txt','tgt')
print(fail1)
data1, fail2 = utils.read_corpus('/home/anushap/Code-Generation/nmt_model/data/code2code/nl_test.txt','src')
print(fail2)
fout = open("data/code_test_bleu.txt", "w")
fail = fail1+fail2
for i in range(len(data)):
    if i not in fail:
        each = ' '.join(data[i][1:-1])
        fout.write(each)
        fout.write('\n')

fout.close()
