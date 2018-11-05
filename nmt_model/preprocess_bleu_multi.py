import utils_multi

data, fail1 = utils_multi.read_corpus('/home/anushap/Code-Generation/nmt_model/data/2code/code_test.txt','tgt')
print(fail1)
data1, fail2 = utils_multi.read_corpus('/home/anushap/Code-Generation/nmt_model/data/2code/src_code_test.txt','src_code')
print(fail2)
data2, fail3 = utils_multi.read_corpus('/home/anushap/Code-Generation/nmt_model/data/2code/nl_test.txt','src_nl')
print(fail3)
fout = open("/home/anushap/Code-Generation/nmt_model/data/2code/code_test_bleu.txt", "w")
fail = fail1+fail2+fail3
for i in range(len(data)):
    if i not in fail:
        each = ' '.join(data[i][1:-1])
        fout.write(each)
        fout.write('\n')

fout.close()
