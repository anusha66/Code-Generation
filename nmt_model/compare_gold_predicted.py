
file1=open('work_dir_2code/decode.txt','r')
file2=open('data/2code/code_test_bleu.txt','r')

c=0

f1 = file1.readlines()
f2 = file2.readlines()

k = 1

for line1, line2 in zip(f1, f2):
   if line1==line2:
     c = c+1
     print(line1, k)
   k = k+1
   
print(c)
print(c/len(f1))

