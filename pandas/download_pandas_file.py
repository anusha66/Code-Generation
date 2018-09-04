import os
import pdb

f = open('url_list_pandas.txt', 'r')

lines = f.readlines()
count = 1

token = "3690b2d4b0ed487434b395cb9096ff7160ca6c7f"

'''
curl -H "Accept:application/vnd.github.v3.raw" -H "Authorization:token 3197c481f056ee1e1126db079e2c814c0e48e9cf" -O -L https://api.github.com/repos/micksterct/flask-play-1/contents/notebooks/excel-play.ipynb
'''

for line in lines:

    print(line)

    new = line.split('blob')
    t1 = new[0].split('/')
    t2 = new[1].split('/')

    path = '/'.join(t2[2:])
    print(path)

    os.system('curl -H "Accept:application/vnd.github.v3.raw" -H "Authorization:token 3197c481f056ee1e1126db079e2c814c0e48e9cf" -o '+str(count)+'".ipynb" -L https://api.github.com/repos/'+t1[3]+"/"+t1[4]+"/contents/"+path)
    #+t2[2]+"/"+t2[3]
    count = count + 1

f.close()
