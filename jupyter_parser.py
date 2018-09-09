import pdb
import json
import glob
import pickle
import tokenize
from io import StringIO

def get_comments(content, begin):

    if ("'''" in content or '"""' in content) and begin == 0:

            begin = 1

            return (content, 1), begin

    elif ("'''" in content or '"""' in content) and begin == 1:

            begin = 0

            return (content, 1), begin

    elif begin == 1:

            return (content, 1), begin

    tokenizer = tokenize.generate_tokens(StringIO(content).readline)

    for token in tokenizer:

        if token[0] == tokenize.COMMENT:

            return (token[1],1), begin

        else:

            return (token[4],0), begin


def get_data(filename):
    fp = open(filename, 'r')
    data = json.load(fp)
    fp.close()
    return data


# key is cell id of code, value is its previous markdown value
def get_content(data):
    content = dict()
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'markdown':
            continue
        content[i] = cell['source']
    return content


def get_examples(data):
    code_2_intent = dict()
    for i in range(len(data['cells']) - 1):
        prev = data['cells'][i]
        current = data['cells'][i + 1]
        if prev['cell_type'] == 'markdown' and current['cell_type'] == 'code':
            code_2_intent[i+1] = prev['source']

    return code_2_intent


def dump_pickle(filename, obj):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def main():

    dataset = []

    path = '/home/anushap/git-downlaod/pandas_notebooks/*'
    files = glob.glob(path)

    for file in files:

        print(file)

        begin = 0

        data = get_data(file)
        code_2_intent = get_examples(data) #closest intent
        content = get_content(data)

        #print(code_2_intent)

        #print(content)

        comments_dic = {}
        code_dic = {}

        for k, v in content.items():

            comments = []
            code = []

            for i in range(len(v)):

                val, begin = get_comments(v[i],begin)

                if val[1] == 1:
                    comments.append(val[0])
                else:
                    code.append(val[0])

            code_dic[k] = code

            comments_dic[k] = comments

        #print(comments_dic)

        #print("-------------\n")

        #print(code_dic)

        for key in code_2_intent.keys():

            intent = code_2_intent.get(key)
            comment = comments_dic.get(key)

            intent.extend(comment)

            comments_dic[key] = intent

        #print("-------------\n")

        #print(comments_dic)

        for k in code_dic.keys():

            dataset.append((code_dic.get(k),comments_dic.get(k)))

    dump_pickle('pandas_dataset.pkl',dataset)

    print('Done !')




if __name__ == '__main__':
    main()
