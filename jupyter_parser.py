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

            return (token[1], 1), begin

        else:

            return (token[4], 0), begin


def get_data(filename):
    fp = open(filename, 'r')
    try:
        data = json.load(fp)
    except:
        tp = open('failed_files.txt', 'a')
        tp.write(filename)
        tp.write('\n')
        tp.close()
        data = None
    fp.close()
    return data


# key is cell id of code, value is its previous markdown value
def get_content(data):
    content = dict()
    try:
        for i, cell in enumerate(data['cells']):
            if cell['cell_type'] == 'markdown':
                continue
            content[i] = cell['source']
    except:
        return False

    return content


def get_examples(data):
    code_2_intent = dict()
    try:
        src_string = []
        for i in range(len(data['cells']) - 1):
            current = data['cells'][i]
            next_ = data['cells'][i + 1]
            if current['cell_type'] == 'markdown' and next_['cell_type'] == 'code':
                src_string.extend(current['source'])
                code_2_intent[i+1] = src_string
                src_string = []

            elif current['cell_type'] == 'markdown' and next_['cell_type'] == 'markdown':
                src_string.extend(current['source'])

            else:
                src_string = []
    except:
        return False

    return code_2_intent


def dump_pickle(filename, obj):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def main():
    dataset = []

    path = '/home/anushap/git-downlaod/pandas_notebooks/*'
    #path = '/home/spothara/Code-Generation/matplotlib/data/*'
    files = glob.glob(path)

    for file in files:

        print(file)

        begin = 0

        data = get_data(file)
        if data is None:
            continue

        code_2_intent = get_examples(data)  # closest intent

        if code_2_intent is False:
            continue

        content = get_content(data)

        if content is False:
            continue
        # print(code_2_intent)

        # print(content)

        comments_dic = {}
        code_dic = {}

        for k, v in content.items():

            comments = []
            code = []

            for i in range(len(v)):

                val, begin = get_comments(v[i], begin)

                if val[1] == 1:
                    comments.append(val[0])
                else:
                    code.append(val[0])

            code_dic[k] = code

            comments_dic[k] = comments

        # print(comments_dic)

        # print("-------------\n")

        # print(code_dic)
        
        for key in code_2_intent.keys():
            intent = code_2_intent.get(key)
            comment = comments_dic.get(key)

            intent.extend(comment)

            comments_dic[key] = intent
        
        # print("-------------\n")

        # print(comments_dic)

        for k in code_dic.keys():
            if code_dic.get(k) == [] or comments_dic.get(k) == []:
               continue
            if (code_dic.get(k), comments_dic.get(k)) in dataset:
               continue
            dataset.append((code_dic.get(k), comments_dic.get(k)))

    dump_pickle('pandas_dataset.pkl', dataset)

    print('Done !')


if __name__ == '__main__':
    main()
