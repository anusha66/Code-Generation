import pdb
import json
import glob
import pickle
import tokenize
from io import StringIO
from collections import defaultdict
import sys

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


def get_cell_data(filename):
    fp = open(filename, 'r')
    try:
        data = json.load(fp)
    except:
        tp = open('../failed_files.txt', 'a')
        tp.write(filename)
        tp.write('\n')
        tp.close()
        data = None
    fp.close()
    return data


# key is cell id of code, value is its relevant cells
def get_content(data):
    content = defaultdict(dict)
    m_ids = []
    code_ids = []
    try:
        for i, cell in enumerate(data['cells']):
            if cell['cell_type'] == 'markdown':
                m_ids.append(i)
            elif cell['cell_type'] == 'code':
                content[i]['code'] = cell['source']
                content[i]['prev_md'] = m_ids[:]
                content[i]['prev_code'] = code_ids[:]
                begin = 0
                comments = []
                for line in cell['source']:
                    val, begin = get_comments(line, begin)
                    if val[1] == 1:
                        comments.append(val[0])
                content[i]['comments'] = comments
                code_ids.append(i)
            else:
                continue

    except:
        return dict()

    return content


def dump_pickle(filename, obj):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def main():
    dataset = []

    path = '../data/*'
    files = glob.glob(path)
    # give context as argument 
    context = int(sys.argv[1])

    for file in files:

        print(file, " is being processed")
        cell_info = get_cell_data(file)
        data = get_content(cell_info)
        for key in data:
            item = data[key]
            if len(item['prev_code']) and len(item['prev_md']):
                prev_code_lines = []
                prev_mark_up = []
                for i in range(1, context+1):
                    if i <= len(item['code']):
                        idx = item['prev_code'][-i]
                        prev_code_lines.extend(cell_info['cells'][idx]['source'])

                    if i <= len(item['prev_md']):
                        idx = item['prev_md'][-i]
                        prev_mark_up.extend(cell_info['cells'][idx]['source'])

                dataset.append((item['code'], prev_code_lines, prev_mark_up))

        print('Finished ', file)

    dump_pickle('matplotlib_dataset_2009.pkl', dataset)


if __name__ == '__main__':
    main()
    print ('Done')
