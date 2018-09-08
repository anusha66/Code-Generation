import json
import pdb
import pickle


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
    data = get_data('matplotlib/data/712.ipynb')
    code_2_intent = get_examples(data)
    content = get_content(data)
    print('Done !')


if __name__ == '__main__':
    main()
