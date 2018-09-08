import json
import pdb
import pickle


def get_data(filename):
    fp = open(filename, 'r')
    data = json.load(fp)
    fp.close()
    return data


def get_content(data):
    content = []
    for cell in data['cells']:
        if cell['cell_type'] == 'markdown':
            continue
        content.extend(cell['source'])
    return content


def get_examples(intents, snippets, data):
    for i in range(len(data['cells']) - 1):
        prev = data['cells'][i]
        current = data['cells'][i + 1]
        if prev['cell_type'] == 'markdown' and current['cell_type'] == 'code':
            intents.append(prev['source'])
            snippets.append(current['source'])

    return intents, snippets


def dump_pickle(filename, obj):
    with open(filename, 'w') as fp:
        pickle.dump(obj, fp)


def main():
    data = get_data('matplotlib/data/712.ipynb')
    intents, snippets = get_examples([], [], data)
    dump_pickle('pickles/intents.pickle', intents)
    dump_pickle('pickles/snippets.pickle', intents)


if __name__ == '__main__':
    main()
