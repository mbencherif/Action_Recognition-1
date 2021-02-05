import os
import sys
import json


def load_labels(dataset_path):
    if os.path.isdir(dataset_path):
        labels = os.listdir(dataset_path)
    return labels


def get_dataset(dataset_path):
    
    assert os.path.exists(dataset_path), IOError(f"{dataset_path} not exists")
    
    train_database = {}
    
    train_fight_data = os.listdir(os.path.join(dataset_path, 'train', 'Fight'))
    train_nonfight_data = os.listdir(os.path.join(dataset_path, 'train', 'NonFight'))
    
    for name in train_nonfight_data:
        # name = os.path.join('train', 'NonFight', file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'training'
        train_database[name]['annotations'] = {'label': 'NonFight'}
    for name in train_fight_data:
        # name = os.path.join('train', 'Fight', file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'training'
        train_database[name]['annotations'] = {'label': 'Fight'}

    val_database = {}
    val_fi_data = os.listdir(os.path.join(dataset_path, 'val', 'Fight'))
    val_no_data = os.listdir(os.path.join(dataset_path, 'val', 'NonFight'))

    for name in val_no_data:
        train_database[name] = {}
        train_database[name]['subset'] = 'validation'
        train_database[name]['annotations'] = {'label': 'NonFight'}
    
    for name in val_fi_data:
        # name = os.path.join('val', 'Fight', file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'validation'
        train_database[name]['annotations'] = {'label': 'Fight'}
    
    return train_database, val_database


def generate_annotation(dataset_path, dst_json):
    labels = load_labels(dataset_path)
    train_dataset, val_dataset = get_dataset(dataset_path)
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_dataset)
    dst_data['database'].update(val_dataset)

    with open(dst_json, 'w') as dst_file:
        json.dump(dst_data, dst_file)
        

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    
    generate_annotation(dataset_path, dataset_path + '.json')