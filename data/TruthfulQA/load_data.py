import pandas as pd

induce_data_path = 'data/TruthfulQA/TruthfulQA_train.csv'
eval_data_path = 'data/TruthfulQA/TruthfulQA_test.csv'
all_data_path = 'data/TruthfulQA/TruthfulQA.csv'

def load_data(type):
    if type == 'induce':
        base_path = induce_data_path
    elif type == 'eval':
        base_path = eval_data_path
    else:
        base_path = all_data_path
    path = base_path
    with open(path, 'r') as f:
        data = pd.read_csv(f)
        # print the column names
        input_, output_ = data['Question'], data['Best Answer']
    # Convert to list
    input_ = input_.tolist()
    output_ = output_.tolist()
    return input_, output_
