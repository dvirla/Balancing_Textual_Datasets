import pandas as pd
import tensorflow as tf

from T5.fine_tuning_T5.preprocessors import fill_in_the_blank


def replace_X_with_extra_id(text):
    i = 1
    new_text = []
    for word in text.split():
        if word == 'X':
            new_text.append(f'<extra_id_{i}>')
            i +=1
            if i > 100:
                break
        else:
            new_text.append(word)
    return ' '.join(new_text)


def _fill_in_the_blank(texts):
    dataset = tf.data.Dataset.from_tensor_slices({'text': texts})
    dataset = fill_in_the_blank(dataset)
    input_ids_list= []
    labels_list = []
    for example in dataset:
        inputs = replace_X_with_extra_id(str(example['inputs'].numpy().decode("utf-8") ))
        labels = replace_X_with_extra_id(str(example['targets'].numpy().decode("utf-8") ))
        input_ids_list.append(inputs)
        labels_list.append(labels)
    return pd.DataFrame({'input_ids': input_ids_list, 'labels': labels_list})
        
        
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
frame = pd.read_pickle('/home/vmadmin/StudentData/Regular/train_df.pkl')

for ind in [0, 1, 2, 3, 4]:
    print(ind)
    frame_c = frame[frame['industry'] == f'__label__{ind}']
    texts = frame_c['text'].values.tolist()

    labeled_df = _fill_in_the_blank(texts)
    labeled_df.to_pickle(f'/home/vmadmin/StudentData/Generative/TrainingDatasets/train_set_{ind}.pkl')

