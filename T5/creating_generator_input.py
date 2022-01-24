from transformers import T5ForConditionalGeneration, AutoTokenizer
import pickle
import pandas as pd

with open('/home/vmadmin/StudentData/Generative/GenrationInputs/GenerationInputDicts/sentences_per_ind.pkl', 'rb') as f:
    sentences_per_ind = pickle.load(f)
    
with open('/home/vmadmin/StudentData/Generative/GenrationInputs/GenerationInputDicts/kw_per_ind.pkl', 'rb') as f:
    kw_per_ind = pickle.load(f)

def create_inputs(ind_name, amount, sentences_per_ind, kw_per_ind, kw_p=0.5):
    inputs = []
    for _ in range(amount):
        use_kw = random.random() > kw_p
        basic_input = f"summarize: {ind_name.replace('_', ' ').replace('/', ' ')}. "
        if use_kw:
            kws = random.choices(kw_per_ind[ind_name], k=2)
            _input = basic_input + ' '.join(kws)
        else:
            sentence = random.choice(sentences_per_ind[ind_name])
            _input = basic_input + sentence
        inputs.append(_input)
    return inputs
    
ind_names = ['package/freight delivery', 'animation', 'paper & forest products', 'international affairs', 'wireless']
model_inputs_per_ind = {}
max_ind_size = max(ind_size_dict.values())

for ind_num, ind_name in enumerate(ind_names):
    ind_size = ind_size_dict[f'__label__{ind_num}']
    amount = max_ind_size - ind_size
    model_inputs_per_ind[ind_name] = create_inputs(ind_name, amount, sentences_per_ind, kw_per_ind, kw_p=0.5)
    
with open('/home/vmadmin/StudentData/Generative/GenrationInputs/GenerationInputDicts/model_inputs_per_ind.pkl', 'wb') as f:
    pickle.dump(model_inputs_per_ind, f)