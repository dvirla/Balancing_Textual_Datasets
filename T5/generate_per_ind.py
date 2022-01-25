from tqdm import tqdm
import pickle
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch


ind_names = ['package/freight delivery', 'animation', 'paper & forest products', 'international affairs', 'wireless']
tokenizer = AutoTokenizer.from_pretrained("t5-base")
with open('/home/vmadmin/StudentData/Generative/GenrationInputs/GenerationInputDicts/model_inputs_per_ind.pkl',
          'rb') as f:
    model_inputs_per_ind = pickle.load(f)
batch_size = 16

for ind_num, ind_name in enumerate(ind_names):
    print(f"Started {ind_name} generation process")
    model = T5ForConditionalGeneration.from_pretrained(f'/home/vmadmin/StudentData/Generative/Models/model_{ind_num}.hfm')
    model = model.to(device='cuda')
    ind_input = model_inputs_per_ind[ind_name]
    inputs = tokenizer(ind_input, return_tensors="pt", padding=True)

    if ind_num != 0:
        generated_list = []
        with open(f'/home/vmadmin/StudentData/Generative/GeneratedOutput/generated_list_{ind_num}.pkl', 'wb') as f:
            pickle.dump(generated_list, f)
    else:
        with open(f'/home/vmadmin/StudentData/Generative/GeneratedOutput/generated_list_{ind_num}.pkl', 'rb') as f:
            generated_list = pickle.load(f)
    num_batches = inputs['input_ids'].shape[0] // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_batches//2)):
            # if i < 27:
            #     continue
            start_idx = i*batch_size + 1
            if start_idx + batch_size >= len(inputs['input_ids']):
                with open(f'/home/vmadmin/StudentData/Generative/GeneratedOutput/generated_list_{ind_num}.pkl', 'wb') as f:
                    pickle.dump(generated_list, f)
                break
            outputs = model.generate(
                input_ids=inputs['input_ids'][start_idx:start_idx + batch_size].to(device='cuda'),
                attention_mask=inputs['attention_mask'][start_idx:start_idx + batch_size].to(device='cuda'),
                do_sample=False,
                num_beams=4, no_repeat_ngram_size=2, min_length=200, max_length=300
            )
            generated_list += tokenizer.batch_decode(outputs.to(device='cpu'), skip_special_tokens=True)
            if i%10 == 0:
                with open(f'/home/vmadmin/StudentData/Generative/GeneratedOutput/generated_list_{ind_num}.pkl', 'wb') as f:
                    pickle.dump(generated_list, f)
    print("======================================================")
