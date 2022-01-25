# Restoring Balance to the Universe
### Balancing datasets using T5
***By Eitan Bentora & Dvir Lafer***
<br>
This is a guide to use 3 different methods in order to overcome imbalanced **text** datasets:
1. Undersampling - Randomly selecting records from every industry we want to undersample and removing all others (up to a certain percentile).
2. Oversampling - Randomly selecting records (with replacements) from the smaller classes.
3. Generative model - Training a T5 model for every rare class in the dataset.
We will use those models to generate new records for every class using different inputs which are based on real texts from the class.
<br>

### The Data<br>
<ins>The dataset:</ins> For this discussion we will use a dataset composed from HTML home pages of different websites, classified to industries based on their linkedin page.
<br>

<ins>A snippet of the original dataset::</ins>
<br>
![Raw Dataset Example](Images/RawDatasetExample.jpeg "Raw Dataset Example")
<br>

**<ins>EDA:</ins>** Since the dataset is composed of 147 classes and more than 1.7 million records, we chose to focus on 10 classes:
* 5 of the most frequent classes.
* 5 of the least frequent classes among those that have more than 1K records.

**We can see the huge imbalance in the following image(Notice the log scale):**

![Industries Volumes](Images/IndustriesVolumes.png "Industries Volumes")
<br>

### Preprocessing
Our preprocess included the following stages:
1. Separating words that were wrongly concatenated, probably due to scraping, e.g.: RetailBusiness -> Retail Business.
2. Removing every character which is not an english letter, a number or one of the following punctuation marks: `['(', ')', '-', '.', ':', ',', '?', '&']`.
3. reducing multiple spaces to a single space.
4. reducing multiple dots to a single dot.
5. removing tags that remained from html.
6. removing the first sentence and last 2 sentences from the texts because the usually contained HTML garbage (e.g.: copyrights, dropdown menu etc.)
7. Transforming the dataset into a format which is used by [fasttext for text classification](https://fasttext.cc/docs/en/supervised-tutorial.html) library.

The preprocessing took place in the file `preprocess.py` <br>

### T5 Fine Tuning
In order to perform fine-tuning for [T5 model by HuggingFace](https://huggingface.co/docs/transformers/model_doc/t5), we first needed to adapt the given data into a fill-in-the-blank task (the original T5's pre-training task):
<br>
![Fill in the Blank Task](Images/FillInTheBlank.jpeg "Fill in the Blank Task")
<br>
This adaption was done using the `create_labeled_data.py` script.
<br>
Once we have the adapted data, we performed fine-tuning to the pre-trained T5 (we used `t5-base`). <br>
The fine-tuning was done using `fine-tuning.py`.<br>
For each of the 5 least frequent classes we fine-tuned a different model that was trained only on the texts from that industry.
We wanted each model to learn the specific style of its industry so that it will generate text similar to the real records.

### T5 Text Generation
To generate text, we used the 'summarize' T5 task. In order to create rich and diverse generated text (with different contexts) we created a list of different inputs.
<br>
The input was in built the following structure: `"summarize: <industry name>. <input>"`<br>
in order to create the `<input>` we used the script `creating_generator_input.py` which does the following:
  * This script uses two external files: `kw_per_ind.pkl` & `sentences_per_ind.pkl`. The first file is composed of selected keywords per industry (i.e. class), and the second file is composed out of randomly selected sentences per industry.
  * Using these files, each `<input>` was chosen in one of the two ways:
    * With probability of 0.5 an input based on 2 random keywords from `kw_per_ind.pkl`
    * With probability of 0.5 input based on 1 random sentence from `sentences_per_ind.pkl`.
  * Then we tokenized the entire input using the T5 Tokenizer <br>

The text generation per industry was preformed using `generate_per_ind.py`
* Note that since the models were fine-tuned per industry, each of the generation process used a different model specific to the industry.
* The generation was performed with a 4 branch Beam Search and enforcing that no 2gram will appear more than once in the text. 

### Comparing the methods
In order to evaluate which method resulted in a better model w.r.t rare industries, we used several metrics:
1. F1-Micro.
2. F1-Macro.
3. F1-Weighted.
4. Recall per industry.
<br>
Those methods are being compared in `Framework.ipynb` notebook as we will explain in the next section.
<br>


### Reproduction
* Note that all scripts and notebooks have hard-coded paths, if you want to reproduce with different dataset or setup you should change those paths.
1. Run `preprocess.py`.
2. Run `T5/fine_tuning_T5/create_labeled_data.py`.
3. Run `T5/fine_tuning_T5/fine_tuning.py`.
4. Run `T5/creating_generator_input.py`.
5. Run `T5/generate_per_ind.py`.
6. Run all cells in `Framework.ipynb` and watch the magic happen (Specifically the last cells comparing the metrics mentioned above).


### Running with different configurations
In order to run with different configurations you will need to change the code in the following manner:
1. In order to use different model to train on the textual dataset (we used fasttext), you should change the class `Model` in `CustomModel.py` and implement the `train` function and all other functions you wish to use differently.
2. If you wish to generate text with different model other than T5, you should check how to fine_tune it and implement it, then output the generated text as a list (see `generate_per_ind.py`).
3. If you wish to work on a different dataset you should rewrite every path directing to dataset in `Framework.ipynb` and in `create_labeled_data.py`.
4. In order to check different datasets using fasttext (or configured model via `CustomModel.py`) you can use `auto_checking_engine.py` by giving it the following parameters when running (in the same order):
   * fasttext_train_file_path
   * save_model_path
   * train_dataset_path
   * test_dataset_path
   * ids
