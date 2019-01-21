import numpy as np
import re
import glob
import PyPDF2
import os
from os import chdir, getcwd, listdir, path
from time import strftime

def create_dataset_file(text_output, source_files_folder):
    #Concatenates the files into one file
    with open(text_output, "w") as f1:
        for file in glob.glob(source_files_folder + "/*.txt"):
            with open(file) as f:
                for line in f:
                    f1.write(line)
        f1.close()


def clean_txt(txt_file_path):
    with open(txt_file_path, "w") as f:
        for line in f:
            if len(line) <= 10:
                line = '\n'
                return clean_str(line)
            else:
                return clean_str(line)

def clean_str(string):
    #Tokenization/string cleaning for all datasets except for SST.
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(human_data_file, robot_data_file):
    # Loading data from the files
    human_examples = list(open(human_data_file, "r").readlines())
    human_examples = [s.strip() for s in human_examples]
    robot_examples = list(open(robot_data_file, "r").readlines())
    robot_examples = [s.strip() for s in robot_examples]
    # Splitting the sentences by words
    x_text = human_examples + robot_examples
    x_text = [clean_str(sent) for sent in x_text]
    #Generating labels
    human_labels = [[0, 1] for _ in human_examples]
    robot_labels = [[1, 0] for _ in robot_examples]
    y = np.concatenate([human_labels, robot_labels], 0)
    return [x_text, y]

def load_data(data_file):
    """
    Loads  data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    sentences = list(open(data_file, "r", encoding=None).readlines())
    sentences = [s.strip() for s in sentences]
    x_text = sentences
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [clean_str_sst(sent) for sent in x_text]
    # Generate labels
    return [x_text]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def clean_txt_new(txt_file_path):
    with open(txt_file_path, "r+") as f:
        for line in f:
             if len(line) == 0:
                 line = ' \n'
                 clean_str(line)
             else:
                clean_str(line)
    f.close()
def clean_txt(txt_file_path):
    with open(txt_file_path, "r+") as f:
        for line in f:
            if len(line) <= 10:
                line = '\n'
                return clean_str(line)
            else:
                return clean_str(line)


#converts pdf, returns its text content as a string
def ConvertPdfFilesToText(filepath):
    # read the pdf file to page_content
    page_content =''
    pdf_file = open(filepath, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    for i in range(number_of_pages):
        page = read_pdf.getPage(i)
        page_content += page.extractText()
    print(strftime("%H:%M:%S"), " pdf  -> txt ")
    #build the text file path
    head,tail = os.path.split(filepath)
    var = '\\'
    tail = tail.replace(".pdf",".txt")
    text_file = head+var +tail
    with open(text_file, 'a', encoding="utf-8") as out:
        out.write(page_content)
    return text_file

def prepare_for_test(input_file, output_file):
    with open(output_file, "w") as f1:
        with open(input_file, "r+") as f2:
            for line in f2:
                if len(line) > 5:
                    f1.write(line)
    f2.close()
    f1.close()