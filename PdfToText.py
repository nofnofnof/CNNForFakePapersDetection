import os
from os import chdir, getcwd, listdir, path
import PyPDF2
from time import strftime
import glob, io, re, string


def check_path(prompt):
    abs_path = input(prompt)
    while path.exists(abs_path) != True:
        print("\nThe specified path does not exist.\n")
        abs_path = input(prompt)
    return abs_path

def PunctRemoverToFile( pathFile):
    exclude = set(string.punctuation)

    f = open(pathFile, 'r', encoding="utf8")
    text = f.readlines()
    f.close()

    op_file_path = str(pathFile) + 'Changed.txt'
    if os.path.isfile(op_file_path):  # Checks if o/p file exists
        os.remove(op_file_path)  # deletes the existing o/p file

    for x in range(0, len(text)):
        s = text[x]
        s = s.replace('-', ' ')
        s1 = ''.join(ch for ch in s if ch not in exclude)
        result = ''.join([i for i in s1 if not i.isdigit()])
        op_file = open(op_file_path, 'a', encoding="utf8")
        op_file.write(re.sub('\s{1,}', '', result.strip()).lower())
        op_file.close()
    return op_file_path

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



text_file_output = ConvertPdfFilesToText('D:\Projects\pdf_exemple_file.pdf')


