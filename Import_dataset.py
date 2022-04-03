import gzip


def print_dataset(number_of_lines):
    file_name = 'msmarco-docs.trec.gz'
    f = gzip.open(file_name, 'r') #read from a buffer so we never load the 8 GB in memory
    i = 0
    number_of_texts = 0
    for line in f: #line is type python_bytes
        if(i< number_of_lines):
            line_to_string = line.decode("utf-8")
            print(line_to_string[:-1])
            i+=1
        else:
            break
    f.close()

def print_n_texts(number_of_texts):
    file_name = 'msmarco-docs.trec.gz'
    f = gzip.open(file_name, 'r') #read from a buffer so we never load the 8 GB in memoryç
    number_of_texts_counter = 0
    for line in f:
        if(number_of_texts_counter < number_of_texts):
            line_to_string = line.decode("utf-8")
            if(line_to_string == "\n"): #found the end of a text
                number_of_texts_counter += 1
                print("\n #################### END OF THE TEXT ####################")
            print(line_to_string[:-1])
        else:
            break
    f.close()
    if(number_of_texts == number_of_texts_counter):
        print("Succesfully printed", number_of_texts,"texts")
    else:
        print("Text printing ERROR!")


if __name__ == "__main__":
    print_n_texts(3)