import os

import dataset
from preprocess import process_raw_lines
from stats import plot_labels

def main():
    wikilarge_tokens = None
    wikismall_tokens = None

    print("Reading dataset files")

    if(len(os.listdir('output/wikilarge')) == 0):
        print("Processing wikilarge files")
        comp_lines, simp_lines = dataset.wikilarge()
        wikilarge_tokens = process_raw_lines(comp_lines, simp_lines) 
        dataset.serialize_tokens(dataset.WIKILARGE_TRAIN_PREFIX, 'wikilarge', wikilarge_tokens)
    else:
        print("Deserializing wikilarge")
        wikilarge_tokens = dataset.deserialize_tokens(dataset.WIKILARGE_TRAIN_PREFIX, 'wikilarge')

    if(len(os.listdir('output/wikismall')) == 0):
        print("Processing wikismall files")
        comp_lines, simp_lines = dataset.wikismall()
        wikismall_tokens = process_raw_lines(comp_lines, simp_lines) 
        dataset.serialize_tokens(dataset.WIKISMALL_TRAIN_PREFIX, 'wikismall', wikismall_tokens)
    else:
        print("Deserializing wikismall")
        wikismall_tokens = dataset.deserialize_tokens(dataset.WIKISMALL_TRAIN_PREFIX, 'wikismall')
        

    print("Generating wikilarge stats")
    plot_labels(wikilarge_tokens["edit_labels"], 'wikilarge_labels.png')

    print("Generating wikismall stats")
    plot_labels(wikismall_tokens["edit_labels"], 'wikismall_labels.png')
    print("Finished")

if __name__ == "__main__":
    main()
