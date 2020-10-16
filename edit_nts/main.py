import os

import dataset
from preprocess import process_raw_lines
from stats import plot_labels

def main():
    wikilarge_tokens = None

    if(len(os.listdir('output')) == 0):
        print("Reading dataset files")

        print("Processing wikilarge files")
        comp_lines, simp_lines = dataset.wikilarge()
        wikilarge_tokens = process_raw_lines(comp_lines, simp_lines) 
        dataset.serialize_tokens(dataset.WIKILARGE_TRAIN_PREFIX, wikilarge_tokens)
    else:
        print("Deserializing processed datasets")

        print("Deserializing wikilarge")
        wikilarge_tokens = dataset.deserialize_tokens(dataset.WIKILARGE_TRAIN_PREFIX)
        

    print("Token keys: {}".format(wikilarge_tokens.keys()))
    print("Num edit labels: {}".format(len(wikilarge_tokens["edit_labels"])))
    print("First edit sequence: {}".format(wikilarge_tokens["edit_labels"][0]))

    print("Generating wikilarge stats")
    plot_labels(wikilarge_tokens["edit_labels"], 'edit_labels.png')
    print("Finished")

if __name__ == "__main__":
    main()
