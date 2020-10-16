#!/usr/bin/env python

import os

from functools import reduce

import matplotlib.pyplot as plt

def count_labels(edit_labels):
    def increment_counts(counts, labels):
        prev_adds, prev_keeps, prev_deletes = counts
        count_labels = lambda label: len([x for x in labels if x == label])
        
        new_keeps = count_labels("KEEP")
        new_deletes = count_labels("DEL")
        new_stops = 1
        new_adds = len(labels) - new_keeps - new_deletes - new_stops

        return prev_adds + new_adds, prev_keeps + new_keeps, prev_deletes + new_deletes
    
    return reduce(increment_counts, edit_labels, (0, 0, 0))
    
def plot_labels(edit_labels, filename):
    names = ["Adds", "Keeps", "Deletes"]
    values = count_labels(edit_labels)

    fig, ax = plt.subplots()
    ax.ticklabel_format(style="plain")
    ax.bar(names, values)

    fig.savefig(os.path.join('figures', filename), bbox_inches="tight")
