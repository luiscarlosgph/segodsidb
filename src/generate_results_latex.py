"""
@brief   Script that generates a Latex table where every row represents a class
         and every column represents a cross-validation fold. The cells will
         contain the metric, which for ODSI-DB we have decided will be balanced
         accuracy.

@author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    5 Aug 2021.
"""

import argparse
import numpy as np
import os
import json
import numpy as np

# My imports
import torchseg.data_loader as dl

def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-i': 'Dictionary of fold names to paths.',
        '-c': 'Caption of the table of results.',
        '-l': 'Table label.',
        '-s': 'List containing a subset of classes to display.',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    # Create command line parser
    parser = argparse.ArgumentParser(description='PyTorch segmenter.')
    parser.add_argument('-i', '--input', required=True, type=str, 
        help=help('-i'))
    parser.add_argument('-c', '--caption', required=True, type=str, 
        help=help('-c'))
    parser.add_argument('-l', '--label', required=True, type=str, 
        help=help('-l'))
    parser.add_argument('-s', '--classes', required=False, default='None', 
                        type=str, help=help('-l'))

    # Read parameters
    args = parser.parse_args()
    args.input = eval(args.input)
    args.classes = eval(args.classes)
    assert(type(args.input) == dict)

    return args


def validate_cmdline_params(args):
    # Check that the JSON files exist
    #for fid in args.input:
    #    if not os.path.isfile(args.input[fid]):
    #        raise ValueError('[ERROR] The path `' + str(args.input[fid]) \
    #            + '` does not exist.')
    return args


def read_json(path):
    data = None
    with open(path) as f:
        data = json.load(f)
    return data


def print_latex_table(metrics, caption, label, class_subset=None):
    """
    @brief  Print a table with classes as rows and folds as columns.
    @param[in]  metric_per_class  Dictionary of class names -> metric value.
    @param[in]  metric_name       Name of the metric.
    @returns  Nothing.
    """
    # Get the list of classes that the user wants to consider 
    list_of_classes = None
    if class_subset:
        list_of_classes = class_subset
    else:
        list_of_classes = list(metrics[metrics.keys()[0]].keys())
    
    #for c in list_of_classes:
    #    if class_subset is not None and c not in class_subset:
    #        del metric_per_class[c]

    # Average over classes
    #metric_avg = np.nanmean([v for k, v in metric_per_class.items()])

    # Convert metric to percentage
    #metric_per_class = {k: 100. * v for k, v in metric_per_class.items()}
    #metric_avg *= 100.

    # Compute average over classes
    average = {}
    for m in metrics:
        acc = 0.
        for c in metrics[m]:
            if c not in list_of_classes:
                continue
            acc += metrics[m][c] 
        avg = acc / len(list_of_classes)
        average[m] = avg

    # Table header
    print()
    print("\\begin{table*}[!htb]")
    print("    \\centering")
    print("    \\caption{" + caption + "}")
    print("    \\vspace{0.2cm}")
    print("    \\begin{tabular}{l" + "c" * len(metrics) + "}")
    print("        \\hline")
    print("        \\multicolumn{1}{c}{\\bfseries Class} ")
    
    # Print column names
    for m in metrics:
        print("        & \\multicolumn{1}{c}{\\bfseries " \
            + m.capitalize().replace('_', ' ') + "}")  # + " (\\%)}")
    print("        \\\\")
    print("        \\hline")

    # Print all the metrics for each class
    for c in sorted(list_of_classes): 
        line_str = '        ' + c
        for m in metrics: 
            line_str += "  &  {:.2f}".format(metrics[m][c] * 100) 
        line_str += "  \\\\"
        print(line_str)

    print("        \\hline")

    # Print the average for each metric
    line_str = "        \\multicolumn{1}{c}{\\bfseries Average} "
    for m in metrics:
        line_str += "  &  {:.2f}".format(average[m] * 100)
    line_str += "  \\\\"
    print(line_str)

    # Table footer
    print("        \\end{tabular}")
    print("    \\vspace{0.2cm}")
    print("    \\label{" + label + "}")
    print("\\end{table*}")


def main():
    # Read command line parameters
    args = parse_cmdline_params()
    validate_cmdline_params(args)
    
    classnames = dl.OdsiDbDataLoader.OdsiDbDataset.classnames

    # List of suffixes 
    lsuffix = ['_sensitivity.json', '_specificity.json', 
               '_accuracy.json', '_balanced_accuracy.json']

    # Produce latex table for all the metrics
    metrics = {}
    for suffix in lsuffix:
        # Read JSON files with the resuls for each class
        fold_results = {fold_id: read_json(path + suffix) for fold_id, path in args.input.items()}

        # Average over folds
        metric_per_class = {classnames[k]: None for k in classnames}
        for class_name in metric_per_class:
            values = [fold_results[fold_id][class_name] for fold_id in fold_results]
            metric_per_class[class_name] = np.nanmean(values)

        # Store the per-class results for this metric
        metrics[suffix.split('.')[0][1:]] = metric_per_class

    # Print Latex table formated for Photonics West 
    print_latex_table(metrics, caption=args.caption, label=args.label, class_subset=args.classes)
                      #metric_name="Balanced accuracy (\\%)", 


if __name__ == '__main__':
    main()
