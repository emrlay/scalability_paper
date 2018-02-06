import openpyxl
import numpy as np
import plot_normal_graph as png

_default_column_tags = ['B', 'C', 'D', 'E', 'F']
"""
149, 150: Weighted Approach, METIS & SPEC
157, 158: Propagated Approach, METIS & SPEC
151: E2CP
152: Cop-KMeans
153, 154: E2CP Ensemble, METIS & SPEC
155, 156: Cop-KMeans Ensemble, METIS & SPEC
"""
_default_row_indices = [149, 150, 157, 158, 151, 152, 153, 154, 155, 156]
_default_dataset = ['OPTDIGITS',
                    'ISOLET',
                    'waveform',
                    'COIL20',
                    'Segmentation',
                    'SRBCT',
                    'LungCancer',
                    'Wap',
                    're0']


def load_result_from_excel(file_name, dataset_name, column_tags=_default_column_tags,
                           row_indices=_default_row_indices):
    """
    extract experimental results from excel file in given format.
    results will be stored in a dictionary where keys are the names of datasets, values are actual results.

    :param file_name: name of excel file where results are
    :param dataset_name: name of datasets that should be extracted
    :param column_tags: columns where results are (in a list)
    :param row_indices: upper bound and lower bound of results' position (in a tuple)
    :return: experimental results, as a ndarray
    """
    wb = openpyxl.load_workbook(file_name, data_only=True)
    avail_dataset = wb.get_sheet_names()
    # exp_results = {}
    exp_results = []
    dataset_names = []
    for dataset in dataset_name:
        if dataset not in avail_dataset:
            print 'Experiment results of dataset:' + dataset + ' are not provided in the excel file, please check.'
            continue
        sheet = wb.get_sheet_by_name(dataset)
        single_dataset_result = []
        for row in row_indices:
            row_result = []
            for column in column_tags:
                row_result.append(sheet[str(column)+str(row)].internal_value)
            single_dataset_result.append(row_result)
        single_dataset_result = np.array(single_dataset_result)
        dataset_names.append(dataset)
        exp_results.append(single_dataset_result)
        print 'Dataset: ' + dataset + ' loading completed.'
        # exp_results[dataset] = single_dataset_result
    return exp_results, dataset_names

if __name__ == '__main__':
    comparisons = ['Weight(METIS)', 'Weight(Spec)', 'Prop(METIS)', 'Prop(SPEC)', 'E2CP', 'Cop-KMeans',
                   'E2CP(METIS)', 'E2CP(Spec)', 'Cop-KMeans(METIS)', 'Cop-KMeans(Spec)']
    colors = {'Weight(METIS)': 'magenta',
              'Weight(Spec)': 'crimson',
              'E2CP': 'turquoise',
              'Cop-KMeans': 'deepskyblue',
              'E2CP(METIS)': 'lime',
              'E2CP(Spec)': 'springgreen',
              'Cop-KMeans(METIS)': 'saddlebrown',
              'Cop-KMeans(Spec)': 'sienna',
              'Prop(METIS)': 'blue',
              'Prop(SPEC)': 'royalblue'}
    markers = {'Weight(METIS)': 's',
               'Weight(Spec)': '^',
               'E2CP': 'D',
               'Cop-KMeans': 'd',
               'E2CP(METIS)': '|',
               'E2CP(Spec)': '.',
               'Cop-KMeans(METIS)': '+',
               'Cop-KMeans(Spec)': 'x',
               'Prop(METIS)': 'p',
               'Prop(SPEC)': 'v'}
    linestyles = {'Weight(METIS)': '-',
                  'Weight(Spec)': '-',
                  'E2CP': '--',
                  'Cop-KMeans': '--',
                  'E2CP(METIS)': ':',
                  'E2CP(Spec)': ':',
                  'Cop-KMeans(METIS)': ':',
                  'Cop-KMeans(Spec)': ':',
                  'Prop(METIS)': '-',
                  'Prop(SPEC)': '-'}
    xlabel = '#constraints'
    ylabel = 'NMI'
    xticks = ['0.25n', '0.5n', 'n', '1.5n', '2n']

    data, names = load_result_from_excel('Incre_result_added_prop.xlsx', _default_dataset)
    print data
    print names
    #
    png.plot(comparisons, data, colors, markers, linestyles, names, xlabel, ylabel, xticks, 3, 3)
    #
    # data, names = load_result_from_excel('Diff_result_added_prop.xlsx', _default_dataset)
    # png.plot_boxplot(data, names, 3, 3, 'NMI', '')
