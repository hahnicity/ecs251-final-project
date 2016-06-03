import csv
from glob import glob
import os

from numpy import append, array, empty, inf, nan
from pandas import DataFrame, read_csv, Series


def process_features(df):
    # Pretty sure these features won't matter
    del df['BN']
    #del df['ventBN']
    del df[' ']
    del df['BS']
    del df['IEnd']
    del df['maxP']  # This is a dupe of PIP
    # Experimental
    del df['TVi']
    del df['TVe']
    del df['TVe:TVi ratio']
    del df['maxF']
    del df['minF']
    del df['BS.1']
    del df['x01']
    del df['TVi1']
    del df['TVe1']
    del df['x02']
    del df['TVi2']
    del df['TVe2']
    del df['iTime']
    # start
    #del df['eTime']
    #del df['Maw']
    #del df['I:E ratio']
    #del df['ipAUC']
    #del df['epAUC']
    #del df['PIP']
    #del df['PEEP']
    #del df['inst_RR']
    #del df['minF_to_zero']
    # end
    del df['plat_pressure']
    del df['brunner']
    df = df.replace([inf, -inf], nan).dropna()
    return df


def get_cohort_files(cohort):
    if cohort not in ["ardscohort", "controlcohort"]:
        raise Exception("Input must either be ardscohort or controlcohort")
    dirs = os.listdir(cohort)
    cohort_files = []
    for dir in dirs:
        files = glob("{}/{}/0*_breath_meta.csv".format(cohort, dir))
        for f in files:
            cohort_files.append(f)

    return cohort_files


def collate_from_breath_meta_to_list(cohort):
    """
    Gets all breath_meta.csv files in our specific cohort and then gets all
    the data from these files and stores them in a list.
    """
    cohort_files = get_cohort_files(cohort)
    data = []
    for f in cohort_files:
        with open(f) as meta:
            reader = csv.reader(meta)
            for line in reader:
                data.append(line)
    return data


def collate_from_breath_meta_to_data_frame(cohort, breaths_to_stack, samples):
    cohort_files = get_cohort_files(cohort)
    df = process_features(read_csv(cohort_files[0]))
    initial_features = list(df.columns.values)
    rolling = create_rolling_frame(df, breaths_to_stack)
    file_array = [cohort_files[0]] * len(rolling)
    for f in cohort_files[1:]:
        if samples and len(rolling) >= samples:
            break
        new = process_features(read_csv(f))
        if len(new.index) == 0:
            continue
        new = create_rolling_frame(new, breaths_to_stack)
        rolling = append(rolling, new, axis=0)
        file_array.extend([f] * len(new))
    df = DataFrame(rolling)
    df['filename'] = file_array
    df = df.rename(columns={(len(initial_features) - 1) * breaths_to_stack: 'start_vent_bn'})
    return df


def create_rolling_frame(df, breaths_in_frame):
    matrix = df.as_matrix()
    # The +1 is for the start bn
    rolling = empty((0, ((len(matrix[0]) - 1) * breaths_in_frame) + 1), float)
    # The [1:] means cut off the vent bn
    row = matrix[0][1:]
    start_bn = 0
    for i, _ in enumerate(df.index[:-1]):
        # Ensure we can attach initial vent bn without interfering with our model
        if start_bn == 0:
            start_bn = int(matrix[i][0])
        if (i + 1) % breaths_in_frame == 0:
            row = append(row, [start_bn])
            rolling = append(rolling, [row], axis=0)
            row = array([])
            start_bn = 0
        # The [1:] means cut off the vent bn
        row = append(row, matrix[i + 1][1:])
    return rolling


def collate_all_from_breath_meta_to_data_frame(breaths_to_stack, samples):
    ards = collate_from_breath_meta_to_data_frame("ardscohort", breaths_to_stack, samples)
    y = Series(1, index=ards.index)
    ards['y'] = y
    control = collate_from_breath_meta_to_data_frame("controlcohort", breaths_to_stack, samples)
    y = Series(-1, index=control.index)
    control['y'] = y
    return ards.append(control, ignore_index=True)
