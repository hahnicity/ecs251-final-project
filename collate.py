import csv
from glob import glob


def collate_from_breath_meta(cohort):
    """
    Gets all breath_meta.csv files in our specific cohort and then gets all
    the data from these files and stores them in a dictionary.
    """
    if cohort not in ["ardscohort", "controlcohort"]:
        raise Exception("Input must either be ardscohort or controlcohort")
    dirs = os.listdir(cohort)
    cohort_files = []
    for dir in dirs:
        files = glob("{}/{}/0*_breath_meta.csv".format(cohort, dir))
        for f in files:
            cohort_files.append(f)

    data = []
    for f in cohort_files:
        with open(f) as meta:
            reader = csv.reader(meta)
            for line in reader:
                data.append(line)
    return data



if __name__ == "__main__":
    main()
