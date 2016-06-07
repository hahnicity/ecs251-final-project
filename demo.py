from time import sleep

import requests

from collate import collate_all_from_breath_meta_to_data_frame

cohort = collate_all_from_breath_meta_to_data_frame(20, 5000)

ards_example = list(cohort.iloc[0].values[0:180])
ards_pt_id = cohort.iloc[0].values[181].split("/")[1]

print("We're going to send data for an ARDS patient")
response = requests.post(
    "http://ecs251-demo/analyze/",
    json={"patient_id": ards_pt_id, "breath_data": ards_example}
)
sleep(5)
print("Now we're going to send data for an control patient")
i = -1
control_example = list(cohort.iloc[i].values[0:180])
control_pt_id = cohort.iloc[i].values[181].split("/")[1]
requests.post(
    "http://ecs251-demo/analyze/",
    json={"patient_id": control_pt_id, "breath_data": control_example}
)
