command = [
    "python",
    "-u",
    "-m",
    "cehrbert_data.apps.generate_training_data",
    "-i", f"allofus_omop_v8",
    "-o", f"allofus_omop_v8/patient_sequence_with_inpatient_hour_token",
    "-iv",
    "-ip",
    "--gpt_patient_sequence",
    "--include_concept_list",
    "--include_inpatient_hour_token",
    "--att_type", "day",
    "--inpatient_att_type", "day",
    "-tc", "condition_occurrence", "procedure_occurrence", "drug_exposure",
    "-d", "1985-01-01"
]

output = subprocess.run(command, capture_output=True)
print(output.stderr)