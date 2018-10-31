import pandas as pd

disease_cure_file = "fact_base/disease_cure.csv"
cure_desc_file = "fact_base/cure_description.csv"


cure_df = pd.read_csv(disease_cure_file)
cure_dict = cure_df.to_dict()

cure_desc_df = pd.read_csv(cure_desc_file)

cure_dict_keys = ["Disease", "Diagnosis 1", "Diagnosis 2", "Diagnosis 3"]


def get_facts(disease_index):
    disease_data = {}
    cure_data = {}

    for k in cure_dict_keys:
        disease_data[k] = cure_dict[k][disease_index]

    for d in ['Diagnosis 1', 'Diagnosis 2', 'Diagnosis 3']:
        if disease_data[d] != 'NaN':
            cure_data[disease_data[d]] = "".join(list(cure_desc_df[cure_desc_df['Pesticide']
                                                                   == disease_data[d]]['Description']))

    temp_disease = disease_data['Disease']
    crop_name, disease_name = temp_disease.split('___')
    disease_data['crop_type'] = crop_name.replace("_", " ")
    disease_data['Disease'] = disease_name.replace("_", " ")
    return {'disease': disease_data, 'description': cure_data}
