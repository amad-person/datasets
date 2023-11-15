import yaml
import json
from rdt import HyperTransformer
from rdt.transformers import LabelEncoder
from folktables import ACSDataSource, ACSPublicCoverage


if __name__ == "__main__":
    year = '2021'
    data_source = ACSDataSource(survey_year=year,
                                horizon='1-Year',
                                survey='person')
    acs_data = data_source.get_data(states=None, download=True)
    df, df_labels, _ = ACSPublicCoverage.df_to_pandas(acs_data)
    df = df.dropna(how='any')

    # convert AGEP to categorical
    df['AGEP'] = df['AGEP'] / 10
    df['AGEP'] = df['AGEP'].astype(int)

    # convert all cols to categorical
    ht = HyperTransformer()
    config = {
        "sdtypes": {
            "AGEP": "categorical",
            "SCHL": "categorical",
            "MAR": "categorical",
            "SEX": "categorical",
            "DIS": "categorical",
            "CIT": "categorical",
            "MIG": "categorical",
            "ANC": "categorical",
            "NATIVITY": "categorical",
            "DEAR": "categorical",
            "DEYE": "categorical",
            "DREM": "categorical",
            # "PINCP": "categorical",
            "ST": "categorical",
            "RAC1P": "categorical",
        },
        "transformers": {
            "AGEP": LabelEncoder(),
            "SCHL": LabelEncoder(),
            "MAR": LabelEncoder(),
            "SEX": LabelEncoder(),
            "DIS": LabelEncoder(),
            "CIT": LabelEncoder(),
            "MIG": LabelEncoder(),
            "ANC": LabelEncoder(),
            "NATIVITY": LabelEncoder(),
            "DEAR": LabelEncoder(),
            "DEYE": LabelEncoder(),
            "DREM": LabelEncoder(),
            # "PINCP": LabelEncoder(),
            "ST": LabelEncoder(),
            "RAC1P": LabelEncoder(),
        }
    }
    ht.set_config(config)
    df = ht.fit_transform(df)

    # save dataset
    dataset_name = f"acs_public_cov_{year}"
    df.to_csv(f"{dataset_name}.csv", index_label="ID")

    # create schema and domain
    schema_dict = {}
    domain_dict = {}
    feature_names = df.columns
    for f in feature_names:
        col = df[f]
        min_value = 0
        max_value = col.max()
        schema_dict[f] = list(range(min_value, max_value + 1))
        domain_dict[f] = int(max_value - min_value + 1)

    # save schema
    with open(f"{dataset_name}_schema.yaml", "w") as sch_file:
        yaml.dump(schema_dict, sch_file)

    # save domain
    with open(f"{dataset_name}-domain.json", "w") as dom_file:
        json_obj = json.dumps(domain_dict)
        dom_file.write(json_obj)
