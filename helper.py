import pandas as pd
import os


def read_datasets(folder_path="/Users/eyupburakatahanli/Desktop/tubitak_2209_projem/dataset_combined/dataset_combined"):
    #folder_path = "/Users/eyupburakatahanli/Desktop/tubitak_2209_projem/dataset_combined/dataset_combined"
    os.chdir(folder_path)
    dfs = {}

    for filename in os.listdir():
        if filename.endswith(".csv"):
            df_name = filename.split(".")[0].replace(" ", "")
            df = pd.read_csv(filename)
            exec(df_name + " = pd.DataFrame(df)")
            dfs[df_name] = df

    return dfs