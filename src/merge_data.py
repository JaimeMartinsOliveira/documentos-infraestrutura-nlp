import os
import pandas as pd

def merge_csv_files(input_folder='data/processed', output_file='data/processed/train.csv'):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(f"Arquivos CSV encontrados: {csv_files}")

    dataframes = []
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        print(f"Lendo {file_path} ...")
        df = pd.read_csv(file_path, encoding='latin1')
        #df = pd.read_csv(file_path, encoding='iso-8859-1')
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    merged_df.to_csv(output_file, index=False)
    print(f"Dados combinados salvos em {output_file}")

if __name__ == "__main__":
    merge_csv_files()
