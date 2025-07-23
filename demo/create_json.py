"""
This script creates a directed graph and saves it to a JSON file.
生成电网联络线信息，并将其保存为JSON文件。
"""
import networkx as nx
import json
import os
import pandas as pd
import random
import csv

# Create directory if it doesn't exist
path = 'data/6-bus'
os.makedirs(path, exist_ok=True)

# file='data/6-bus/wp_data.csv'
# df = pd.read_csv(file)
# df = df.where(df >= 0, -df)
# output_file_path = 'data/6-bus/wp_data.csv'
# df.to_csv(output_file_path, index=False)
config_file = 'data/6-bus/des_config.csv'
config_df = pd.read_csv(config_file)

# CSV文件路径
csv_file_path = 'des_config.csv'
# 根据每一行的数据，创建JSON文件,从第二行开始

for index, row in config_df.iloc[1:].iterrows():
    desid = int(row['desid'])
    # 根据模板创建JSON内容
    json_content = {
        "name": "des",
        "id": desid,
        "devices": {
            "P_load": {
                "path": "data/6-bus/p_load_data.csv",
                "index_col": random.randint(0, 4),
                "magnification factor": random.uniform(0.1, 0.5)
            },
            "H_load": {
                "path": "data/6-bus/h_load_data.csv",
                "index_col": random.randint(0, 4),
                "magnification factor": random.uniform(0.1, 0.5)
            }
        }
    }
    # 根据pv,pw,等键对应的数值，判断是否在json_content中添加pv,pw等设备
    if row['pv'] == 1:
        json_content['devices']['PV'] = {"path": "data/6-bus/pv_data.csv",
                                         "index_col": random.randint(0, 6),
                                         "magnification factor": random.uniform(0.1, 0.5)}
    if row['pw'] == 1:
        json_content['devices']['WP'] = {"path": "data/6-bus/wp_data.csv",
                                         "index_col": random.randint(0, 5),
                                         "magnification factor": random.uniform(0.1, 0.5)}
    if row['chp'] == 1:
        json_content['devices']['CHP'] = {"P_max": 10,
                                          "alpha": 1.5,
                                          "gas_p": 41.67659}
    if row['eh'] == 1:
        json_content['devices']['EH'] = {"P_max": 10, "eta": 0.95}
    if row['batteries'] == 1:
        json_content['devices']['BT'] = {"P_max": 3, "E_max": 100, "E_init": 3, "r_loss": 0.001,
                                         "eta_charge": 0.95, "eta_discharge": 0.95}
    if row["hs"] == 1:
        json_content['devices']['HS'] = {"H_max": 3, "E_max": 100, "E_init": 3, "r_loss": 0.001}
    if row["ehp"] == 1:
        json_content['devices']['EHP'] = {"cop": 3, "P_max": 10, "P_min": 0}
    if row["ahp"] == 1:
        json_content['devices']['AHP'] = {"cop": 1.78, "P_max": 10, "P_min": 0}
    # 构造JSON文件名
    filename = os.path.join(path, f'des{desid}_config.json')
    # 写入JSON文件
    with open(filename, 'w') as jsonfile:
        json.dump(json_content, jsonfile, indent=2)
        print(f'JSON file {filename} created successfully.')
