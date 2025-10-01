import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path_traffic = "traffic_hcm.csv"

df_traffic = kagglehub.dataset_load(   
    KaggleDatasetAdapter.PANDAS,
    "qh20166/data-dacn",
    file_path_traffic,
)

print("=== 5 dòng đầu tiên ===")
print(df_traffic.head())

current_speed = df_traffic.at[49, "currentSpeed"]
print("\n=== currentSpeed của dòng thứ 50 ===")
print(current_speed)

print("\n=== Thông tin dòng 49 (dòng thứ 50) ===")
print(df_traffic.loc[49])


file_path_weather = "weather.csv"

df_weather = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "qh20166/data-dacn",   
    file_path_weather,
)

print("=== 5 dòng đầu tiên của weather.csv ===")
print(df_weather.head())