import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path_traffic = "traffic_2509261758.csv"

df_traffic = kagglehub.dataset_load(   
    KaggleDatasetAdapter.PANDAS,
    "qh20166/data-dacn",
    file_path_traffic,
)

# Xuất ra 5 dòng đầu
print("=== 5 dòng đầu tiên ===")
print(df_traffic.head())

# Xuất ra currentSpeed của dòng thứ 50
current_speed = df_traffic.at[49, "currentSpeed"]
print("\n=== currentSpeed của dòng thứ 50 ===")
print(current_speed)

# Xuất ra toàn bộ thông tin của dòng 49
print("\n=== Thông tin dòng 49 (dòng thứ 50) ===")
print(df_traffic.loc[49])


file_path_weather = "weather.csv"

# Load file weather.csv thành DataFrame
df_weather = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "qh20166/data-dacn",   # dataset của bạn
    file_path_weather,
)

# Xuất thử vài dòng đầu
print("=== 5 dòng đầu tiên của weather.csv ===")
print(df_weather.head())