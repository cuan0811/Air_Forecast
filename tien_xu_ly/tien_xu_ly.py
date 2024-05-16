import pandas as pd
import tien_xu_ly.test as test

# Đường dẫn đến tệp Excel .xlsm
file_path = 'D:\Student\project\data-mining\Data_set\data_set_model.xlsx'

# Đọc dữ liệu từ file Excel
df = pd.read_excel(file_path, engine='openpyxl')

# Lọc bỏ các hàng có giá trị NaN
df = df.dropna(subset=df.columns.difference(['Label']), how='any')
# Duyệt qua các hàng trong DataFrame
for index, row in df.iterrows():
    # Truy cập dữ liệu từ cột tương ứng trong DataFrame
    data_PM25 = row['PM2.5']
    data_PM10 = row['PM10']
    data_SO2 = row['SO2']
    data_NO2 = row['NO2']
    data_CO = row['CO']
    data_O3 = row['O3']

    # Chuyển đổi dữ liệu sang dạng số thực
    PM25 = float(data_PM25)
    PM10 = float(data_PM10)
    SO2 = test.ug_per_m3_to_ppm(float(data_SO2), 64.0648)
    NO2 = test.ug_per_m3_to_ppm(float(data_NO2), 46.0048)
    CO = test.ug_per_m3_to_ppm(float(data_CO), 28.01)
    O3 = test.ug_per_m3_to_ppm(float(data_O3), 47.9982)

    # Tính toán AQI
    AQI = test.calculate_AQI(PM25, PM10, SO2, NO2, CO / 1000, O3 / 1000)

    # Phân loại AQI
    class_aqi = test.class_AQI(AQI)
    df.at[index, 'Label'] = class_aqi
    # In kết quả
    print(f"PM25: {PM25}, PM10: {PM10}, SO2: {SO2}, NO2: {NO2}, CO: {CO}, O3: {O3}")
    print(f"AQI: {AQI}, Class AQI: {class_aqi}")
    print("Thành công")
    print("--------------------------------------------------------------------------")

import pandas as pd
new_file_path = 'D:\Student\project\data-mining\Data_Set\data_set_model_processed4.csv'
df.to_excel(new_file_path, index=False)
