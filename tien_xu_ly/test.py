import pandas as pd
def calculate_AQI_PM25(PM25):
    AQI_PM25= 0
    if 0 <= PM25 <= 9.0:
        AQI_PM25 = ((50 - 0) / (9.0 - 0)) * (PM25 - 0) + 0
    elif 9.1 <= PM25 <= 35.4:
        AQI_PM25 = ((100 - 51) / (35.4 - 9.1)) * (PM25 - 9.1) + 51
    elif 35.5 <= PM25 <= 55.4:
        AQI_PM25 = ((150 - 101) / (55.4 - 35.5)) * (PM25 - 35.5) + 101
    elif 55.5 <= PM25 <= 125.4:
        AQI_PM25 = ((200 - 151) / (125.4 - 55.5)) * (PM25 - 55.5) + 151
    elif 125.5 <= PM25 <= 225.4:
        AQI_PM25 = ((300 - 201) / (225.4 - 125.5)) * (PM25 - 125.5) + 201
    elif 225.5 <= PM25 <=325.4:
        AQI_PM25 = ((500 - 301) / (325.4 - 225.5)) * (PM25 - 225.5) + 301
    elif PM25:
        AQI_PM25 = ((500 - 301) / (350.4 - 225.5)) * (PM25 - 350.4) + 501

    return AQI_PM25

def calculate_AQI_PM10(PM10):
    AQI_PM10 = 0
    if 0 <= PM10 <= 54:
        AQI_PM10 = ((50 - 0) / (54 - 0)) * (PM10 - 0) + 0
    elif 55 <= PM10 <= 154:
        AQI_PM10 = ((100 - 51) / (154 - 55)) * (PM10 - 55) + 51
    elif 155 <= PM10 <= 254:
        AQI_PM10 = ((150 - 101) / (254 - 155)) * (PM10 - 155) + 101
    elif 255 <= PM10 <= 354:
        AQI_PM10 = ((200 - 151) / (354 - 255)) * (PM10 - 255) + 151
    elif 355 <= PM10 <= 424:
        AQI_PM10 = ((300 - 201) / (424 - 355)) * (PM10 - 355) + 201
    elif 425 <= PM10 <= 604:
        AQI_PM10 = ((500 - 301) / (604 - 425)) * (PM10 - 425) + 301
    elif PM10 >= 605:
        AQI_PM10 = ((500 - 301) / (604 - 425)) * (PM10 - 605) + 501
    return AQI_PM10

def calculate_AQI_SO2(SO2):
    AQI_SO2 = 0
    if 0 <= SO2 <= 35:
        AQI_SO2 = ((50 - 0) / (35 - 0)) * (SO2 - 0) + 0
    elif 36 <= SO2 <= 75:
        AQI_SO2 = ((100 - 51) / (75 - 36)) * (SO2 - 36) + 51
    elif 76 <= SO2 <= 185:
        AQI_SO2 = ((150 - 101) / (185 - 76)) * (SO2 - 76) + 101
    elif 186 <= SO2 <= 304:
        AQI_SO2 = ((200 - 151) / (304 - 186)) * (SO2 - 186) + 151
    elif 305 <= SO2 <= 604:
        AQI_SO2 = ((300 - 201) / (604 - 305)) * (SO2 - 305) + 201
    elif 604 <= SO2 <= 1004:
        AQI_SO2 = ((500 - 301) / (1004 - 604)) * (SO2 - 604) + 301
    elif SO2 >= 1005:
        AQI_SO2 = ((500 - 301) / (1004 - 604)) * (SO2 - 604) + 501
    return AQI_SO2

def calculate_AQI_NO2(NO2):
    AQI_NO2 = 0
    if 0 <= NO2 <= 53:
        AQI_NO2 = ((50 - 0) / (53 - 0)) * (NO2 - 0) + 0
    elif 54 <= NO2 <= 100:
        AQI_NO2 = ((100 - 51) / (100 - 54)) * (NO2 - 54) + 51
    elif 101 <= NO2 <= 360:
        AQI_NO2 = ((150 - 101) / (360 - 101)) * (NO2 - 101) + 101
    elif 361 <= NO2 <= 649:
        AQI_NO2 = ((200 - 151) / (649 - 361)) * (NO2 - 361) + 151
    elif 650 <= NO2 <= 1249:
        AQI_NO2 = ((300 - 201) / (1249 - 650)) * (NO2 - 650) + 201
    elif 1250 <= NO2 <= 2049:
        AQI_NO2 = ((500 - 301) / (2049 - 1250)) * (NO2 - 1250) + 301
    elif NO2 >= 22050:
        AQI_NO2 = ((500 - 301) / (2049 - 1250)) * (NO2 - 2050) + 501

    return AQI_NO2

def calculate_AQI_CO(CO):
    AQI_CO = 0
    if 0 <= CO <= 4.4:
        AQI_CO = ((50 - 0) / (4.4 - 0)) * (CO - 0) + 0
    elif 4.5 <= CO <= 9.4:
        AQI_CO = ((100 - 51) / (9.4 - 4.5)) * (CO - 4.5) + 51
    elif 9.5 <= CO <= 12.4:
        AQI_CO = ((150 - 101) / (12.5 - 9.5)) * (CO - 9.5) + 101
    elif 12.5 <= CO <= 15.4:
        AQI_CO = ((200 - 151) / (15.4 - 12.5)) * (CO - 12.5) + 151
    elif 15.5 <= CO <= 30.4:
        AQI_CO = ((300 - 201) / (30.4 - 15.5)) * (CO - 15.5) + 201
    elif 30.5 <= CO <= 50.4:
        AQI_CO = ((500 - 301) / (50.4 - 30.5)) * (CO - 30.5) + 301
    elif CO >= 50.5:
        AQI_CO = ((500 - 301) / (50.4 - 30.5)) * (CO - 50.5) + 501
    return AQI_CO

def calculate_AQI_O3(O3):
    AQI_O3 = 0
    if 0 <= O3 <= 0.054:
        AQI_O3 = ((50 - 0) / (0.054 - 0)) * (O3 - 0) + 0
    elif 0.055 <= O3 <= 0.124:
        AQI_O3 = ((100 - 51) / (0.124 - 0.055)) * (O3 - 0.055) + 51
    elif 0.125 <= O3 <= 0.164:
        AQI_O3 = ((150 - 101) / (0.164 - 0.125)) * (O3 - 0.125) + 101
    elif 0.165 <= O3 <= 0.204:
        AQI_O3 = ((200 - 151) / (0.204 - 0.165)) * (O3 - 0.165) + 151
    elif 0.205 <= O3 <= 0.404:
        AQI_O3 = ((300 - 201) / (0.404 - 0.205)) * (O3 - 0.205) + 201
    elif 0.405 <= O3 <= 0.604:
        AQI_O3 = ((500 - 301) / (0.604 - 0.405)) * (O3 - 0.405) + 301
    elif O3 >= 0.605:
        AQI_O3 = ((500 - 301) / (0.604 - 0.405)) * (O3 - 0.605) + 501

    return AQI_O3

def calculate_AQI(PM25, PM10, SO2, NO2, CO, O3):
    AQI_PM25 = calculate_AQI_PM25(PM25)
    # Thêm các hàm tính toán AQI cho PM10, SO2, NO2, CO, O3 tương tự như hàm calculate_AQI_PM25
    AQI_PM10 = calculate_AQI_PM10(PM10)
    AQI_SO2 = calculate_AQI_SO2(SO2)
    AQI_NO2 = calculate_AQI_NO2(NO2)
    AQI_CO = calculate_AQI_CO(CO)
    AQI_O3 = calculate_AQI_O3(O3)
    AQI = max(AQI_PM25, AQI_PM10, AQI_SO2, AQI_NO2, AQI_CO, AQI_O3)
    print("PM2.5-AQI: " + str(AQI_PM25) + " | PM10-AQI: " + str(AQI_PM10) + " | SO2-AQI: " + str(AQI_SO2) + " | NO2-AQI: " + str(AQI_NO2) + " | CO-AQI: " + str(AQI_CO) + " | O3: " + str(AQI_O3))
    return AQI

def ug_per_m3_to_ppm(ug_per_m3,molecular_weight):
    # Chuyển đổi μg/m³ sang ppm
    ppm = (ug_per_m3 * 24.45) / molecular_weight
    return ppm

def class_AQI(AQI):
    if 0 <= AQI <= 50:
        return "Good"
    elif 51 <= AQI <= 100:
        return "Moderate"
    elif 101 <= AQI <= 150:
        return "Unhealthy_for_Sensitive_Groups"
    elif 151 <= AQI <= 200:
        return "Unhealthy"
    elif 201 <= AQI <= 300:
        return "Very_Unhealthy"
    else:
        return "Hazardous"

def data_test(data):
    print(data)
    if str(data.iloc[0]) == 'NA' or float(data.iloc[0]) < 0:
        return 0
    else:
        return 1
