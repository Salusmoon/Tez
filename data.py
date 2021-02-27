import pandas as pd
import numbers as np
import random 
import math






def build_sensor_data(deneme):      ## Sensor Datas
    df= pd.DataFrame()
    for i in range(deneme):
            d1 = random.randint(1, 99)   #80
            d2 = random.randint(1, 99)   #38
            d3 = random.randint(1, 99)   #64
            d4 = random.randint(1, 99)   #73 
            df2 = pd.DataFrame({"x1":[d1],
                            "x2":[d2],
                            "x3":[d3],
                            "x4":[d4],
                            "y1":[99],
                            "y2":[99],
                            "y3":[99],
                            "y4":[99]})
            df = df.append(df2)
    return df



def build_wawe_data(deneme):           ## Wawe Datas
    df = pd.DataFrame()
    for i in range(deneme):
        wawe_x= random.randint(0,99) #74
        wawe_y= random.randint(0,99) #74    
        df2 = pd.DataFrame({"x":[wawe_x],
                            "y":[wawe_y]})
        df=df.append(df2)
    return df    


sensors = build_sensor_data(100)
wawes = build_wawe_data(100)



sensors.to_csv (r'/home/salusmoon/Desktop/Earthquake/Sensor_data.csv', index = False, header=True)
wawes.to_csv (r'/home/salusmoon/Desktop/Earthquake/Wawe_data.csv', index = False, header=True)