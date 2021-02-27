import numpy as np
import random 
import math
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from math import cos, sin, pi, sqrt, atan2
import unicodedata



edge = 100


wawe_points = []
sensor_point =[]
def build_matrix(edge):  
    matrix =  np.empty([100,100],str)
    matrix[:] = "."
    d1 = random.randint(1, 99)   #80
    d2 = random.randint(1, 99)   #38
    d3 = random.randint(1, 99)   #64
    d4 = random.randint(1, 99)   #73
    #matrix[d1][99] = "A"  #Top level 
    #matrix[d2][99] = "A" # bottom level
    #matrix[d3][99] = "A"   #left side
    #matrix[d4][99] = "A"   #right side
    matrix[0][d1] = "A"  #Top level 
    matrix[99][d2] = "A" # bottom level
    matrix[d3][0] = "A"   #left side
    matrix[d4][99] = "A"   #right side
    return matrix

def wawe_create(matrix):
    wawe_x= random.randint(0,99) #74
    wawe_y= random.randint(0,99) #20
    matrix[wawe_x][wawe_y] = "W"
    return matrix 

def wawe_point(matrix , point):
    wawe_point = np.where(matrix == point)
    points = [int(wawe_point[0]),int(wawe_point[1])]
    return points   

def sensor_points(matrix, point):
    xy_value = []
    points = np.where(matrix == point)
    x_value = points[0]
    y_value = points[1]
    for i in range(len(x_value)):
        sensor = [int(x_value[i]), int(y_value[i])]
        xy_value.append(sensor)
    return xy_value

def distance_normal(matrix , wawe, sensor):
    distances = []
    for i in range(len(sensor)):
        x= (wawe[0]-sensor[i][0])*(wawe[0]-sensor[i][0])
        y= (wawe[1]-sensor[i][1])*(wawe[1]-sensor[i][1])
        distance =  math.sqrt(x+y)
        distances.append(distance)
    return distances
        
def time_for_distance(distance):
    Vp = 4000
    Vs = Vp*10/16
    time = []
    Ts = []
    Tp = []
    for i in range(len(distance)):
        Tp.append(distance[i] / Vp)
        Ts.append(distance[i] / Vs)
    time.append(Tp)
    time.append(Ts)
    return time

def create_signal():
    wawe_signal = signal.unit_impulse(1000)
    wawe_signal[0]= 0
    return wawe_signal

def delta(time_p, time_s, time_freq):
    P_delta = []
    SH_delta = []
    SV_delta = []
    delta = []
    for i in range(len(time_p)):
        array = [ create_signal(), create_signal(), create_signal(), create_signal()]
        P_step = round(time_p[i]/time_freq)
        array[0][P_step] = 1 
        P_delta.append(array[0])
        S_step = round(time_s[i]/time_freq )
        array[1][S_step]=1
        SH_delta.append(array[1])
        array[2][S_step]=1
        SV_delta.append(array[2])
        array[3][P_step] = 1
        array[3][S_step] = 1
        delta.append(array[3])
    dict = {"P_peak": P_delta,"SH_delta": SH_delta, "SV_delta" : SV_delta,"delta": delta}
    deltas = pd.DataFrame(data=dict)
    return deltas

def delta_time_dif(deltas, time_freq):
    time_array=[]
    for i in range(len(deltas)):
        delta = deltas["delta"][i]
        peak = find_peaks(delta)
        peak_diff = peak[0][1] - peak[0][0]
        peak_time = peak_diff * time_freq
        time_array.append(peak_time)
    return time_array
    

def distance_for_delta(peak_diff):
    Vp= 4000
    Vs= Vp *10/16
    distance_delta = []
    for i in range(len(peak_diff)):
        distance = peak_diff[i]*((Vp*Vs)/(Vp-Vs))
        distance_delta.append(distance)
    return distance_delta

def data(sensor , distance):
    sensors=[]
    x = []
    y = []
    r = []
    rgb =[]
    for i in range(len(sensor)):
        sensors.append(i)
        x.append(sensor[i][0])  
        y.append(sensor[i][1])
        r.append(distance[i])
        rgb.append(np.random.rand(3,))
    dict = {"Sensors": sensors, "x": x,"y": y, "r" : r,"rgb": rgb}
    data= pd.DataFrame(data=dict)
    return data


def circle(sensor, distance):
    circle = plt.Circle((sensor[0], sensor[1]), distance)
    return circle

def draw_grafik(data, wawe, deneme, data2):
    plt.figure()
    ax = plt.gca()
    for a, b, size, color in zip(data["x"], data["y"], data["r"], data["rgb"]):

    # plot circles using the RGBA colors
        circle = plt.Circle((a, b), size, color=color, fill=False)
        ax.add_artist(circle)
    
    
    plt.scatter(data["x"],data["y"], color=data["rgb"], marker = 'x')

    ax.scatter(wawe_points[0],wawe_points[1],marker = 'o', color="white", edgecolors="black")
    plt.scatter(data["x"],data["y"],s=0, facecolors='none')
    plt.scatter(data2["x"],data2["y"],marker="+",c=data2["P value"])
    plt.colorbar()
    plt.grid()
    plt.title("deneme {}".format(deneme))
    export_pdf.savefig()
    plt.close

def predic_wawe(array1, array2):
    x_array=[]
    y_array=[]
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            if np.isnan(array1[i][j]) :
                pass
            else:
                x_array.append(array1[i][j])
                y_array.append(array2[i][j])
    data_x = point_find(x_array, "x")
    data_y = point_find(y_array, "y")
    x=[]
    y=[]
    p_value=[]
    for i in range(len(data_x)):
        for j in range(len(data_y)):
            x.append(data_x["x"][i])
            y.append(data_y["y"][j])
            p_value.append(data_x["Count"][i]*data_y["Count"][j])
    dict = {"x":x, "y":y, "P value": p_value}
    data= pd.DataFrame(data=dict)
    data=data.sort_values(by=["P value"],ascending=False)
    
            
    return data

def point_find(array, plane): 
    value=[]
    counts=[]
    while(len(array) != 0):
        index = []
        for i in range(len(array)):
            if i==0:
                p = array[i]
                index.append(i)
            else:
                if ( abs(p-array[i])<=1):
                    p = (p+array[i])/2
                    index.append(i)
        value.append(p)
        counts.append(len(index))
        array = np.delete(array,index)
    dict = {plane: value , "Count": counts}
    data = pd.DataFrame(data=dict)
    return data

def point_find_median(array):
    array.sort()
    mean = np.median(array)
    print(mean)
    index= []
    for i in range(len(array)):
        if (abs(mean-array[i])<= 5):
            pass
        else :
            index.append(i)

    array=np.delete(array, index)
    return array

def predic_wawe_median(array1, array2):
    x_array=[]
    y_array=[]
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            if np.isnan(array1[i][j]) :
                pass
            else:
                x_array.append(array1[i][j])
                y_array.append(array2[i][j])
    x_array=point_find_median(x_array)
    y_array=point_find_median(y_array)
    data_x = point_find(x_array, "x")
    data_y = point_find(y_array, "y")
    x=[]
    y=[]
    p_value=[]
    for i in range(len(data_x)):
        for j in range(len(data_y)):
            x.append(data_x["x"][i])
            y.append(data_y["y"][j])
            p_value.append(data_x["Count"][i]*data_y["Count"][j])
    dict = {"x":x, "y":y, "P value": p_value}
    data= pd.DataFrame(data=dict)
    data=data.sort_values(by=["P value"],ascending=False)
    
            
    return data

def intersection(data):
    x_matrix =  np.empty([len(data["Sensors"]),len(data["Sensors"])])
    y_matrix =  np.empty([len(data["Sensors"]),len(data["Sensors"])])
    x_matrix[:] = None
    y_matrix[:] = None

    for i in range(len(data["Sensors"])):
        for j in range(len(data["Sensors"])):
            if( i == j) or (i>=1 and j<=i):
                pass
            else:
                x1,y1,r1 = data["x"][i],data["y"][i],data["r"][i]
                x2,y2,r2 = data["x"][j],data["y"][j],data["r"][j]    
                dx,dy = x2-x1,y2-y1
                d = sqrt(dx*dx+dy*dy)

                if d > r1+r2:
                    pass
                    #print("#1")
                    #print("çözüm yok")
                elif (d < abs(r1-r2)):
                    pass
                    #print("#2")
                    #print("çözüm yok")
                elif d == 0 and r1 == r2:
                    pass
                    #print("#3")
                    #print("çözüm yok")
                else:
                    a = (r1*r1-r2*r2+d*d)/(2*d)
                    h = sqrt(r1*r1-a*a)
                    xm = x1 + a*dx/d
                    ym = y1 + a*dy/d
                    xs1 = xm + h*dy/d
                    xs2 = xm - h*dy/d
                    ys1 = ym - h*dx/d
                    ys2 = ym + h*dx/d


                    #print(data["Sensors"][i] , data["Sensors"][j])

                    if (xs1>=0 and xs1<=100) and (ys1>=0 and ys1<=100):
                        #x_array.append(xs1)
                        #y_array.append(ys1)
                        x_matrix[i][j]=xs1
                        y_matrix[i][j]= ys1
                        #print("x :", xs1,"y :", ys1)
                    if (xs2>=0 and xs2<=100) and (ys2>=0 and ys2<=100):
                        #x_array.append(xs2)
                        #y_array.append(ys2)
                        y_matrix[j][i]= ys2
                        x_matrix[j][i]= xs2
                        #print("x :", xs2,"y :", ys2)
                    #x_matrix[i][j]=x_array[0]
                    #y_matrix[i][j]= y_array[0]
                    #if len(x_array)==2:
                        #y_matrix[j][i]= y_array[1]
                        #x_matrix[j][i]=x_array[1]
    return x_matrix,y_matrix




# deneme ortamları :  1 tane yukarıda, 2 tane L şeklinde, 1 tane random 50 şer  


with PdfPages(r'plot.pdf') as export_pdf:

    file = open("tek.txt", "w")

    for deneme in range(50):
        matrix = build_matrix(edge)
        wawe_create(matrix)
        wawe_points = wawe_point(matrix, "W")
        sensor_point = sensor_points(matrix, "A")
        distances_hipo = distance_normal(matrix, wawe_points, sensor_point)
        time= time_for_distance(distances_hipo)
        time_p = time[0]
        time_s = time[1]
        delta_to_sensor = delta(time_p,time_s, 0.0001)
        peak_time = delta_time_dif(delta_to_sensor, 0.0001)
        distance_delta = distance_for_delta(peak_time)
        data_sensor = data(sensor_point, distance_delta)
        result_matrix = intersection(data_sensor)
        result_x= result_matrix[0]
        result_y = result_matrix[1]
        wawe_predict= predic_wawe(result_x,result_y)
        wawe_predict2 = predic_wawe_median(result_x,result_y)
        draw_grafik(data_sensor, wawe_points, deneme, wawe_predict)
        draw_grafik(data_sensor,wawe_points,deneme, wawe_predict2)
        print(deneme)
        print(data_sensor)
        print("wawe", wawe_points)
        file.write("DENEME SAYISI :" +  str(deneme) + "\n")
        file.write("SENSOR : \n")
        np.savetxt(file, data_sensor.values[:,:-2], fmt='%d', delimiter="\t", header="\tX\tY")
        file.write("WAWE:    ")
        file.write(str(wawe_points[0]) + "    " + str(wawe_points[1]) + " \n")
        file.write("WAWE_PRED \n")
        file.write("X :\n")
        np.savetxt(file, result_x, fmt='%10.5f')
        file.write("Y :\n")
        np.savetxt(file, result_y, fmt='%10.5f')
        file.write("\n")
        file.write("predict with mod\n")
        np.savetxt(file, wawe_predict, fmt='%d', delimiter="\t", header="\tX\tY\tP")
        file.write("\n")
        file.write("predict with mod and meadin\n")
        np.savetxt(file, wawe_predict2, fmt='%d', delimiter="\t", header="\tX\tY\tP")
        file.write("\n")
        



#print("wawe: " , wawe_points)
#print("sensors: " , sensor_points)
#print("distance: " , distances_hipo)
#print("timeP: " , time_p)
#print("timeS: " , time_s)
#print("deltas : " , delta_to_sensor)
#print("peak time : " , peak_time)
#print("distance for delta : " , distance_delta)
#print(data)
#np.savetxt('output.txt', matrix, fmt='%s')   






