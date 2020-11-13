import numpy as np
import random 
import math
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages



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
    x = []
    y = []
    r = []
    rgb =[]
    for i in range(len(sensor)):
        x.append(sensor[i][0])  
        y.append(sensor[i][1])
        r.append(distance[i])
        rgb.append(np.random.rand(3,))
    dict = {"x": x,"y": y, "r" : r,"rgb": rgb}
    data= pd.DataFrame(data=dict)
    return data


def circle(sensor, distance):
    circle = plt.Circle((sensor[0], sensor[1]), distance)
    return circle

def draw_grafik(data, wawe, deneme):
    plt.figure()
    ax = plt.gca()
    for a, b, size, color in zip(data["x"], data["y"], data["r"], data["rgb"]):

    # plot circles using the RGBA colors
        circle = plt.Circle((a, b), size, color=color, fill=False)
        ax.add_artist(circle)
    
    plt.scatter(data["x"],data["y"], color=data["rgb"], marker = 'x')
    ax.scatter(wawe_points[0],wawe_points[1],marker = 'o')
    plt.scatter(data["x"],data["y"],s=0, facecolors='none')
    plt.grid()
    plt.title("deneme {}".format(deneme))
    export_pdf.savefig()
    plt.close


def kesişim(data):
    
    
    
    pass





with PdfPages(r'plot.pdf') as export_pdf:

    file = open("deneme ", "w")

    for deneme in range(10):

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
        draw_grafik(data_sensor, wawe_points, deneme)
        file.write("deneme sayısı :" +  str(deneme) + "\n")
        file.write("TİME P : \n" )
        file.writelines(["%s\n" % item  for item in time_p])
        file.write("TİME S : \n")
        file.writelines(["%s\n" % item  for item in time_s])
        file.write("distances_hipo: \n")
        file.writelines(["%s\n" % item  for item in distances_hipo])
        file.write("SENSOR : \n")
        np.savetxt(file, data_sensor.values[:,:-2], fmt='%d', delimiter="\t", header="X\tY")
        file.write("yarıçaplar delta için : \n")
        file.writelines(["%s\n" % item  for item in data_sensor["r"]])
        print(data_sensor)
        file.write("WAWE:  \n")
        file.write(str(wawe_points[0]) + " " + str(wawe_points[1]) + " \n")
        file.write("  \n  ")



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





