import numpy as np
import random 
import math
from scipy import signal
from scipy.signal import find_peaks



edge = 100
wawe_points = []
sensor_point =[]
def build_matrix(edge):
    matrix =  np.empty([100,100],str)
    matrix[:] = "."
    # d1 = random.randint(1, 99)
    # d2 = random.randint(1, 99)
    # d3 = random.randint(1, 99)
    # d4 = random.randint(1, 99)
    matrix[0][80] = "A"  #Top level 
    matrix[99][73] = "A" # bottom level
    matrix[64][0] = "A"   #left side
    matrix[38][99] = "A"   #right side
    return matrix

def wawe_create(matrix):
    # wawe_x= random.randint(0,99)
    # wawe_y= random.randint(0,99)
    matrix[74][20] = "W"
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
    wawe_signal = signal.unit_impulse(100)
    return wawe_signal

def delta(time_p, time_s, time_freq):
    signal_array = []
    for i in range(len(time_p)):
        base_signal = create_signal()
        delta_step = round(time_p[i]/time_freq)
        base_signal[0] = 0
        base_signal[delta_step]=1
        delta_step = round(time_s[i]/time_freq )
        base_signal[delta_step]=1
        signal_array.append(base_signal)
    return signal_array

def delta_time_dif(deltas, time_freq):
    time_array=[]
    for i in range(len(delta_to_sensor)):
        delta = delta_to_sensor[i]
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
    




matrix = build_matrix(edge)
wawe_create(matrix)
wawe_points = wawe_point(matrix, "W")
sensor_points = sensor_points(matrix, "A")
distances_hipo = distance_normal(matrix, wawe_points, sensor_points)
time= time_for_distance(distances_hipo)
time_p = time[0]
time_s = time[1]
delta_to_sensor = delta(time_p,time_s, 0.001)
peak_time = delta_time_dif(delta_to_sensor, 0.001)
distance_delta = distance_for_delta(peak_time)




print("wawe: " , wawe_points)
print("sensors: " , sensor_points)
print("distance: " , distances_hipo)
print("timeP: " , time_p)
print("timeS: " , time_s)
print("deltas : " , delta_to_sensor)
print("peak time : " , peak_time)
print("distance for delta : " , distance_delta)
np.savetxt('output.txt', matrix, fmt='%s')   





