
import numpy as np
import random 
import math
from numpy.core.numeric import False_
from numpy.lib.function_base import append
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from math import cos, isnan, nan, sin, pi, sqrt, atan2
import unicodedata
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.collections import LineCollection



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
    # L model 
    matrix[d1][99] = "A"  #Top level 
    matrix[d2][99] = "A" # bottom level
    matrix[0][d3] = "A"   #left side
    matrix[0][d4] = "A"   #right side
    # kare model
    # matrix[0][d1] = "A"  #Top level 
    # matrix[99][d2] = "A" # bottom level
    # matrix[d3][0] = "A"   #left side
    # matrix[d4][99] = "A"   #right side
    # -- model 
    # matrix[d1][99] = "A"  #Top level 
    # matrix[d2][99] = "A" # bottom level
    # matrix[d3][99] = "A"   #left side
    # matrix[d4][99] = "A"   #right side
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
    # wawe_signal[0]=0
    step = 0
    while (step != len(wawe_signal)):
        wawe_signal[step]= random.random()
        step=step+1
    return wawe_signal

def sig_wawe():                 # sin wawe create   sig değerleri çok büyük düzelt aq
    
    t = np.linspace(0, 500, 1000, endpoint=False)
    t=t[::-1]

    sig1 = np.sin(2 * np.pi * t)
    sig2 = np.sin(1 * np.pi/2 * t)
    sig3 = np.sin(1.5 * np.pi * t)
    sig4 = np.sin(2.5 * np.pi/3 * t)
    sig5 = np.sin(0.5 * np.pi*3 * t)

    sig= sig1+sig2+sig3+sig4+sig5

    return sig


def delta(time_p, time_s, time_freq):
    delta = []

    for i in range(len(time_p)):
        wawe=create_signal()
        P_step = round(time_p[i]/time_freq)
        S_step = round(time_s[i]/time_freq )
        #P wawe
        wawe[0:P_step]=0
        wawe[P_step] =np.random.default_rng().uniform(low=0.5, high=1, size=1)
        array = np.random.default_rng().uniform(low=0.5, high=1, size=(S_step-P_step)//2)


        range_a=len(array)
        wawe[P_step+1:P_step+1+range_a] = array
        # S wawe
        wawe[P_step+1+range_a:S_step]=np.random.rand((S_step-P_step)-1-range_a)/10
        wawe[S_step]= np.random.default_rng().uniform(low=0.5, high=1, size=1)
        array = np.random.default_rng().uniform(low=1, high=1.5, size=(S_step-P_step)//2)
        wawe[S_step+1:S_step+1+range_a] = array
        wawe[S_step+1+range_a:]= np.random.rand(len(wawe)-S_step-1-range_a)/10
        # sig = sig_wawe()
        # wawe= wawe*sig
        # print(P_step)
        # print(S_step)
        for i in range(len(wawe)):
            if i==0:
                pass
            elif i%2==1:
                wawe[i]=wawe[i]*(-1)
            else: 
                wawe[i]=wawe[i]*1
        # plt.plot(wawe)
        # plt.show()
        delta.append(wawe)
    dict = {"delta": delta}
    deltas = pd.DataFrame(data=dict)
    return deltas
#dalgaları düzenle gürültü küçült

def delta_time_dif(deltas, time_freq):
    peak=[]
    time_array=[]
    sta_all=[]
    lta_all=[]
    for index in range(len(deltas)):
        delta = deltas["delta"][index]
        # plt.plot(delta)
        # plt.show()
        # plt.close()
        ### sta
        sta=[]
        for i in range(len(delta)):
            part=delta[i:i+10]
            p_avg=0
            for a in range(len(part)):
                p1=abs(part[a])
                p_avg= p_avg+p1
            if p_avg==0:
                p_avg=0
            else:
                p_avg=p_avg/10
            sta.append(p_avg)
                # plt.plot(delta)
                # plt.show()
                # plt.close()
        # peak_diff_sta= peak[1]-peak[0]
        # time_diff_sta= peak_diff_sta*time_freq
        # time_array_sta.append(time_diff_sta)
        sta_all.append(sta)
        # plt.plot(sta_all[index])
        # plt.show()
        # plt.close()
        ## lta
        lta=[]
        for i in range(len(delta)):
            part=delta[i:i+50]
            p_avg=0
            p_avg= 0
            for a in range(len(part)):
                p1=abs(part[a])
                p_avg= p_avg+p1
            if p_avg==0:
                p_avg=0
            else:
                p_avg=p_avg/50
            lta.append(p_avg)
                # plt.plot(delta)
                # plt.show()
                # plt.close()
        lta_all.append(lta)
        sta_lta_ratio=[]
        for i in range(len(delta)):
            sta_ratio= sta[i]
            lta_ratio= lta[i]
            if lta_ratio==0:
                abc=sta_ratio
            else:
                abc= sta_ratio/lta_ratio
            
            sta_lta_ratio.append(abc)
        # plt.close()
        # plt.plot(sta_lta_ratio)
        # plt.show()
        # plt.close()
        for i in range(len(sta_lta_ratio)):
            if sta_lta_ratio[i+1]>=1 and sta_lta_ratio[i]<=1:
                check=[]
                for j in range(20):
                    if sta_lta_ratio[i+1]>=sta_lta_ratio[i-j]:
                        check.append(True)
                    else:
                        check.append(False)
                if False not in check:
                    peak.append(i+1)

            if len(peak)==2:
                power=peak[0]//50
                if power==0:
                    power=1
                if abs(peak[0]-peak[1])<=5*power:
                    peak=[peak[0]]
                else:
                    break
        # plt.close()
        # plt.plot(sta_lta_ratio)
        # l1 = [(peak[0], 0), (peak[0], sta_lta_ratio[peak[0]])]
        # l2 = [(peak[1], 0), (peak[1], sta_lta_ratio[peak[1]])]
        # lc1 = LineCollection([l1, l2], color=["k","red"], lw=1)

        # plt.gca().add_collection(lc1)

        # plt.show()
        # plt.close()
        # print(peak)

        peak_diff = peak[1] - peak[0]
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

def draw_grafik(data, wawe, deneme, data2, method):
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
    plt.title("deneme {0} , method {1}".format(deneme,method))
    export_pdf.savefig()
    plt.close

def predic_wawe(array1, array2):                # mod ile
    x_array=[]
    y_array=[]
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            if np.isnan(array1[i][j]) :
                pass
            else:
                x_array.append(array1[i][j])
                y_array.append(array2[i][j])
    data_x = point_find(x_array, "x")          # x ve y arrayları
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

def point_find_median(array):            # mod ve median ile
    array.sort()
    mean = np.median(array)
    index= []
    for i in range(len(array)):
        if (abs(mean-array[i])<= 5):
            pass
        else :
            index.append(i)

    array=np.delete(array, index)
    return array

def predic_wawe_median(array1, array2):              # mod ve median ile
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


def predic_wawe_k_means(array1, array2):
    xy_array = []
    x_array=[]
    y_array=[]
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            if np.isnan(array1[i][j]) :
                pass
            else:
                x_array.append(array1[i][j])
                y_array.append(array2[i][j])
                array = [array1[i][j],array2[i][j]]
                xy_array.append(array)
    kmedoids = KMedoids(n_clusters=2, random_state=0).fit(xy_array)
    clf = LocalOutlierFactor(n_neighbors=2)
    xy = {"x":x_array, "y":y_array, "P value": kmedoids.labels_}
    data= pd.DataFrame(data=xy)
    data=data.sort_values(by=["P value"],ascending=False)
    #data = data[data['P value'] == 0]
    return data


def predic_wawe_k_means2(array1, array2):
    xy_array = []
    x_array=[]
    y_array=[]
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            if np.isnan(array1[i][j]) :
                pass
            else:
                x_array.append(array1[i][j])
                y_array.append(array2[i][j])
                array = [array1[i][j],array2[i][j]]
                xy_array.append(array)
    kmedoids = KMedoids(n_clusters=2, random_state=0).fit(xy_array)
    clf = LocalOutlierFactor(n_neighbors=5)
    #print(kmedoids.labels_)
    #print(kmedoids.cluster_centers_)
    abc= clf.fit_predict(xy_array)
    #print(abc)
    for i in range(len(abc)):
        if abc[i] == -1:
            if kmedoids.labels_[i] ==0:
                kmedoids.labels_[i] = 1
            else:
                kmedoids.labels_[i] = 0 
    # print(kmedoids.labels_)
    xy = {"x":x_array, "y":y_array, "P value": kmedoids.labels_}
    data= pd.DataFrame(data=xy)
    data=data.sort_values(by=["P value"],ascending=False)
    #,
    # data = data[data['P value'] == 0]
    return data

def predic_wawe_k_means3(array1, array2):
    xy_array = []
    x_array=[]
    y_array=[]
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            if np.isnan(array1[i][j]) :
                pass
            else:
                x_array.append(array1[i][j])
                y_array.append(array2[i][j])
                array = [array1[i][j],array2[i][j]]
                xy_array.append(array)
    index= outlier_index(xy_array)

    xy_array=np.delete(xy_array, index, axis=0)
    y_array=np.delete(y_array, index)
    x_array=np.delete(x_array, index)
    label = outlier_label(xy_array)
    xy = {"x":x_array, "y":y_array, "P value": label}
    data= pd.DataFrame(data=xy)
    data=data.sort_values(by=["P value"],ascending=False)
    # print(len(data))
    # print(data["x"][0])
    # for i in range(len(data)):
    #     distance =  math.hypot(data["x"][i] - center[0][0], data["y"][0] - center[0][1])
    #     if abs(distance_avg-distance) >= 5:
    #         kmedoids.labels_[i]=1
    # data = data[data['P value'] == 0]
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



def outlier_index(array):
    kmedoids = KMedoids(n_clusters=1, random_state=0).fit(array)
    center = kmedoids.cluster_centers_
    distance_avg = kmedoids.inertia_/len(array)
    index=[]
    for i in range(len(array)):                  # first outlier 
        distance =  math.hypot(array[i][0] - center[0][0], array[i][1] - center[0][1])
        if distance >= distance_avg:
            kmedoids.labels_[i]=1
            index.append(i)
    return index

def outlier_label(array):
    kmedoids = KMedoids(n_clusters=1, random_state=0).fit(array)
    center = kmedoids.cluster_centers_
    distance_avg = kmedoids.inertia_/len(array)
    for i in range(len(array)):                  # first outlier 
        distance =  math.hypot(array[i][0] - center[0][0], array[i][1] - center[0][1])
        if distance >= distance_avg:
            kmedoids.labels_[i]=1

    return kmedoids.labels_

## apporiximate find
def find_nearest_best(data, wawe):
    arrayX = np.asarray(data["x"])
    arrayY = np.asarray(data["y"])
    distance_avg= 5
    distance_best=1
    check=[]
    distances=[]
    for i in range(len(arrayX)):
        xDistance=abs(arrayX[i]-wawe[0])
        ydistance=abs(arrayY[i]-wawe[1])
        Pdistance = sqrt((xDistance**2)+(ydistance**2))
        distances.append(Pdistance)
        if Pdistance<=distance_best:
            check.append("en yakın")
        elif Pdistance<=distance_avg:
            check.append("yakın")
        else:
            check.append("uzak")
    data["approximate"] = distances
    return check


def count(data):
    en_yakın_count = data.count("en yakın")
    yakın_count = data.count("yakın")
    uzak_count = data.count("uzak")
    count={"best points": en_yakın_count, "close points": yakın_count, "far points": uzak_count}
    return count

def close_point(data):
    if "en yakın" not in  data:
        if "yakın" in data:
            return 1
        else:
            return 0
    else:
        return 0

def best_point(data):
    if "en yakın" in data:
        return 1
    else:
        return 0
            

# deneme ortamları :  1 tane yukarıda, 2 tane L şeklinde, 1 tane random 50 şer  


with PdfPages(r'plot.pdf') as export_pdf:

    file = open("tek.txt", "w")

    ratio1 = 0
    ratio2 = 0
    ratio3 = 0
    ratio4 = 0
    c_ratio1=0
    c_ratio2=0
    c_ratio3=0
    c_ratio4=0
    e_count=0
    deneme=0
    while deneme+e_count<100:
        try:
            print(deneme)
            matrix = build_matrix(edge)
            wawe_create(matrix)
            wawe_points = wawe_point(matrix, "W")
            sensor_point = sensor_points(matrix, "A")
            distances_hipo = distance_normal(matrix, wawe_points, sensor_point)
            time= time_for_distance(distances_hipo)
            time_p = time[0]
            time_s = time[1]
            delta_to_sensor = delta(time_p,time_s, 0.0001)
            # plt.plot(delta_to_sensor["delta"][0], "o-")
            # plt.show()
            peak_time = delta_time_dif(delta_to_sensor, 0.0001)
            distance_delta = distance_for_delta(peak_time)
            data_sensor = data(sensor_point, distance_delta)
            result_matrix = intersection(data_sensor)
            result_x= result_matrix[0]
            result_y = result_matrix[1]
            # print(deneme)
            # print(data_sensor)
            # print("wawe", wawe_points)
            wawe_predict= predic_wawe(result_x,result_y)
            wawe_predict2 = predic_wawe_median(result_x,result_y)
            # wawe_predict3 = predic_wawe_k_means(result_x,result_y)
            # wawe_predict3_0 = wawe_predict3[wawe_predict3['P value'] == 0]
            # wawe_predict3_1 = wawe_predict3[wawe_predict3['P value'] == 1]
            wawe_predict4 = predic_wawe_k_means2(result_x,result_y)
            wawe_predict5 = predic_wawe_k_means3(result_x,result_y)
            draw_grafik(data_sensor, wawe_points, deneme, wawe_predict, "normal method")
            draw_grafik(data_sensor,wawe_points,deneme, wawe_predict2, " with median")
            # draw_grafik(data_sensor,wawe_points,deneme, wawe_predict3_0, "K-means cluster:0")
            # draw_grafik(data_sensor,wawe_points,deneme, wawe_predict3_1, "K-means cluster:1")
            # draw_grafik(data_sensor,wawe_points,deneme, wawe_predict3, "cluster ")
            draw_grafik(data_sensor,wawe_points,deneme, wawe_predict4, "K-means with outlier ")
            draw_grafik(data_sensor,wawe_points,deneme, wawe_predict5, "K-means center with outlier ")

            check1=find_nearest_best(wawe_predict,wawe_points)
            point_count1=count(check1)
            ratio= best_point(check1)
            close_ratio1=close_point(check1)
            c_ratio1=c_ratio1+close_ratio1
            ratio1 = ratio1+ratio

            check2=find_nearest_best(wawe_predict2,wawe_points)
            point_count2=count(check2)
            ratio= best_point(check2)
            close_ratio2=close_point(check2)
            c_ratio2=c_ratio2+close_ratio2
            ratio2 = ratio2+ratio


            check3=find_nearest_best(wawe_predict4,wawe_points)
            point_count3=count(check3)
            ratio= best_point(check3)
            close_ratio3=close_point(check3)
            c_ratio3=c_ratio3+close_ratio3
            ratio3 = ratio3+ratio

            check4=find_nearest_best(wawe_predict5,wawe_points)
            point_count4=count(check4)
            ratio= best_point(check4)
            close_ratio4=close_point(check4)
            c_ratio4=c_ratio4+close_ratio4
            ratio4 = ratio4+ratio

            file.write("DENEME SAYISI :" +  str(deneme) + "\n")
            # file.write("SENSOR : \n")
            # np.savetxt(file, data_sensor.values[:,:-2], fmt='%d', delimiter="\t", header="\tX\tY")
            # file.write("WAWE:    ")
            # file.write(str(wawe_points[0]) + "    " + str(wawe_points[1]) + " \n")
            # file.write("WAWE_PRED \n")
            # file.write("X :\n")
            # np.savetxt(file, result_x, fmt='%10.5f')
            # file.write("Y :\n")
            # np.savetxt(file, result_y, fmt='%10.5f')
            # file.write("\n")
            file.write("predict with mod\n")
            np.savetxt(file, wawe_predict, fmt='%d %d %d %f', delimiter="-------", header="\tX\tY\tP\tApproximate")
            file.write("\n")
            file.write(str(point_count1))
            file.write("\n")
            file.write("predict with mod and meadin\n")
            np.savetxt(file, wawe_predict2, fmt='%d %d %d %f', delimiter="\t", header="\tX\tY\tP\tApproximate")
            file.write("\n")
            file.write(str(point_count2))
            file.write("\n")
            # file.write("predict with Kmeans \n")
            # np.savetxt(file, wawe_predict3, fmt='%d', delimiter="\t", header="\tX\tY\tP")
            # file.write("\n")     
            file.write("predict with Kmeans with outlier \n")
            np.savetxt(file, wawe_predict4, fmt='%d %d %d %f', delimiter="\t", header="\tX\tY\tP\tApproximate")
            file.write("\n")
            file.write(str(point_count3))
            file.write("\n")
            file.write("predict with  one center Kmeans with outlier \n")
            np.savetxt(file, wawe_predict5, fmt='%d %d %d %f', delimiter="\t", header="\tX\tY\tP\tApproximate")
            file.write("\n")
            file.write(str(point_count4))
            file.write("\n")
            deneme=deneme+1
        except IndexError:
            e_count=e_count+1
            continue
        except ValueError or TypeError:
            continue

    file.write("method 1 best point ratio : " + str(ratio1))
    file.write("\n")
    file.write("method 1  close ratio : " + str(c_ratio1))
    file.write("\n")
    file.write("method 2 ratio : " + str(ratio2))
    file.write("\n")
    file.write("method 2  close ratio : " + str(c_ratio2))
    file.write("\n")
    file.write("method 3 ratio : " + str(ratio3))
    file.write("\n")
    file.write("method 1  close ratio : " + str(c_ratio3))
    file.write("\n")
    file.write("method 4 ratio : " + str(ratio4))
    file.write("\n")
    file.write("method 1  close ratio : " + str(c_ratio4))
    file.write("\n")
    file.write(str(e_count)+ " adet durumda çözüm yapılamamıştır.")
    all_best_ratio=[ratio1,ratio2,ratio3,ratio4,0]
    all_close = [c_ratio1,c_ratio2,c_ratio3, c_ratio4, 0]
    except_count=[0,0,0,0, e_count]
    labels = ['1. method', '2. method', '3. method', '4. method', 'unsolvable']
    fig, ax = plt.subplots()
    ax.bar(labels, all_best_ratio, label="Best point ratio" )
    for i in range(len(labels)-1):
        plt.text(i,all_best_ratio[i],all_best_ratio[i])
    ax.bar(labels, all_close, label="close point ratio")
    for i in range(len(labels)-1):
        plt.text(i,all_close[i],all_close[i])
    ax.bar(labels, except_count, label= "unsolvable" )
    plt.text(4,except_count[4],except_count[4])
    ax.set_ylabel("ratio")
    ax.set_title("solution rates")
    ax.set_ylim([0,deneme+e_count])
    ax.legend()
    export_pdf.savefig()
    plt.close
print("finish")




