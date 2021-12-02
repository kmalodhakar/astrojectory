
from numpy import square
import datetime
import pandas as pd
import seaborn as sns
from math import sin, cos, asin, acos, sqrt, pi
import matplotlib.pyplot as plt


main_df = pd.read_csv("nasa.csv")

main_df.rename(lambda x: x.lower().strip().replace(" ", "_"), inplace=True, axis="columns")
# print(main_df.info())

df = main_df.drop(["name", "absolute_magnitude", "miles_per_hour", "epoch_date_close_approach", "miss_dist.(astronomical)", \
                            "miss_dist.(lunar)",  "miss_dist.(kilometers)", "miss_dist.(miles)", "orbit_determination_date",\
                            "minimum_orbit_intersection","orbiting_body", "equinox", "epoch_osculation", "est_dia_in_m(max)","est_dia_in_m(min)", \
                            "jupiter_tisserand_invariant", "aphelion_dist", "perihelion_time" , "hazardous","est_dia_in_miles(max)" , \
                            "est_dia_in_miles(min)", "est_dia_in_feet(max)", "est_dia_in_feet(min)", "relative_velocity_km_per_hr"\
                            ], axis="columns")

df = df.rename({ "neo_reference_id" : "id", "close_approach_date" : "close_date", "relative_velocity_km_per_sec" : "rel_vel",\
            "orbit_uncertainity" : "orbit_wt",  "asc_node_longitude" : "asc_node", "eccentricity" : "e", "semi_major_axis" : "a"}, axis = "columns")
df["est_dia"] = (df["est_dia_in_km(max)"] + df["est_dia_in_km(min)"])/2
df  = df.drop(["est_dia_in_km(max)","est_dia_in_km(min)"], axis ="columns")

#print(df.head())

def time_prediction(df):
    a, e, w, i,l, mean_motion, data_mean_anomaly, close_date = df['a'], df['e'], df['perihelion_arg'], df['inclination'],df['asc_node'],df['mean_motion'], df['mean_anomaly'], df['close_date']
    coords_2d = (-1.2214676848667936, -0.6847940907836357)
    t = compute_2d_coords_to_time(coords_2d,e[1],a[1],mean_motion[1],data_mean_anomaly[1])
    print(t)
    c = datetime.datetime.strptime(close_date[1],"%Y-%m-%d")
    d = datetime.timedelta(days=int(t))
    date_t = (c+d).strftime("%Y-%m-%d")
   
    #t = compute_3d_coords_to_time(coords_3d,e[1],a[1],w[1],i[1],l[1],mean_motion[1],data_mean_anomaly[1])
    return date_t

def compute_2d_coords_to_time(coords_2d,e,a,mean_motion,data_mean_anomaly):
    '''
        a: semi major axis
        coords_2d:2D coordinates of the planet
        e: eccentricity
        mean_motion: mean_motion
    '''
    assert isinstance(coords_2d,tuple) and 0 <= e <= 1
    (x,y) = coords_2d
    e_anomaly = acos(x/a+e)
    m_anomaly = e_anomaly - e*sin(e_anomaly)
    while data_mean_anomaly-m_anomaly>=2*pi:
        e_anomaly += 2*pi
        m_anomaly = e_anomaly - e*sin(e_anomaly)

    #print(m_anomaly)
    #assert y == a*sin(e_anomaly)*sqrt(1-pow(e, 2))
    print(e_anomaly)
    t = m_anomaly/mean_motion
    return t

def compute_3d_coords_to_time(coords_3d,e,a,w,i,l,mean_motion,data_mean_anomaly):
    '''
        a: semi major axis
        coords_3d:3D coordinates of the planet
        w: argument of perihelion
        i: inclinatio
        e: eccentricity
        l:longitude of ascending node
        mean_motion: mean_motion
    '''
    y = coords_3d[2]/sin(i)
    tempy = y*cos(i)
    tempx = (coords_3d[0]+sin(l)*tempy)/cos(l)
    x = (tempx + sin(w)*y)/cos(w)
    coords_2d = (x,y)#convert 3-d coords to 2-d coords
    return compute_2d_coords_to_time(coords_2d,e,a,mean_motion,data_mean_anomaly)
   

print(time_prediction(df))

