
from numpy import square
import time
import datetime
import pandas as pd
import seaborn as sns
from math import sin, cos, sqrt
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
# print(df.info())
#df.to_csv("trajectory_filtered.csv")
#corr_mat = df.corr()
#fig = sns.heatmap(corr_mat, square = True, robust=True).get_figure()
#fig.savefig("corr_mat.png")

# a = semi major axis (a)
# e = eccentricity (e)
# omega = longitude of ascending node (asc_node)
# w = argument of perihelion (perihelion_arg)
# i = inclination (inclination)
# m = mean anomaly (mean_anomaly)


def kepler_solver(df):
    """
    Solve the kepler equation to find position
    :param df: input dataframe
    :return: 3D coordinates
    """
    a, e, l, w, i, mean_motion, close_date = df['a'], df['e'], df['asc_node'], df['perihelion_arg'], df['inclination'],df['mean_motion'],df['close_date']
    given_date = "1995-07-23"
    c_date = time.strptime(close_date[1],"%Y-%m-%d")
    d_date = time.strptime(given_date,"%Y-%m-%d")
    c_date = datetime.datetime(c_date[0],c_date[1],c_date[2])
    d_date = datetime.datetime(d_date[0],d_date[1],d_date[2])
    t = (d_date-c_date).days
    print(t)
    e_anomaly = newton_method(e[1],t, mean_motion[1])
    coords_2d = compute_2d(a[1], e_anomaly,e[1])
    coords_3d = compute_3d(coords_2d, w[1], i[1], l[1])
    #return coords_3d
    return coords_3d


def newton_method(e,t,mean_motion):
    """
    Solve newton's method
    :param e: eccentricity
    :param t:time elapse
    :return: eccentric anomaly
    """
    
    assert isinstance(e, float) and 0 <= e <= 1, "eccentricity is not a float or between 0 and 1"
    m = t*mean_motion#get mean anomaly 
    assert isinstance(m, float) and m >= 0, "mean anomaly is not a float or less than 0"
    e_anomaly = m
   
    est_error = 1
    while abs(est_error) > 10**-6:
        f = e_anomaly - e*sin(e_anomaly) - m
        g = 1 - e*cos(e_anomaly)
        e_anomaly = e_anomaly - (f/g)
        est_error = f/g
  
    return e_anomaly


def compute_2d(a, e_anomaly, e):
    """
    Compute the 2D coordinates
    :param a: semi major axis
    :param e_anomaly: eccentric anomaly
    :param e: eccentricity
    :return: 2D coordinates
    """
    print(e_anomaly)
    x = a*(cos(e_anomaly) - e)
    y = a*sin(e_anomaly)*sqrt(1-pow(e, 2))

    coord = (x, y)
    return coord


def compute_3d(coords, w, i, l):
    """
    Compute the 3D coordinates
    :param coords:
    :param w: argument of perihelion
    :param i: inclination
    :param l: longitude of ascending node
    :return: Tuple of 3D coordinates (x, y, z)
    """
    # rotate by w
    x = cos(w)*coords[0] - sin(w)*coords[1]
    y = sin(w)*coords[0] + cos(w)*coords[1]
    # rotate by i (inclination)
    y = cos(i) * coords[1]
    z = sin(i)*coords[1]
    # rotate by longitude of ascending node
    xtemp = x
    x = cos(l)*xtemp - sin(l)*y
    y = sin(l)*xtemp + cos(l)*x
    return tuple([x, y, z])


coords = kepler_solver(df)

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(coords[0], coords[1], coords[2], "ro")
#ax.grid(False)
#plt.show()
print(coords)



