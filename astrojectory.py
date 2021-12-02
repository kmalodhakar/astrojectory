#import animate
from numpy import square
import time
import datetime
import pandas as pd
import seaborn as sns
from math import sin, cos, sqrt,pi
import matplotlib.pyplot as plt

class asteroid:
    def __init__(self,a,e,omega,w,i,period, close_date, given_dates):
        # a = semi major axis (a)
        # e = eccentricity (e)
        # omega = longitude of ascending node (asc_node)
        # w = argument of perihelion (perihelion_arg)
        # i = inclination (inclination)
        #period = orbital period (orbital_period)
        #close date = date of periapsis
        # given dates list of given dates
        self._a = a
        assert isinstance(e, float) and 0 <= e <= 1, "eccentricity is not self._a float or between 0 and 1"
        self._e = e
        self._omega = omega
        self._w = w
        self._i = i
        assert isinstance(period,float) and period >=0
        self._period = period
        self._close_date = close_date
        self._given_dates = given_dates

    def kepler_solver(self):
        """
        Solve the kepler equation to find position
        """
        coords_list =[]
        for d in given_dates:
            interval = compute_interval(self._close_date,d)
            e_anomaly = self.newton_method(interval)
            # print(interval)
            coords_2d = self.compute_2d(e_anomaly)
            coords_list.append(self.compute_3d(coords_2d))
        return coords_list


    def newton_method(self,interval, atol = 10**-6):
        """
        Solve newton's method
        :time interval between two dates
        :return: eccentric anomaly
        """
        assert isinstance(interval, int) and interval >= 0, "mean anomaly is not self._a float or less than 0"
        m = interval * ((2 * pi) / self._period) 
        e_anomaly = m

        est_error = 1
        while abs(est_error) > atol:
            f = e_anomaly - self._e*sin(e_anomaly) - m
            g = 1 - self._e*cos(e_anomaly)
            e_anomaly = e_anomaly - (f/g)
            est_error = f/g
    
        return e_anomaly


    def compute_2d(self,e_anomaly):
        """
        Compute the 2D coordinates
        :param e_anomaly: eccentric anomaly
        :return: 2D coordinates
        """
        # print(e_anomaly)
        x = self._a*(cos(e_anomaly) - self._e)
        y = self._a*sin(e_anomaly)*sqrt(1-pow(self._e, 2))

        coord = (x, y)
        return coord


    def compute_3d(self,coords):
        """
        Compute the 3D coordinates
        :param coords:
        :return: Tuple of 3D coordinates (x, y, z)
        """
        # rotate by w
        x = cos(self._w)*coords[0] - sin(self._w)*coords[1]
        y = sin(self._w)*coords[0] + cos(self._w)*coords[1]
        # rotate by i (inclination)
        y = cos(self._i) * coords[1]
        z = sin(self._i)*coords[1]
        # rotate by longitude of ascending node
        xtemp = x
        x = cos(self._omega)*xtemp - sin(self._omega)*y
        y = sin(self._omega)*xtemp + cos(self._omega)*x
        return tuple([x, y, z])

def compute_interval(periapsis_date,given_date):
    """
    Given two dates, computes the number of days between them
    """
    day1 = time.strptime(periapsis_date,"%Y-%m-%d")
    day2 = time.strptime(given_date,"%Y-%m-%d")
    day1 = datetime.datetime(day1[0],day1[1],day1[2])
    day2 = datetime.datetime(day2[0],day2[1],day2[2])
    t = (day2-day1).days
    # print(t)
    return t

def data_preprocess(main_df):
    main_df.rename(lambda x: x.lower().strip().replace(" ", "_"), inplace=True, axis="columns")
    # print(main_df.info())

    df = main_df.drop(["name", "absolute_magnitude", "miles_per_hour", "epoch_date_close_approach", "miss_dist.(astronomical)", \
                            "miss_dist.(lunar)",  "miss_dist.(kilometers)", "miss_dist.(miles)", "orbit_determination_date",\
                            "minimum_orbit_intersection","orbiting_body", "equinox", "epoch_osculation", "est_dia_in_m(max)","est_dia_in_m(min)", \
                            "jupiter_tisserand_invariant", "aphelion_dist", "perihelion_time" , "hazardous","est_dia_in_miles(max)" , \
                            "est_dia_in_miles(min)", "est_dia_in_feet(max)", "est_dia_in_feet(min)", "relative_velocity_km_per_hr", "mean_motion"\
                            ], axis="columns")

    df = df.rename({ "neo_reference_id" : "id",
                     "close_approach_date" : "close_date",
                     "relative_velocity_km_per_sec" : "rel_vel",
                    "orbit_uncertainity" : "orbit_wt", 
                    "asc_node_longitude" : "asc_node",
                    "eccentricity" : "e", 
                    "semi_major_axis" : "a"
                    }, axis = "columns")
    df["est_dia"] = (df["est_dia_in_km(max)"] + df["est_dia_in_km(min)"])/2
    df  = df.drop(["est_dia_in_km(max)","est_dia_in_km(min)"], axis ="columns")
    # print(df.info())
    # df.to_csv("trajectory_filtered.csv")
    return df


####### driver code  #########
if __name__ == "__main__":
    main_df = pd.read_csv("nasa_orig.csv")
    df = data_preprocess(main_df)
    # corr_mat = df.corr()
    # fig = sns.heatmap(corr_mat, square = True, robust=True).get_figure()
    # fig.savefig("corr_mat.png")
    given_dates = ["2021-12-01"]
    q = 1
    ast1 = asteroid(a = df['a'][q], e = df['e'][q], 
                    omega = df['asc_node'][q],
                    w =  df['perihelion_arg'][q], 
                    i = df['inclination'][q], 
                    period = df['orbital_period'][q], 
                    given_dates = given_dates, close_date = df["close_date"][q])
    q = 2
    ast2 = asteroid(a = df['a'][q], e = df['e'][q], 
                    omega = df['asc_node'][q],
                    w =  df['perihelion_arg'][q], 
                    i = df['inclination'][q], 
                    period = df['orbital_period'][q], 
                    given_dates = given_dates, close_date = df["close_date"][q])
    coords1 = ast1.kepler_solver()
    coords2 = ast2.kepler_solver()
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.plot3D(coords[0], coords[1], coords[2], "ro")
    #ax.grid(False)
    #plt.show()
    print(coords1)
    print(coords2)



