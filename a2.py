import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

def seq_array(array_shape, s = 1.):
    """ Function that generates a sequential array with the shape provided by
        the user and starting value.
        
        Parameters
        ----------
        array_shape: tuple
            The shape of the output array (rows, columns)
        s: int of float
            The starting value (defaults to 1.0)
            
        
        Returns
        -------
        sequential_array: numpy array
            A numpy array with the specified shape and a sequential sequence
    """
    
    sequential_array = np.arange(s, s + np.prod(array_shape), 1)
    # np.prod calculates the product of array elements
    # From line 42, sequential_array is a 1D array

    return sequential_array.reshape(array_shape)
    # np.reshape allows for the 1D array generated in line 42 to be transformed
    # into a 2D array



def determine_hub_speed_np(v_meas, h_meas, h_hub, alpha):
    """ Function to correct the wind speed for differences between  velocity
        measurement height and the hub height, using NumPy arrays.

    Parameters
    ----------
    v_meas: np.ndarray
        Wind speeds measured at various heights.
    h_meas: float
        Height at which wind speed is measured.
    h_hub: float
        Height of the hub.
    alpha: float
        Correlation coefficient.

    Returns
    -------
    v_hub: np.ndarray
        Wind speeds at the hub height.
    """
    
    # Calculate the correction factor for each element in v_meas
    correction_factor = (h_hub / h_meas) ** alpha
    # Apply the correction factor to v_meas using NumPy element-wise operations
    v_hub = v_meas * correction_factor
    
    return v_hub


def uv_to_speed_direction(u, v):
    """
    Convert wind speeds from u and v components into speed and direction.

    Parameters:
    u (float): East-West component (towards east positive) m/s.
    v (float): North-South component (towards north positive) m/s.

    Returns:
    speed_and_direction : A tuple containing wind speed (m/s) and wind
    direction (in degrees, wind from the north = 0 or 360 degrees).
    """
    # Calculate wind speed
    speed = np.sqrt(u**2 + v**2)
    
    # Calculate wind direction in degrees using arctan2
    direction_rad = np.arctan2(-u, -v)  # Negative signs to account for wind direction convention
    direction_deg = (np.degrees(direction_rad) + 360) % 360  # Convert to degrees (0-360)
    
    speed_and_direction = (speed, direction_deg)
    
    return speed_and_direction


def generate_windfarm_power_curve(power_curve_filename, turbine_number):
    wind_speed_array = np.loadtxt(power_curve_filename, skiprows=1, delimiter=",")
    wind_speed_array[:, 1] *= (turbine_number/1000)
    
    x_array = wind_speed_array[:, 0]
    y_array = wind_speed_array[:, 1]
    
    fig,ax = plt.subplots()
    ax.plot(x_array, y_array)
    fig.suptitle('Wind Farm Power Curve', fontsize=16)
    ax.set(xlabel="Wind_Speed", ylabel="Power (MW)")
    ax.grid()
    plt.show()
   
    return wind_speed_array


def generate_time_wind_power(wind_filename, power_filename):
    """ Function description
    
    Parameters
    ----------
    wind_filename: TYPE
        DESCRIPTION.
    power_filename: TYPE
        DESCRIPTION.
    
    Returns
    -------
    NAME: tuple
        A tuple of three 1D numpy arrays. The first member of the tuple is an array
        of times with datetime type with time in UTC. The second member is wind speed.
        The final member is power float type containing values for power generation in MW.

    """

    # Read wind data from the wind file
    time_data = np.loadtxt(wind_filename, dtype=str, delimiter=',', skiprows=1)
    u_data = np.loadtxt(wind_filename, dtype=float, delimiter=',', skiprows=1, usecols=(1))
    v_data = np.loadtxt(wind_filename, dtype=float, delimiter=',', skiprows=1, usecols=(2))
    power_data = np.loadtxt(power_filename, dtype=float, delimiter=',', skiprows=1, usecols=(1))
    
    # Read power data from the power file
    
    # times array    
    datetime_strings = time_data[:, 0]
    # Convert the date strings to datetime objects
    t_ar = np.array([datetime.strptime(date_str, '%d/%m/%Y %H:%M') for date_str in datetime_strings])

    # wind speed array
    wind_speed = np.sqrt(np.square(u_data) + np.square(v_data))
    # corrected wind speed:
    w_ar = determine_hub_speed_np(wind_speed, 10, 80, 0.143)
    
    # create power array
    p_ar = np.array(power_data)
        
    return (t_ar, w_ar, p_ar)



def wind_farm_figure(windfarm_curve, times, wind, power):

    plt.figure(1)
    plt.plot(power, wind)
    plt.plot(windfarm_curve)
    
    
class Site:
    def __init__(self, alpha, rho, h_meas):
        """
        Initialize a Site object with the provided site variables.

        Parameters:
        alpha (float): Correlation coefficient.
        rho (float): The air density.
        h_meas (float): Height at which the wind speed is measured.
        """
        self.alpha = alpha
        self.rho = rho
        self.h_meas = h_meas
        self.v_meas = 0.0  # Initialize the measured wind speed to 0.0

    def get_alpha(self):
        """Returns the correlation coefficient."""
        return self.alpha

    def get_rho(self):
        """Returns the air density."""
        return self.rho

    def get_height(self):
        """Returns the height of the measurement."""
        return self.h_meas

    def get_meas_speed(self):
        """Returns the wind speed measurement."""
        return self.v_meas

    def set_meas_speed(self, u, v):
        """
        Calculates and sets the measured speed based on u and v measurements.

        Parameters:
        u (float): The wind speed measurement in the u direction (east-west).
        v (float): The wind speed measurement in the v direction (north-south).
        """
        self.v_meas = (u**2 + v**2)**0.5
    

class Turbine:
    def __init__(self, h_hub, r, omega, curve_coeffs, speeds):
        self.h_hub = h_hub
        self.r = r
        self.omega = omega
        self.curve_coeffs = curve_coeffs
        self.speeds = speeds
        self.v_hub = 0  # Initialise hub speed to 0

    def get_hub_speed(self):
        return self.v_hub

    def determine_hub_speed(self, site):
        h_meas = site.h_meas
        alpha = site.alpha
        v_meas = site.v_meas  # Update to use v_meas

        self.v_hub = v_meas * (self.h_hub / h_meas) ** alpha

    def cap_hub_speed(self):
        v_rated = self.speeds[2]  # Rated speed
        if self.v_hub > v_rated:
            self.v_hub = v_rated

    def determine_windpower(self, site):
        v_hub = self.v_hub
        rho = site.rho

        p_wind = (0.5 * rho * math.pi * (self.r ** 2) * v_hub ** 3) / 1000
        return p_wind

    def determine_mech_coef(self):
        lambda_ratio = self.r * self.omega / self.v_hub
        coeffs = self.curve_coeffs
        cp = sum(ai * lambda_ratio ** i for i, ai in enumerate(coeffs, start=1))
        return cp

    def determine_mech_power(self, site):
        v_hub = self.v_hub
        rho = site.rho

        if v_hub == 0:
            return 0.0  # If v_hub is 0, power is 0
        else:
            p_mech = self.determine_windpower(site) * self.determine_mech_coef()
            return p_mech

def determine_total_energy(turbine, site, wind_filename):    
    # Initialise total energy
    total_energy = 0.0
    u_data = np.loadtxt(wind_filename, dtype=float, delimiter=',', skiprows=1, usecols=(1))
    v_data = np.loadtxt(wind_filename, dtype=float, delimiter=',', skiprows=1, usecols=(2))
    
    # Loop through each hour and calculate energy
    for i in range(len(u_data)):
        site.set_meas_speed(u_data[i], v_data[i])  # Set wind speed from the file
        turbine.determine_hub_speed(site)  # Determine hub speed
        turbine.cap_hub_speed()  # Cap hub speed as needed
        p_mech = turbine.determine_mech_power(site)  # Determine mechanical power
        p_elec = p_mech * 0.9  # Convert to electrical power with 90% efficiency
        total_energy += p_elec  # Accumulate total energy
        
    return total_energy


wind_filename = 'capital_wind.csv'
power_filename = 'capital_gen.csv'




'''
generate_time_wind_power(wind_filename, power_filename)
windfarm_curve = generate_windfarm_power_curve(power_filename, 67)
times = generate_time_wind_power(wind_filename, power_filename)[0]
wind = generate_time_wind_power(wind_filename, power_filename)[1]
power = generate_time_wind_power(wind_filename, power_filename)[2]
'''


