import numpy as np
import matplotlib.pyplot as plt

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
    """
    Correct the wind speed at the hub height using NumPy arrays.

    Parameters:
    v_meas: np.ndarray
        Wind speeds measured at various heights.
    h_meas: float
        Height at which wind speed is measured.
    h_hub: float
        Height of the hub.
    alpha: float
        Correlation coefficient.

    Returns:
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

    return (speed, direction_deg)


def generate_windfarm_power_curve(power_curve_filename, turbine_number):
    wind_speed_array = np.loadtxt(power_curve_filename, skiprows=1, delimiter=",")
    wind_speed_array[:, 1] *= (turbine_number/1000)
    
    x_array = wind_speed_array[:, 0]
    y_array = wind_speed_array[:, 1]
    
    fig,ax = plt.subplots()
    ax.plot(x_array, y_array)
    fig.suptitle('Wind Farm Power Curve')
    ax.set(xlabel="Wind_Speed", ylabel="Power (MW)")
    ax.grid()
    plt.show()
   
    return wind_speed_array

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
        """
        Initialize a Turbine object with the provided turbine variables.

        Parameters:
        h_hub (float): Height of the hub.
        r (float): The radius of the turbine blades.
        omega (float): Angular velocity.
        curve_coeffs (tuple): Tuple of coefficients (a1, a2, ..., ak).
        speeds (tuple): Tuple of threshold speeds (cut-in, cut-out, and rated speeds).
        """
        self.h_hub = h_hub
        self.r = r
        self.omega = omega
        self.curve_coeffs = curve_coeffs
        self.speeds = speeds
        self.v_hub = 0.0

    def get_hub_speed(self):
        """Returns the hub speed."""
        return self.v_hub

    def cap_hub_speed(self):
        """Caps the hub speed at the rated speed."""
        if self.v_hub > self.speeds[2]:
            self.v_hub = self.speeds[2]

    def determine_windpower(self, site):
        """Returns the wind power."""
        v_hub = self.v_hub
        alpha = site.get_alpha()
        rho = site.get_rho()
        return 0.5 * rho * v_hub**3 * np.pi * self.r**2

    def determine_mech_coef(self):
        """Returns the mech coefficient."""
        v_hub = self.v_hub
        curve_coeffs = self.curve_coeffs
        return curve_coeffs[0] + curve_coeffs[1] * v_hub + curve_coeffs[2] * v_hub**2 + curve_coeffs[3] * v_hub**3 + curve_coeffs[4] * v_hub**4

    def determine_mech_power(self, site):
        """Returns the mech power."""
        v_hub = self.v_hub
        mech_coef = self.determine_mech_coef()
        return 0.5 * site.get_rho() * np.pi * self.r**2 * v_hub**3 * mech_coef

