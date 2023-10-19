import numpy as np

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

    return sequential_array
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





