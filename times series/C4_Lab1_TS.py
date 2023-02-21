import matplotlib.pyplot as plt
import numpy as np

def plot_series(time, series, fromat='-', start=0, end=None, label=None):
    """
    visualize time series data
    :param time: (array of int) contains the time steps
    :param series: (array of int) containt the measurement for each time step
    :param fromat: (string) line style when plotting the graph
    :param start: first step to plot
    :param end: last step to plot
    :param label: (list of strings) tag for the line
    :return:
    """
    # Ste up dimension of the graph figure
    plt.figure(figsize=(10, 6))

    #plot the time series data
    plt.plot(time[start:end], series[start:end], format)

    #Label the X-axis
    plt.xlabel('Time')

    #label the Y-axis
    plt.ylabel('Value')

    # if label:
    #     plt.legend(frontsize=14, labels=label)

    #Orverlay a grid on the graph
    plt.grid(True)

    #Draw the graph
    plt.show()

def trend(time, slope=0):
    """
    Generates synthetic data that follows a straight line given a slope value
    :param time: (array of int) contains the time steps
    :param slope: (float) determines the direction and steepness of the line
    :return: series (array of float) measurements tha follow a straight line
    """

    #compute the linear series given the slope
    series = slope*time

    return series

def seasonal_pattern(season_time):
    """
    Just an ard´bitrary patter tha could be changed anytime
    :param season_time: (array of float) contains the measurement per time step
    :return: data_patter (array of float) - contains reised measurement values according to the defined pattern
    """
    # Generate the values using an arbitrary pattern
    data_pattern = np.where(season_time<0.4, np.cos(season_time*2*np.pi), 1/np.exp(3*season_time))

    return data_pattern

def seanonality(time, period, amplitude=1, phase=0):
    """
    Repeat the same pattern at each period
    :param time: (array of int) - contains the time steps
    :param period: (int) - number of time steps before the pattern repeats
    :param amplitude: (int) peak measured value in a period
    :param phase: - number of time steps to shift the easured values
    :return: data_pattern (array of float) - seasonal data scaled by the defined amplitude
    """
    #define the measured values per period
    season_time = ((time+phase) % period) /period

    #generates the seasonal data scaled by the defined amplitude
    data_pattern = amplitude*seasonal_pattern(season_time)

    return data_pattern

def noise(time, noise_level=1, seed= None):
    """
    Generates a normally distributed noisy signal
    :param time: (array of int) contains time steps
    :param noise_level: (float) - scalling factor for the generazed signal
    :param seed: (int) - number generator seed for repeatability
    :return: noise (array of float) - the noisy signal
    """
    #Initialize the random number generator
    random_ = np.random.RandomState(seed)

    #Generate a random number for each time step and sclale by the noise level
    noise = random_.randn(len(time))*noise_level

    return  noise

#AUTOCORRELATION

def autocorrelation(time, amplitude, seed=None):
    """
    Generate autocorrelated data
    :param time: (array of int) - contains time steps
    :param amplitude: (float) - sclalling factor
    :param seed: (int) - number generator seed for repeatability
    :return: myarray (array of floats) - autocorrelazted data
    """
    # Initialize random number generator
    rdom = np.random.RandomState(seed)

    #Initialize array of random numbers equal to the length of given time steps plus 50
    myarray = rdom.randn(len(time) + 50)

    #set first 50 elements to a constant
    myarray[:50] = 100

    # define scaling factors
    phi1 = 0.5
    phi2 = -0.1

    # Autocorrelate element 51 onwarsds with the current measurement a (t-50) and (t-30), where t is the current time step
    for steps in range(50, len(time)+50):
        myarray[steps] += phi1*myarray[steps-50]
        myarray[steps] += phi2 * myarray[steps - 33]

    # Get the autocorrelated data and scale with the given amplitude-the first 50 elements of the original array is truncated because
    # those are just constant and not autocorrelated
    myarray = myarray[50:]*amplitude

    return myarray

def autocorrelation(time, amplitude, seed=None):
    """
    Generate autocorrelated data
    :param time: (array of int) - contains time steps
    :param amplitude: (float) - sclalling factor
    :param seed: (int) - number generator seed for repeatability
    :return: myarray (array of floats) - autocorrelazted data
    """
    # Initialize random number generator
    rdom = np.random.RandomState(seed)

    #Initialize array of random numbers equal to the length of given time steps plus 50
    myarray = rdom.randn(len(time) + 1)

    #Define the scaling factor
    phi = 0.8

    # define scaling factors
    phi1 = 0.5
    phi2 = -0.1

    # Autocorrelate element 51 onwarsds with the current measurement a (t-1), where t is the current time step
    for steps in range(1, len(time)+1):
        myarray[steps] += phi*myarray[steps-1]


    # Get the autocorrelated data and scale with the given amplitude-the first 50 elements of the original array is truncated because
    # those are just constant and not autocorrelated
    myarray = myarray[1:]*amplitude

    return myarray

#IMPULSE
def inpulse(time, num_inpulses, amplitude = 1, seed=None):
    """
    Generate autocorrelated data
    :param time: (array of int) - contains time steps
    :param num_inpulses: (int) - number of inpulses to generate
    :param amplitude: (float) - sclaling factor
    :param seed: (int) - number generator seed for repeatability
    :return: series (array of floats) - arrays containing the inpulses
    """
    # Initialize random number generator
    rdom = np.random.RandomState(seed)

    #Initialize array of random numbers equal to the length of given time steps plus 50
    inpulse_indices = rdom.randint(len(time), size=num_inpulses)

    #set first 50 elements to a constant
    series = np.zeros(len(time))

    # Insert random Inpulses
    for index in inpulse_indices:
        series[index] += rdom.rand()*amplitude

    return series


if __name__ == '__main__':
    #generate time steps. Assume 1 per day for one year (365)
    time = np.arange(365)

    #define the slope
    slope = 0.1

    #generate measurements with the defined slope
    series = trend(time, slope)

    # plot the results
    plt.plot(time, series)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(fontsize=14, labels='slope')
    plt.show()

    # Generate time steps
    time = np.arange(4 * 365 + 1)

    # Define the parameters of the seasonal data
    period = 365
    amplitude = 40

    #Generate the seasonal data
    series = seanonality(time, period=period, amplitude=amplitude)

    #Plot the results
    #plot_series(time, series)
    plt.plot(time, series)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(fontsize=14, labels='slope')
    plt.show()

    # define seasonal parameters
    slope = 0.05
    period = 365
    amplitude = 40

    #Generate the data
    series = trend(time, slope)+seanonality(time, period=period, amplitude=amplitude)
    plt.plot(time, series)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(fontsize=14, labels='slope')
    plt.show()

    # NOISE
    #define noise level
    noise_level = 5

    #Generate noisy signal
    noise_signal = noise(time, noise_level=noise_level, seed=42)

    plt.plot(time, noise_signal)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(fontsize=14, labels='slope')
    plt.show()

    # SERIES AND NOISE SIGNAL

    #Add the noise to the time series
    series += noise_signal
    plt.plot(time, series)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(fontsize=14, labels='slope')
    plt.show()

    # AUTOCORRELATION

    # Use time steps from previous section
    series = autocorrelation(time, amplitude=10, seed=42)

    #plot the first 200 elements to see the pattern more clearly
    plt.plot(time[:200], series[:200])
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(fontsize=14, labels='slope')
    plt.show()

    #INPULSES
    inpulses_signal = inpulse(time, num_inpulses=10, seed=42)
    plt.plot(time, inpulses_signal)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(fontsize=14, labels='slope')
    plt.show()



