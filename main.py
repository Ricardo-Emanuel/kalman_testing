import time
import numpy as np
import matplotlib.pyplot as plt
from kalmanfilter import KalmanFilter

def main():

    # X-axix
    dt = 0.01
    t = np.arange(0, 100, dt)
    # Define a model track
    gps_x = 0.05*(t**2)
    u_x = 1
    std_acc = 0.25     # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
    std_meas = 1.2    # and standard deviation of the measurement is 1.2 (m)

    # create KalmanFilter object
    kf_x = KalmanFilter(dt, u_x, std_acc, std_meas)
    predictions_x = []
    measurements_x = []
    i = 0
    for x in gps_x:
        # Mesurement
        z = kf_x.H * x + np.random.normal(0, 50)

        measurements_x.append(z.item(0))
        predictions_x.append(kf_x.predict()[0])
        if i == 0 or  i % 5 == 0:
            kf_x.update(z.item(0))
        i += 1
        #time.sleep(0.008)

    # Y-axis
    dt = 0.01
    t = np.arange(0, 100, dt)
    # Define a model track
    gps_y = 0.05*(2*(t**2))
    u_y = 2
    std_acc = 0.25     # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
    std_meas = 1.2    # and standard deviation of the measurement is 1.2 (m)

    # create KalmanFilter object
    kf_y = KalmanFilter(dt, u_y, std_acc, std_meas)
    predictions_y = []
    measurements_y = []
    i = 0
    for x in gps_y:
        # Mesurement
        z = kf_y.H * x + np.random.normal(0, 50)

        measurements_y.append(z.item(0))
        predictions_y.append(kf_y.predict()[0])
        if i == 0 or  i % 5 == 0:
            kf_y.update(z.item(0))
        i += 1
        #time.sleep(0.008)

    print(time.process_time())
    fig = plt.figure()
    fig.suptitle('Example of Kalman filter for tracking a moving object in 2-D', fontsize=20)
    plt.plot(t, measurements_y, label='Measurements', color='b',linewidth=0.5)
    plt.plot(t, np.array(gps_y), label='Real Track', color='y', linewidth=1.5)
    plt.plot(t, np.squeeze(predictions_y), label='Kalman Filter Prediction', color='r', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()