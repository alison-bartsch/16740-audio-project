import numpy as np
import math
import matplotlib.pyplot as plt
import random
from robot import Robot
import time
r = Robot()

minval=65
maxval=1024


def explore(points):
    shifts = r.collect_baselines(200)
    input("\nTURN ON METRONOME! HIT ENTER TO CONTINUE.\n")

    z = -45
    data = []

    for p_t in points:
        x = p_t[0]
        y = p_t[1]
        r.move_to(x,y)
        time.sleep(1)
        pv = r.peak_value(100, 1.0, shifts)
        print(pv)
        data.append(((x,y), pv))

    return data


# s:    sound source location
# a_t:  control inputs
# p_t:  end effector position
# d_t:  optimal moving direction (normalize(s-p_t))



# state the sound source location
# s = (0, 30)


# ----------------------------------------------------------
# -------------- Determine Sampling Locations --------------
# ----------------------------------------------------------

# want to move to 3 different locations - have a random option and a fixed triangle option
point_locations = []
locations_random = True

if locations_random:
    # randomly sample an angle
    theta = math.radians(random.randint(0,360))
    r = 15
    side_length = 2*r*math.cos(math.radians(30))

    x1 = math.cos(theta)*r
    y1 = math.sin(theta)*r


    vec_mag = math.sqrt(x1**2 + y1**2)
    dir_ctr = (-x1/vec_mag, -y1/vec_mag)


    dir2 = (math.cos(math.radians(30)*dir_ctr[0]) - math.sin(math.radians(30)*dir_ctr[1]), 
        math.sin(math.radians(30)*dir_ctr[0]) + math.cos(math.radians(30)*dir_ctr[1]))

    mag2 = math.sqrt(dir2[0]**2 + dir2[1]**2)
    dir2 = (dir2[0]/mag2, dir2[1]/mag2)


    dir3 = (math.cos(math.radians(-30)*dir_ctr[0]) - math.sin(math.radians(-30)*dir_ctr[1]), 
        math.sin(math.radians(-30)*dir_ctr[0]) + math.cos(math.radians(-30)*dir_ctr[1]))

    mag3 = math.sqrt(dir3[0]**2 + dir3[1]**2)
    dir3= (dir3[0]/mag3, dir3[1]/mag3)

    x2 = x1 - dir2[0]*side_length
    y2 = y1 - dir2[1]*side_length

    x3 = x1 - dir3[0]*side_length
    y3 = y1 - dir3[1]*side_length


    # round all the points to the nearest mutliple of 5
    x1 = 5 * round(x1/5)
    y1 = 5 * round(y1/5)
    x2 = 5 * round(x2/5)
    y2 = 5 * round(y2/5)
    x3 = 5 * round(x3/5)
    y3 = 5 * round(y3/5)

    point_locations = [(x1, y1), (x2, y2), (x3, y3)]

else:
    point_locations = [(-15, -15), (0, 20), (15, -5)]


# ----------------------------------------------------------
# --------- Artificial Data: Testing Without Robot ---------
# ----------------------------------------------------------

# data = np.zeros((3,2,2))

# # Case: sound source is from the left
# s = (0, 30)

# data[0] = np.array([[-15, -15], [175, 40]])
# data[1] = np.array([[0, 20], [200, 50]])
# data[2] = np.array([[15, -5], [100, 70]])


# # Case: sound source is from the right
# s = (0, -30)

# data[0] = np.array([[-15, -15], [75, 200]])
# data[1] = np.array([[0, 20], [80, 100]])
# data[2] = np.array([[15, -5], [50, 150]])


# # Case: sound source is from the front
# s = (30, 0)

# data[0] = np.array([[-15, -15], [55, 70]])
# data[1] = np.array([[0, 20], [90, 80]])
# data[2] = np.array([[15, -5], [125, 160]])


# # Case: sound source is from the back
# s = (-30, 0)

# data[0] = np.array([[-15, -15], [145, 120]])
# data[1] = np.array([[0, 20], [90, 70]])
# data[2] = np.array([[15, -5], [60, 80]])




# collect data for each p_t 
data = explore(point_locations)
LR_difference = []
LR_avg = []
x_loc = []

# estimate sound source location
s_prime = (0,0)

for i in range(len(data)):
    L = data[i][1][0]
    R = data[i][1][1]
    LR_difference.append(L-R)
    LR_avg.append(np.mean([L, R]))
    x_loc.append(data[i][0][0])

diff_avg = np.mean(LR_difference)
print("LR Difference: ", diff_avg, "\n")
max_idx = np.argmax(LR_avg)

other_pts = np.delete(x_loc, max_idx)
x_dir = x_loc[max_idx] - other_pts[0] + x_loc[max_idx] - other_pts[1]
print("X Direction: ", x_dir, "\n")

# look at if sound source is from left or right
if diff_avg > 25:
    s_prime = (0, 30)
    print("Left")
elif diff_avg < -25:
    s_prime = (0, -30)
    print("Right")

# look at if sound source is front or back
elif x_dir > 0:
    s_prime = (30, 0)
    print("Front")
else:
    s_prime = (-30, 0)
    print("Back")


print("\nGuess Sound Location: ", s_prime)
# print("Actual Sound Location: ", s)


# move to the guess location

r.move_to(s_prime[0],s_prime[1])

