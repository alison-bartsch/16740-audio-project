import numpy as np
import matplotlib.pyplot as plt
import random

ser = serial.Serial('/dev/cu.usbserial-130')
time.sleep(2)
print(ser.name)

ser.readline()
ser.readline()

minval=65
maxval=1024


# ----------------------------------------------------------
# ------------------ Necessary Functions -------------------
# ----------------------------------------------------------

def write_read(x):
    ser.write(bytes(x))
    time.sleep(0.1)
    data = ser.readline().decode("utf-8")
    return data


def collect_sound(steps=1):
    cmd = str.encode('read {} \n'.format(steps))
    data = write_read(cmd)
    data = data.split(";")[:-1]
    data = [list(map(int, s.split(','))) for s in data]
    data = np.array(data).transpose()
    return data[0], data[1]


def collect_baselines(steps):
    move_to(0,0,-45)
    l1, l2 = collect_sound(steps=steps)
    return np.array([np.mean(l1), np.mean(l2)])


def move_to(x,y,z):
    s = str.encode('goto {},{},{} \n'.format(x,y,z))
    print("move to command:", s)
    ret = write_read(s)
    time.sleep(1)


def peak_value(steps, percent, shifts):
    l1, l2 = collect_sound(steps)
    l1 = np.abs(l1 - shifts[0])
    l2 = np.abs(l2 - shifts[1])
    l1 = sorted(l1)
    l2 = sorted(l2)
    idx = int(percent * (len(l1)-1))
    return np.array([l1[idx], l2[idx]])


def explore(points):
    shifts = collect_baselines(200)
    input("\nTURN ON METRONOME! HIT ENTER TO CONTINUE.\n")

    z = -45
    data = []

    for p_t in points:
        x = p_t[0]
        y = p_t[1]
        move_to(x,y,z)
        time.sleep(1)
        pv = peak_value(100, 1.0, shifts)
        print(pv)
        data.append(((x,y), pv))

    return data


# s:    sound source location
# a_t:  control inputs
# p_t:  end effector position
# d_t:  optimal moving direction (normalize(s-p_t))



# state the sound source location
s = (0, 30)


# want to move to 3 different locations - have a random option and a fixed triangle option
point_locations = []
locations_random = False

if locations_random:
    point_locations = []
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

for i in range(data.shape[0]):
    L = data[i][1,0]
    R = data[i][1,1]
    LR_difference.append(L-R)
    LR_avg.append(np.mean([L, R]))
    x_loc.append(data[i][0,0])

diff_avg = np.mean(LR_difference)
print("LR Difference: ", diff_avg, "\n")
max_idx = np.argmax(LR_avg)

other_pts = np.delete(x_loc, max_idx)
x_dir = x_loc[max_idx] - other_pts[0] + x_loc[max_idx] - other_pts[1]
print("X Direction: ", x_dir, "\n")

# look at if sound source is from left or right
if diff_avg > 25:
    s_prime = (0, 30)
elif diff_avg < -25:
    s_prime = (0, -30)

# look at if sound source is front or back
elif x_dir > 0:
    s_prime = (30, 0)
else:
    s_prime = (-30, 0)


print("\nGuess Sound Location: ", s_prime)
print("Actual Sound Location: ", s)


# move to the guess location
z = -45
move_to(s_prime[0],s_prime[1],z)

