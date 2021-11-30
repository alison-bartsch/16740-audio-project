import numpy as np
import matplotlib.pyplot as plt



# ----------------------------------------------------------
# -------------- Process the Data Collected ----------------
# ----------------------------------------------------------


# s:    sound source location
# a_t:  control inputs
# p_t:  end effector position
# d_t:  optimal moving direction (normalize(s-p_t))


# create dictionary of sound source locations
sound_loc = {"back": (30, 0),
            "front": (-30, 0),
            "left": (0, 30),
            "right": (0, -30)}

sources = ["back", "front", "left", "right"]



# iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
for location in sources:
    s = sound_loc[location]
    print(s)

    # iterate through the .npz files
    for i in range(10):
        file_path = "data/" + location + "_data_" + str(i) + ".npz"
        save_path = "data/processed_data/" + location + "_processed_" + str(i) + ".npz"
        npzfile = np.load(file_path)
        raw_data = npzfile["data"]

        data = np.zeros((len(raw_data),8))

        for j in range(len(raw_data)):

            # defining data in terms of defined variables
            p_t = (raw_data[j][0,0], raw_data[j][0,1])
            h = (raw_data[j][1,0], raw_data[j][1,1])
            d_t = (s[0] - raw_data[j][0,0], s[1] - raw_data[j][0,1])

            data[j, 0] = p_t[0]     # [p_t_x]
            data[j, 1] = p_t[1]     # [p_t_y]
            data[j, 2] = h[0]       # [h_left]
            data[j, 3] = h[1]       # [h_right]
            data[j, 4] = s[0]       # [s_x]
            data[j, 5] = s[1]       # [s_y]
            data[j, 6] = d_t[0]     # [d_t_x]
            data[j, 7] = d_t[1]     # [d_t_y]

        # normalize d_t
        norm_x = np.linalg.norm(data[:,6])
        norm_y = np.linalg.norm(data[:,7])
        

        print(data)

        assert False

        np.savez(save_path, data=data)







# ----------------------------------------------------------
# ------------- Testing a Heuristic Approach ---------------
# ----------------------------------------------------------


# 










# npzfile = np.load("data/back_data_0.npz")
# data = npzfile["data"]
# print(data)
# print(data.shape)



# # getting p_t
# xs = np.arange(-30,31,5)
# ys = np.arange(-30,31,5)
# z = -45

# length = 0

# for i,x in enumerate(xs):
#     # first_time=True
#     for j,y in enumerate(ys):
#         if x**2 + y**2 <= 900:
#             position = (x,y)
#             length = length + 1

# print(length)




# getting a_t
# this is the move_to function from the arduino code (it's unclear, because this function only takes the desired p_t)





# calculating d_t
# get s-p_t for all rows of column
# normalize the column






# def merge_left_right(data_file):
#     # input the data_file & print to see the format as a test

#     # take | left - right | for each location

#     # combine into new data file and save to new folder




def visualize_data(data_file, sep):
    # modify this so that it plots from only one set of data (not left and right)

    npzfile = np.load(data_file)
    data = npzfile["data"]

    x,y = np.meshgrid(np.arange(-30-sep//2,31+sep//2,sep), np.arange(-30-sep//2,31+sep//2,sep))
    l = np.ones((60//sep+1,60//sep+1)) * -10
    r = np.ones((60//sep+1,60//sep+1)) * -10
    shift = 30//sep
    for i in range(len(data)):
        xx = int(data[i][0][0])
        yy = int(data[i][0][1])
        l[xx//sep + shift][yy//sep + shift] = data[i][1][0]
        r[xx//sep + shift][yy//sep + shift] = data[i][1][1]

    # print("x")
    # print(x)
    # print("y")
    # print(y)
    # print("l")
    # print(l)
    fig, axs = plt.subplots(1,2)
    c = axs[0].pcolormesh(x, y, l, cmap='RdBu', vmin=40, vmax=250)
    axs[0].set_title('Left')
    axs[0].axis([x.min(), x.max(), y.min(), y.max()])
    axs[0].set_aspect('equal')
    fig.colorbar(c, ax=axs[0])

    c = axs[1].pcolormesh(x, y, r, cmap='RdBu', vmin=40, vmax=250)
    axs[1].set_title('Right')
    axs[1].axis([x.min(), x.max(), y.min(), y.max()])
    axs[1].set_aspect('equal')
    fig.colorbar(c, ax=axs[1])
    fig.set_size_inches(12, 5)

    
    # save the plot
    plot_name = data_file.split('.')[0].split('/')
    plot_path = "figures/" + plot_name[1] + ".png"
    plt.savefig(plot_path)

    plt.show()

