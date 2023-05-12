import matplotlib.pyplot as plt 
import numpy as np

def uncertainty_probabilities(state_current,state_estimate):
    beta = 1
    Q_function = - np.linalg.norm(state_estimate - state_current) / np.size(state_current) * 5

    P = np.exp(Q_function * beta)
    P = 1.0 - P
    return P

#高斯核函数
def gaussian_kernel(x1, x2, l=0.5, sigma_f=0.2):
    m, n = x1.shape[0], x2.shape[0] 
    dist_matrix = np.zeros((m, n), dtype=float) 
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
    return sigma_f ** 2 * np.exp(- 0.5 / l ** 2 * dist_matrix)

def getRobotPosition(t):
    Y = 0
    if t > 0 and t <2* np.pi:
        Y = np.sin(t)*2 
    elif t > 2*np.pi and t < 3 * np.pi:
        Y =  (t - 2*np.pi) * 2
    elif t > 3 * np.pi and t < 4 * np.pi:
        Y =  -(t - 4*np.pi) * 2
    elif t > 4 * np.pi and t < 6 * np.pi:
        Y = np.sin(t)*3
    elif t > 6 * np.pi and t < 8 * np.pi:
        Y = -np.sin(t)*4

    return Y

#根据观察点X，修正生成高斯过程新的均值和协方差 
def update(X, Y, X_star):
    X = np.asarray(X)
    X_star = np.asarray(X_star)
    K_YY = gaussian_kernel(X, X) # K(X,X)
    K_Yf = gaussian_kernel(X, X_star) # K(X, X*)
    K_fY = K_Yf.T # K(X*, X) 协方差矩阵是对称的，因此分块互为转置
    K_YY_inv = np.linalg.inv(K_YY + 1e-8 * np.eye(len(X))) # (N, N)
    mu_star = K_fY.dot(K_YY_inv).dot(Y)
    return mu_star

f, ax = plt.subplots(2, 1, sharex=True,sharey=True) 

T = 24  #total time 
dt = 0.1
time = np.arange(0, T, dt).reshape(-1, 1)
Y_ground_truth = np.arange(0, 24, 0.1).reshape(-1, 1)
count = 0

for t in time:
    Y = getRobotPosition(t)
    Y_ground_truth[count] = Y
    count = count + 1

# originla robot path
ax[0].plot(time, Y_ground_truth, label="ground truth")
ax[0].legend()

plt.show()

ax[1].plot(time, Y_ground_truth, label="ground truth")
print(np.size(time))

# control number
N = np.size(time)
Y_estimate = np.zeros(N)
Y_guess = np.zeros(N)
P_uncertainty = np.zeros(N)
uncertainty_estimate = np.zeros(N)
i = 0
estimate_data_length = 10
gp_data_length = 2 * estimate_data_length
gp_buffer = np.zeros(gp_data_length)
X = np.arange(0,1,0.1).reshape(-1, 1)
X_star = np.arange(0,2,0.1).reshape(-1, 1)
count = 0

fig1,ax1 =plt.subplots(2,1)
ims = []
for t in time:
    # read robot position
    Y = getRobotPosition(t)

    # update gp buffer
    if count < 10:
        gp_buffer[count] = Y
    else:
        for i in np.arange(0,9):
            gp_buffer[i] = gp_buffer[i+1]
        gp_buffer[9] = Y

    if  count > estimate_data_length and count<= N - estimate_data_length:
        Y_ground_truth[count] = Y

        #estimate output at next step
        mu_star = update(X, gp_buffer[0:10], X_star)
        Y_star = mu_star.ravel()
        
        Y_estimate[count+1] = Y_star[10]
        Y_estimate_list = Y_estimate[count-9:count+1]

        P_uncertainty[count+1] = uncertainty_probabilities(gp_buffer[0:10],Y_estimate_list)
        
        ax1[0].plot(time, Y_ground_truth, label="ground truth",c="black")
        ax1[0].plot(time[2 * estimate_data_length+2:count], Y_estimate[2 * estimate_data_length+2:count],label="prediction",c="red",linewidth=3)
        ax1[0].axvspan(time[count-estimate_data_length],time[count], facecolor='0.8')
        ax1[0].axvline(time[count-estimate_data_length],c="green")
        ax1[0].axvline(time[count],c="green")
        ax1[0].text(12,6,'Time={}s'.format(t))
        ax1[0].set_title("Prediction Process with Gaussian Process")

        # ax1[0].xlabel("Time (s)")
        ax1[0].set_ylabel("Position (m)")
        ax1[0].grid()
        ax1[0].legend()

        ax1[1].plot(time[2*estimate_data_length+2:count], P_uncertainty[2*estimate_data_length+2:count], linewidth=2.0,label="probabilities",c="blue")
        ax1[1].set_ylim([0, 0.5])
        ax1[1].set_xlim([0, 25])
        ax1[1].text(17,6,'Time={}s'.format(t))
        ax1[1].legend()
        ax1[1].grid()
        # ax1[1].set_title("Prediction Process with Gaussian Process")

        ax1[1].set_xlabel("Time (s)")
        ax1[1].set_ylabel("Uncertainty (%)")
        plt.pause(1e-2)
        ax1[0].cla()
        ax1[1].cla()

    else:
        Y_estimate[count] = 0
        
    count = count + 1

plt.show()