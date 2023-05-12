import matplotlib.pyplot as plt 
import numpy as np

class GP_Prediction:
    def __init__(self, estimate_data_length):
        self.estimate_data_length = estimate_data_length
        self.count = 0
        self.gp_x_buffer = np.zeros(2 * estimate_data_length)
        self.gp_y_buffer = np.zeros(2 * estimate_data_length)
        self.X_estimate_list = np.zeros(estimate_data_length)
        self.Y_estimate_list = np.zeros(estimate_data_length)

    def uncertainty_probabilities(self, state_current,state_estimate):
        beta = 1
        Q_function = - np.linalg.norm(state_estimate - state_current) / np.size(state_current) * 10
        P = np.exp(Q_function * beta)
        P = 1.0 - P
        return P

    def gaussian_kernel(self,x1, x2, l=0.5, sigma_f=0.2):
        m, n = x1.shape[0], x2.shape[0] 
        dist_matrix = np.zeros((m, n), dtype=float) 
        for i in range(m):
            for j in range(n):
                dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
        return sigma_f ** 2 * np.exp(- 0.5 / l ** 2 * dist_matrix)

    def update(self, X, Y, X_star):
        X = np.asarray(X)
        X_star = np.asarray(X_star)
        K_YY = self.gaussian_kernel(X, X) # K(X,X)
        K_Yf = self.gaussian_kernel(X, X_star) # K(X, X*)
        K_fY = K_Yf.T 
        K_YY_inv = np.linalg.inv(K_YY + 1e-8 * np.eye(len(X))) # (N, N)
        mu_star = K_fY.dot(K_YY_inv).dot(Y)
        return mu_star

    def inference(self, X_cur_pos, Y_cur_pos):
        X = np.arange(0,1,0.1).reshape(-1, 1)
        X_star = np.arange(0,2,0.1).reshape(-1, 1)
        X_est_pos = 0
        Y_est_pos = 0
        uncertainty = 0
        # update gp buffer
        if self.count < self.estimate_data_length:
            self.gp_x_buffer[self.count] = X_cur_pos
            self.gp_y_buffer[self.count] = Y_cur_pos
            X_est_pos = X_cur_pos
            Y_est_pos = Y_cur_pos 
            uncertainty = 1
        else:
            for i in np.arange(0,self.estimate_data_length-1):
                self.gp_x_buffer[i] = self.gp_x_buffer[i+1]
                self.gp_y_buffer[i] = self.gp_y_buffer[i+1]
            self.gp_x_buffer[self.estimate_data_length-1] = X_cur_pos
            self.gp_y_buffer[self.estimate_data_length-1] = Y_cur_pos

        #estimate output at next step
        if  self.count > self.estimate_data_length:
            x_mu = self.update(X, self.gp_x_buffer[0:self.estimate_data_length], X_star)
            y_mu = self.update(X, self.gp_y_buffer[0:self.estimate_data_length], X_star)

            X_pre_total = x_mu.ravel()
            Y_pre_total = y_mu.ravel()
            # print(X_pre_total)
            if self.count < 2 * self.estimate_data_length + 1:
                self.X_estimate_list[self.count-self.estimate_data_length-1] = X_pre_total[self.estimate_data_length]
                self.Y_estimate_list[self.count-self.estimate_data_length-1] = Y_pre_total[self.estimate_data_length]

                uncertainty = 1
            else:
                for i in np.arange(0,9):
                    self.X_estimate_list[i] = self.X_estimate_list[i+1]
                    self.Y_estimate_list[i] = self.Y_estimate_list[i+1]
                self.X_estimate_list[self.estimate_data_length-1] = X_pre_total[self.estimate_data_length]
                self.Y_estimate_list[self.estimate_data_length-1] = Y_pre_total[self.estimate_data_length]

                P_X_uncertainty = self.uncertainty_probabilities(self.gp_x_buffer[0:self.estimate_data_length], self.X_estimate_list)
                P_Y_uncertainty = self.uncertainty_probabilities(self.gp_y_buffer[0:self.estimate_data_length], self.Y_estimate_list)
                uncertainty = P_X_uncertainty * P_Y_uncertainty

            X_est_pos = X_pre_total[self.estimate_data_length]
            Y_est_pos = Y_pre_total[self.estimate_data_length] 

        self.count = self.count + 1

        return X_est_pos,Y_est_pos,uncertainty





# ************************ example for GP_Prediction **************************************
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

T = 18  #total time (sec)
dt = 0.1
time = np.arange(0, T, dt).reshape(-1, 1)
X_ground_truth = np.arange(0, T, 0.1).reshape(-1, 1)
Y_ground_truth = np.arange(0, T, 0.1).reshape(-1, 1)
count = 0

for t in time:
    X = getRobotPosition(t)
    Y = getRobotPosition(t+1)
    X_ground_truth[count] = X
    Y_ground_truth[count] = Y
    count = count + 1

estimate_data_length = 10

count = 0

f, ax = plt.subplots() 
N = np.size(time)
X_estimate = np.zeros(N)
Y_estimate = np.zeros(N)
uncertainty_estimate = np.zeros(N)

GP = GP_Prediction(estimate_data_length)
for t in time:
    if  count<= N - estimate_data_length:
        # read robot position
        X_current = getRobotPosition(t)
        Y_current = getRobotPosition(t+1)
        X_est_pos, Y_est_pos, uncertainty = GP.inference(X_current,Y_current)
        X_estimate[count+1] = X_est_pos
        Y_estimate[count+1] = Y_est_pos
        uncertainty_estimate[count+1] = uncertainty

    count = count + 1

ax.plot(X_estimate[2 * estimate_data_length+3:N-estimate_data_length], Y_estimate[2 * estimate_data_length+3:N-estimate_data_length],label="prediction",c="red",linewidth=3)
ax.plot(X_ground_truth, Y_ground_truth,label="ground truth",c="black",linewidth=1)
ax.grid()
ax.legend()
plt.show()

f, ax1 = plt.subplots() 
ax1.plot(time[2 * estimate_data_length+3:N], uncertainty_estimate[2 * estimate_data_length+3:N],label="uncertainty",c="black",linewidth=1)
ax1.grid()
ax1.legend()
plt.show()
