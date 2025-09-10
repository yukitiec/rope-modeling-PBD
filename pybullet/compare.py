import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import shutil



def load_csv(file_path):
    df = pd.read_csv(file_path, index_col=False, header=None)
    data = df.values
    header = data[0, :]
    data = data[1:, :]

    # Convert to numeric, handling any non-numeric values
    try:
        data = pd.to_numeric(data.flatten(), errors='coerce').reshape(data.shape)
        # Replace any NaN values with 0
        data = np.where(np.isnan(data), 0, data)
    except Exception as e:
        print(f"Warning: Error converting data to numeric: {e}")
        # Try to convert column by column
        for i in range(data.shape[1]):
            data[:, i] = pd.to_numeric(data[:, i], errors='coerce')
        data = np.where(np.isnan(data), 0, data)

    return data, header


if __name__ == "__main__":
    cwd = Path(os.getcwd())
    save_dir = cwd/"result"
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir_analytic = Path(r"C:\Users\kawaw\cpp\rope-modeling\rope-modeling")
    data_dir_pybullet = Path(r"C:\Users\kawaw\cpp\rope-modeling\pybullet\result\rope_state.csv")
    # Find all CSV files in data_dir and get their names without the '.csv' extension
    csv_files_analytic = [f for f in os.listdir(data_dir_analytic) if f.endswith('.csv')]
    csv_names_analytic = [os.path.splitext(name_csv)[0] for name_csv in csv_files_analytic]
    print("CSV files found:", csv_names_analytic)

    data_pybullet = pd.read_csv(data_dir_pybullet)
    data_pybullet = data_pybullet.values
    data_pybullet = data_pybullet.reshape(-1, data_pybullet.shape[1]//3, 3)#(N_step, N_segment, 3)
    print(f"data_pybullet.shape: {data_pybullet.shape}")
    data_dict = dict()
    data_dict_initial = dict()
    data_dict_final = dict()

    for name in csv_names_analytic:
        print(f"Model: {name}")
        file_path = data_dir_analytic/f"{name}.csv"
        data, header = load_csv(file_path)

        # Ensure data is numeric
        data = data.astype(np.float64)

        n_segment = (data.shape[1]-1)//(2*3)
        time_list = data[:, 0]
        pos_list = data[:, 1:1+n_segment*3]
        vel_list = data[:, 1+n_segment*3:]
        pos_seg_list = pos_list.reshape(-1, n_segment, 3)  # (N_step, N_segment, 3)
        vel_seg_list = vel_list.reshape(-1, n_segment, 3)  # (N_step, N_segment, 3)

        # Debug: Check data types and values
        data_dict[name] = np.concatenate([pos_seg_list[:,int(n_segment/2),:], vel_seg_list[:,int(n_segment/2),:]], axis=1) #(N_step, 6)
        data_dict_initial[name] = np.concatenate([pos_seg_list[:,0,:], vel_seg_list[:,0,:]], axis=1) #(N_step, 6)
        data_dict_final[name] = np.concatenate([pos_seg_list[:,-1,:], vel_seg_list[:,-1,:]], axis=1) #(N_step, 6)

    #Plot data_dir
    fig, ax = plt.subplots(2,3,figsize=(21,15)) #(2,3) for position and velocity
    for k,model in enumerate(data_dict.keys()):
        pos = data_dict[model][:,:3]
        vel = data_dict[model][:,3:]
        for i in range(3):
            ax[0,i].plot(pos[:,i], label=model)
            if k == 0:
                ax[0,i].plot(data_pybullet[:,int(data_pybullet.shape[1]/2),i], label="pybullet",linewidth=5,color="k")#middle
            ax[1,i].plot(vel[:,i], label=model)
            ax[0,i].set_xlabel("Time [s]")
            ax[0,i].set_ylabel("Position [m]")
            ax[1,i].set_xlabel("Time [s]")
            ax[1,i].set_ylabel("Velocity [m/s]")
            ax[0,i].set_title(f"segment {i}")
            ax[1,i].set_title(f"segment {i}")
    ax[0,0].legend()
    ax[1,0].legend()
    plt.savefig(save_dir/"pos_vel_middle.png")

    fig2, ax2 = plt.subplots(2,3,figsize=(21,15)) #(2,3) for position and velocity
    for k,model in enumerate(data_dict.keys()):
        pos = data_dict_initial[model][:,:3]
        vel = data_dict_initial[model][:,3:]
        for i in range(3):
            ax2[0,i].plot(pos[:,i], label=model)
            ax2[1,i].plot(vel[:,i], label=model)
            if k == 0:
                ax2[0,i].plot(data_pybullet[:,0,i], label="pybullet",linewidth=5,color="k")#initial
            ax2[0,i].set_xlabel("Time [s]")
            ax2[0,i].set_ylabel("Position [m]")
            ax2[1,i].set_xlabel("Time [s]")
            ax2[1,i].set_ylabel("Velocity [m/s]")
            ax2[0,i].set_title(f"segment {i}")
            ax2[1,i].set_title(f"segment {i}")
    ax2[0,0].legend()
    ax2[1,0].legend()
    plt.savefig(save_dir/"pos_vel_initial.png")

    fig3, ax3 = plt.subplots(2,3,figsize=(21,15)) #(2,3) for position and velocity
    for k,model in enumerate(data_dict.keys()):
        pos = data_dict_final[model][:,:3]
        vel = data_dict_final[model][:,3:]
        for i in range(3):
            ax3[0,i].plot(pos[:,i], label=model)
            ax3[1,i].plot(vel[:,i], label=model)
            if k == 0:
                ax3[0,i].plot(data_pybullet[:,-1,i], label="pybullet",linewidth=5,color="k")#final
            ax3[0,i].set_xlabel("Time [s]")
            ax3[0,i].set_ylabel("Position [m]")
            ax3[1,i].set_xlabel("Time [s]")
            ax3[1,i].set_ylabel("Velocity [m/s]")
            ax3[0,i].set_title(f"segment {i}")
            ax3[1,i].set_title(f"segment {i}")
    ax3[0,0].legend()
    ax3[1,0].legend()
    plt.savefig(save_dir/"pos_vel_final.png")

    # Optionally, show the animation in a window
    # plt.show()
