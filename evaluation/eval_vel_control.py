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
    root = cwd.parent
    data_dir = root/"rope-modeling"
    # Find all CSV files in data_dir and get their names without the '.csv' extension
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_names = [os.path.splitext(name_csv)[0] for name_csv in csv_files]
    print("CSV files found:", csv_names)
    # Create a directory to save the videos
    video_dir = root/"videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    data_dict = dict()
    data_dict_initial = dict()
    data_dict_final = dict()

    for name in csv_names:
        print(f"Model: {name}")
        file_path = data_dir/f"{name}.csv"
        data, header = load_csv(file_path)

        # Ensure data is numeric
        data = data.astype(np.float64)

        n_segment = (data.shape[1]-1)//(2*3)
        time_list = data[:, 0]
        pos_list = data[:, 1:1+n_segment*3]
        vel_list = data[:, 1+n_segment*3:]
        pos_seg_list = pos_list.reshape(-1, n_segment, 3)  # (N_step, N_segment, 3)
        vel_seg_list = vel_list.reshape(-1, n_segment, 3)  # (N_step, N_segment, 3)
        print(pos_seg_list.shape)
        print(vel_seg_list.shape)

        # Debug: Check data types and values
        print(f"pos_seg_list dtype: {pos_seg_list.dtype}")
        print(f"Sample pos values: {pos_seg_list[0, 0, :]}")
        print(f"All values finite: {np.all(np.isfinite(pos_seg_list))}")

        # Debug: Check data types and values
        data_dict[name] = np.concatenate([pos_seg_list[:,int(n_segment/2),:], vel_seg_list[:,int(n_segment/2),:]], axis=1) #(N_step, 6)
        data_dict_initial[name] = np.concatenate([pos_seg_list[:,0,:], vel_seg_list[:,0,:]], axis=1) #(N_step, 6)
        data_dict_final[name] = np.concatenate([pos_seg_list[:,-1,:], vel_seg_list[:,-1,:]], axis=1) #(N_step, 6)
        fig,ax = plt.subplots(2,3,figsize=(21,15))
        for i in range(n_segment//5):
            for j in range(3):
                ax[0,j].plot(time_list, pos_seg_list[:,5*i,j], label=f"segment {5*i}")
                ax[1,j].plot(time_list, vel_seg_list[:,5*i,j], label=f"segment {5*i}")
                ax[0,j].set_xlabel("Time [s]")
                ax[0,j].set_ylabel("Position [m]")
                ax[1,j].set_xlabel("Time [s]")
                ax[1,j].set_ylabel("Velocity [m/s]")
                ax[0,j].set_title(f"segment {5*i}")
                ax[1,j].set_title(f"segment {5*i}")
        ax[0,0].legend()
        ax[1,0].legend()
        #plt.show()
        plt.close(fig)

        # Parameters
        N_sample = pos_seg_list.shape[0]
        N_point = pos_seg_list.shape[1]
        dt = 0.02  # original sampling time
        fps = 25   # video frame rate (half speed)
        interval = int(1000 / fps)  # ms per frame

        fig3d = plt.figure(figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection='3d')

        # Set axis limits based on data
        pos_numeric = pos_seg_list.astype(np.float64)
        x_min, x_max = np.min(pos_numeric[:, :, 0]), np.max(pos_numeric[:, :, 0])
        y_min, y_max = np.min(pos_numeric[:, :, 1]), np.max(pos_numeric[:, :, 1])
        z_min, z_max = np.min(pos_numeric[:, :, 2]), np.max(pos_numeric[:, :, 2])
        ax3d.set_xlim(x_min, x_max)
        ax3d.set_ylim(y_min, y_max)
        ax3d.set_zlim(z_min, z_max)
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title('3D Rope Transition')
        # Do not show or pop up the figure
        # (do nothing here)


        # Create a temporary directory to save images
        temp_img_dir = Path(data_dir) / "temp_rope_frames"
        temp_img_dir.mkdir(parents=True, exist_ok=True)

        for frame in range(N_sample):
            fig_frame = plt.figure(figsize=(10, 8))
            ax_frame = fig_frame.add_subplot(111, projection='3d')

            # Extract positions for this frame
            x = pos_seg_list[frame, :, 0].astype(np.float64)
            y = pos_seg_list[frame, :, 1].astype(np.float64)
            z = pos_seg_list[frame, :, 2].astype(np.float64)

            # Ensure finite values
            x = np.where(~np.isfinite(x), 0, x)
            y = np.where(~np.isfinite(y), 0, y)
            z = np.where(~np.isfinite(z), 0, z)


            # Plot points
            ax_frame.scatter(x, y, z, color='b', s=40, label='Segments')

            # Connect neighborhood points with a line
            ax_frame.plot(x, y, z, color='r', lw=2, label='Rope')

            # Set axis limits and labels
            ax_frame.set_xlim(x_min, x_max)
            ax_frame.set_ylim(y_min, y_max)
            ax_frame.set_zlim(z_min, z_max)
            ax_frame.set_xlabel('X')
            ax_frame.set_ylabel('Y')
            ax_frame.set_zlabel('Z')
            ax_frame.set_title(f'3D Rope Transition (t={frame*dt:.2f}s)')

            # Optionally, add legend
            # ax_frame.legend()
            # Save the figure
            img_path = os.path.join(temp_img_dir, f"frame_{frame:05d}.png")
            plt.savefig(img_path)
            plt.close(fig_frame)

        # Make a video from the saved images using ffmpeg
        video_path = video_dir / f"{name}.mp4"
        # Compose ffmpeg command
        # -r {fps}: input frame rate
        # -i: input pattern
        # -vcodec libx264: use H.264 codec
        # -pix_fmt yuv420p: for compatibility
        ffmpeg_cmd = (
            f"ffmpeg -y -framerate {fps} -i \"{temp_img_dir}/frame_%05d.png\" "
            f"-vcodec libx264 -pix_fmt yuv420p \"{video_path}\""
        )
        print("Running ffmpeg to create video...")
        os.system(ffmpeg_cmd)

        print(f"Video saved to {video_path}")

        # Delete the temporary directory and all its contents
        shutil.rmtree(temp_img_dir)
        print(f"Temporary image directory {temp_img_dir} deleted.")

    #Plot data_dir
    fig, ax = plt.subplots(2,3,figsize=(21,15)) #(2,3) for position and velocity
    for model in data_dict.keys():
        pos = data_dict[model][:,:3]
        vel = data_dict[model][:,3:]
        for i in range(3):
            ax[0,i].plot(pos[:,i], label=model)
            ax[1,i].plot(vel[:,i], label=model)
            ax[0,i].set_xlabel("Time [s]")
            ax[0,i].set_ylabel("Position [m]")
            ax[1,i].set_xlabel("Time [s]")
            ax[1,i].set_ylabel("Velocity [m/s]")
            ax[0,i].set_title(f"segment {i}")
            ax[1,i].set_title(f"segment {i}")
    ax[0,0].legend()
    ax[1,0].legend()
    plt.savefig(video_dir/"pos_vel_middle.png")

    fig2, ax2 = plt.subplots(2,3,figsize=(21,15)) #(2,3) for position and velocity
    for model in data_dict.keys():
        pos = data_dict_initial[model][:,:3]
        vel = data_dict_initial[model][:,3:]
        for i in range(3):
            ax2[0,i].plot(pos[:,i], label=model)
            ax2[1,i].plot(vel[:,i], label=model)
            ax2[0,i].set_xlabel("Time [s]")
            ax2[0,i].set_ylabel("Position [m]")
            ax2[1,i].set_xlabel("Time [s]")
            ax2[1,i].set_ylabel("Velocity [m/s]")
            ax2[0,i].set_title(f"segment {i}")
            ax2[1,i].set_title(f"segment {i}")
    ax2[0,0].legend()
    ax2[1,0].legend()
    plt.savefig(video_dir/"pos_vel_initial.png")

    fig3, ax3 = plt.subplots(2,3,figsize=(21,15)) #(2,3) for position and velocity
    for model in data_dict.keys():
        pos = data_dict_final[model][:,:3]
        vel = data_dict_final[model][:,3:]
        for i in range(3):
            ax3[0,i].plot(pos[:,i], label=model)
            ax3[1,i].plot(vel[:,i], label=model)
            ax3[0,i].set_xlabel("Time [s]")
            ax3[0,i].set_ylabel("Position [m]")
            ax3[1,i].set_xlabel("Time [s]")
            ax3[1,i].set_ylabel("Velocity [m/s]")
            ax3[0,i].set_title(f"segment {i}")
            ax3[1,i].set_title(f"segment {i}")
    ax3[0,0].legend()
    ax3[1,0].legend()
    plt.savefig(video_dir/"pos_vel_final.png")

    # Optionally, show the animation in a window
    # plt.show()
