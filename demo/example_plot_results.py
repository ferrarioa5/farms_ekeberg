
import numpy as np
import matplotlib.pyplot as plt
import os
from farms_core.experiment.data import ExperimentData

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "output", "simulation.hdf5")
exp_data = ExperimentData.from_file(file_path)

animat_list_data = exp_data.animats

timestep = exp_data.timestep
times    = exp_data.times

xlim = [25,30]
figure_size = (15, 5)

fig, axes = plt.subplots(1, len(animat_list_data), figsize=figure_size)

for animat_i, animat in enumerate(animat_list_data):

    joint_pos   = np.array(animat.sensors.joints.positions_all())
    joint_names = animat.sensors.joints.names

    ax = axes[animat_i] if len(animat_list_data) > 1 else axes
    ax.plot(times, joint_pos)
    ax.set_title(f'Animat {animat_i} Joint Positions')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Joint Position [rad]')
    ax.legend(joint_names)
    ax.grid(True)
    ax.set_xlim(xlim)
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "joint_positions.png"))


fig2, axes2 = plt.subplots(1, len(animat_list_data), figsize=figure_size)

for animat_i, animat in enumerate(animat_list_data):

    xfrc_x = np.array(animat.sensors.xfrc.array[:,:,0])
    link_names = animat.sensors.links.names

    ax2 = axes2[animat_i] if len(animat_list_data) > 1 else axes2
    ax2.plot(times, xfrc_x)
    ax2.set_title(f'Animat {animat_i} External Forces')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Force [N]')
    ax2.legend(link_names)
    ax2.grid(True)
    ax2.set_xlim(xlim)
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "external_forces.png"))
