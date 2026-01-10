
from farms_core.model.control import AnimatController
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.sensors.sensor_convention import sc
from farms_core.model.control import ControlType

from dm_control.rl.control import Task
from dm_control.mjcf.physics import Physics

import numpy as np
import importlib

class Dict2Class(object):
    '''
    Turns a dictionary into a class
    '''
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
    def update(self, newdict={}, **kwargs):
        self.__dict__.update(newdict, **kwargs)

class EkebergMuscleController(AnimatController):

    def __init__(self, animat_data, animat_options, experiment_options, config, animat_i, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.animat_data        = animat_data
        self.animat_options     = animat_options
        self.experiment_options = experiment_options
        self.config             = Dict2Class(config)
        self.animat_i           = animat_i
        self.n_joints           = len(self.animat_options.control.motors)
        self.n_iterations       = experiment_options.simulation.runtime.n_iterations

        # load class from string
        module_name, class_name = self.config.load_controller.rsplit('.', 1)
        module = importlib.import_module(module_name)
        controller_class = getattr(module, class_name)

        self.nn = controller_class(
            animat_data,
            animat_options,
            experiment_options,
            self.n_joints,
            self.n_iterations,
            self.config
        )

        # Define muscle parameters for each joint - could be loaded from a config instead (i.e. the animat_options)

        muscles_pars = self.config.muscle_pars

        if isinstance(muscles_pars, dict):
            # muscles_pars is already a dictionary, use it as is
            pass
        elif isinstance(muscles_pars, str):
            # muscles_pars is a path to a CSV file, load it

            if not muscles_pars.endswith('.csv'):
                raise ValueError(f"muscle_pars file path must end with '.csv', got '{muscles_pars}'")
            
            data = np.genfromtxt(muscles_pars, delimiter=',', names=True, dtype=None, encoding='utf-8')
            muscles_pars = {
                row[0]: {field: row[i+1] for i, field in enumerate(data.dtype.names[1:])}
                for row in data
            } 
            self.config.muscle_pars = muscles_pars

        else:
            raise ValueError(f"muscle_pars must be either a dict or a CSV file path, got {type(muscles_pars)}")

        control_names = [motor.joint_name for motor in self.animat_options.control.motors] # list orderded according to the joint motor order
        self.muscle_pars_dict = {
            "alpha": np.array([muscles_pars[joint]["alpha"] for joint in control_names]),
            "beta": np.array([muscles_pars[joint]["beta"] for joint in control_names]),
            "gamma": np.array([muscles_pars[joint]["gamma"] for joint in control_names]),
            "delta": np.array([muscles_pars[joint]["delta"] for joint in control_names]),
        }

        self.offsets = np.array([muscles_pars[joint]["offset"] for joint in control_names])

        self.muscle_method=config.pop("method", "implicit")
        if self.muscle_method == "explicit":
            self.step_muscles = self.step_muscles_explicit
        elif self.muscle_method == "implicit":
            self.step_muscles = self.step_muscles_implicit

        self.torque = np.zeros(self.n_joints)

        self.log_torques = config.get("log_torques", True)


    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):

        """From options"""
        joints_names = [
            joint.name
            for joint in animat_options.morphology.joints
        ]
        joints_control_types: dict[str, list[ControlType]] = {
            motor.joint_name: ControlType.from_string_list(motor.control_types)
            for motor in animat_options.control.motors
        }
        return cls(
            joints_names=AnimatController.joints_from_control_types(
                joints_names=joints_names,
                joints_control_types=joints_control_types,
            ),
            muscles_names=[],
            max_torques=AnimatController.max_torques_from_control_types(
                joints_names=joints_names,
                max_torques={
                    motor.joint_name: motor.limits_torque[1]
                    for motor in animat_options.control.motors
                },
                joints_control_types=joints_control_types,
            ),
            animat_data = animat_data,
            animat_options = animat_options,
            experiment_options = experiment_options,
            config = config,
            animat_i = animat_i,
        )


    def step_muscles_explicit(self, M_diff, M_sum, iteration, time, timestep):

        """
        integrate the muscles explicitly
        """

        joints_pos = np.array(self.animat_data.sensors.joints.positions(iteration))
        joint_vel  = np.array(self.animat_data.sensors.joints.velocities(iteration))

        m_delta_phi = (self.offsets - joints_pos)

        active_torque          = self.muscle_pars_dict["alpha"] * M_diff
        stiffness_intermediate = self.muscle_pars_dict["beta"] * m_delta_phi
        active_stiffness       = M_sum*stiffness_intermediate
        passive_stiffness      = self.muscle_pars_dict["gamma"] * stiffness_intermediate
        damping                = -self.muscle_pars_dict["delta"] * joint_vel

        self.torque = active_torque + active_stiffness + passive_stiffness + damping

        self.animat_data.sensors.joints.array[iteration,:,sc.joint_cmd_torque] = self.torque

    def step_muscles_implicit(self, M_diff, M_sum, iteration, time, timestep):

        """
        integrate the muscles semi-implicitly
        i.e. all the stiffness and damping terms are treated implicitly, except for the active torque
        """
        self.kp     = self.muscle_pars_dict["beta"] * (M_sum + self.muscle_pars_dict["gamma"])  # conversion from ekeberg to pd controller
        self.kd     = self.muscle_pars_dict["delta"]
        self.torque = self.muscle_pars_dict["alpha"] * M_diff

        if self.log_torques:
            self.animat_data.sensors.joints.array[iteration,:,sc.joint_cmd_torque] = self.torque


    def before_step(self, task: Task, action, physics: Physics):
        time = physics.time()
        timestep = physics.timestep()
        iteration = task.iteration % task.buffer_size
        Mdiff, Msum = self.nn.step(iteration=iteration, time=time, timestep=timestep)
        self.step_muscles(Mdiff, Msum, iteration=iteration, time=time, timestep=timestep)


    def springrefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Spring references"""
        output = {}
        if self.muscle_method == "implicit":
            output={
                joint: self.offsets[idx]
                for idx, joint in enumerate(self.joints_names[ControlType.TORQUE])
            }
        return output

    def springcoefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Spring coefficients"""
        output = {}
        if self.muscle_method == "implicit":
            output={
                joint: self.kp[idx]
                for idx, joint in enumerate(self.joints_names[ControlType.TORQUE])
            }
        return output

    def dampingcoefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Damping coefficients"""
        output = {}
        if self.muscle_method == "implicit":
            output={
                joint: self.kd[idx]
                for idx, joint in enumerate(self.joints_names[ControlType.TORQUE])
            }
        return output

    def torques(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Torques"""
        return {
            joint: self.torque[idx]
            for idx, joint in enumerate(self.joints_names[ControlType.TORQUE])
        }