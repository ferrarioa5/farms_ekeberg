"""Network"""

from abc import ABC, abstractmethod
import numpy as np
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from scipy.integrate._ode import ode
from scipy import integrate

class NNController(ABC):
    """NN controller
    All controller classes using the Ekeberg muscle model need to be subclasses of NNController
    and implement the step function.
    """

    def __init__(self, animat_data, animat_options, experiment_options, n_joints, n_iterations):
        super().__init__()
        self.animat_data: AnimatData = animat_data
        self.animat_options = animat_options
        self.experiment_options = experiment_options
        self.n_joints = n_joints
        self.n_iterations = n_iterations

    @abstractmethod
    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            **kwargs,
    ):
        """Step function called at each simulation iteration
        It should return the muscle activations as two arrays:
        M_diff: np.ndarray of shape (n_joints,)
            Difference between left and right muscle activations
        M_sum: np.ndarray of shape (n_joints,)
            Sum of left and right muscle activations
        See examples in this file.
        """


class SalamandraRoboticaController(NNController):
    """Salamandra Robotica Controller Base Class"""

    def __init__(self, animat_data, animat_options, experiment_options, n_joints, n_iterations):
        super().__init__(animat_data, animat_options, experiment_options, n_joints, n_iterations)

        control_joint_names = [joint.joint_name for joint in self.animat_options.control.motors]
        body_joint_names    = [name for name in control_joint_names if "body" in name]
        leg_joint_names     = [name for name in control_joint_names if "leg" in name]

        self.n_body_joints  = len(body_joint_names)
        self.n_leg_joints   = len(leg_joint_names)
        self.left_body_idx  = range(0, self.n_body_joints)
        self.right_body_idx = range(self.n_body_joints, 2*self.n_body_joints)
        self.left_leg_idx   = range(2*self.n_body_joints, 2*self.n_body_joints + self.n_leg_joints)
        self.right_leg_idx  = range(2*self.n_body_joints + self.n_leg_joints, 2*self.n_body_joints + 2*self.n_leg_joints)
        self.left_idx       = np.concatenate([self.left_body_idx, self.left_leg_idx]).astype(int)
        self.right_idx      = np.concatenate([self.right_body_idx, self.right_leg_idx]).astype(int)

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            **kwargs,
    ):
        """Step function"""
        M_diff     = np.zeros(self.n_joints)
        M_sum      = np.zeros(self.n_joints)
        return M_diff, M_sum



class WaveController(SalamandraRoboticaController):
    """WaveController"""

    def __init__(self, animat_data, animat_options, experiment_options, n_joints, n_iterations, config):
        super().__init__(animat_data, animat_options, experiment_options, n_joints, n_iterations)
        self.config = config

        control_joint_names = [joint.joint_name for joint in self.animat_options.control.motors]
        body_joint_names    = [name for name in control_joint_names if "body" in name]
        self.n_body_joints  = len(body_joint_names)

        self.state    = np.zeros((self.n_iterations, 2*self.n_joints))

        self.config.amplitudes_left  = self.config.amp+self.config.bias
        self.config.amplitudes_right = self.config.amp-self.config.bias

    def step(self, iteration, time, timestep):
        """Compute neural activity"""
        time     = iteration * timestep
        aux_sine = np.sin(
            2*np.pi * ( self.config.freq*time - self.config.twl*np.arange(self.n_body_joints)/self.n_joints )
        )
        self.state[iteration, self.left_body_idx]  = self.config.amplitudes_left * (1+aux_sine)/2
        self.state[iteration, self.right_body_idx]  = self.config.amplitudes_right * (1-aux_sine)/2

        M_diff     = (self.state[iteration,self.left_idx] - self.state[iteration,self.right_idx])
        M_sum      = (self.state[iteration,self.left_idx] + self.state[iteration,self.right_idx])

        return M_diff, M_sum


class POscillatorController(SalamandraRoboticaController):
    """Phase Oscillator Controller"""

    def __init__(self, animat_data, animat_options, experiment_options, n_joints, n_iterations, config):
        super().__init__(animat_data, animat_options, experiment_options, n_joints, n_iterations)
        self.config                       = config                                          # unused in this example (could be used to set parameters)
        # self.state                        = np.zeros((self.n_iterations, 2*self.n_joints))  # 2 state variables - phi_L first, phi_R second
        # self.state[0,:self.n_joints]      = 1                                               # introduce asym

        self.state           = np.zeros((self.n_iterations, 2*self.n_joints))  # 2 state variables - phi_L first, phi_R second
        self.state[0]        = np.random.rand(2*self.n_joints)*2*np.pi         # introduce asym
        self.phase_lag       = 2*np.pi/15
        self.weight          = 10.0
        self.freq            = 1
        self.enable_coupling = False # if False, use purely decentralized controller
        self.weight_feedback = -100
        self.dstate          = np.zeros_like(self.state)
        self.solver          = integrate.ode(f=self.rhs)
        self.timestep        = experiment_options.simulation.physics.timestep
        self.solver.set_integrator('dopri5', atol=1e-6, rtol=1e-6)
        self.solver.set_initial_value(y=self.state[0], t=0.0)

    def couple(self, output, input, weight, phase_lag):
        """Unidirectional coupling function between oscillators"""
        return weight * np.sin(input - output - phase_lag)

    def rhs(self, time, state, iteration, joint_pos):
        """Right-hand side of the ODEs for the phase oscillators"""

        self.dstate[iteration] = 2*np.pi*self.freq

        phases_l = state[self.left_body_idx] # left phases
        phases_r = state[self.right_body_idx] # right phases

        if self.enable_coupling:
            # down coupling
            self.dstate[iteration, self.left_body_idx[1:]]  += self.couple(phases_l[1:],phases_l[:-1], self.weight, self.phase_lag)
            self.dstate[iteration, self.right_body_idx[1:]] += self.couple(phases_r[1:],phases_r[:-1], self.weight, self.phase_lag)

            # up coupling
            self.dstate[iteration, self.left_body_idx[:-1]]  += self.couple(phases_l[:-1],phases_l[1:], self.weight, -self.phase_lag)
            self.dstate[iteration, self.right_body_idx[:-1]] += self.couple(phases_r[:-1],phases_r[1:], self.weight, -self.phase_lag)

            # contralateral coupling
            self.dstate[iteration, self.left_body_idx]  += self.couple(phases_l, phases_r, self.weight, np.pi)
            self.dstate[iteration, self.right_body_idx] += self.couple(phases_r, phases_l, self.weight, np.pi)

        joint_pos_spine = joint_pos[:self.n_body_joints]

        self.dstate[iteration, self.left_body_idx[0]]  += self.couple(phases_l[0], phases_r[0], self.weight, np.pi)
        self.dstate[iteration, self.right_body_idx[0]] += self.couple(phases_r[0], phases_l[0], self.weight, np.pi)

        self.dstate[iteration, self.left_body_idx[1:]]  -= self.weight_feedback * joint_pos_spine[:-1] * np.sin(phases_l[1:])
        self.dstate[iteration, self.right_body_idx[1:]] += self.weight_feedback * joint_pos_spine[:-1] * np.sin(phases_r[1:])

        return self.dstate[iteration]

    def step(self, iteration, time, timestep):
        """Compute neural activity"""

        joint_positions = np.array(self.animat_data.sensors.joints.positions(iteration))

        # set inputs to the ODEs - i.e. feedback from joints, forces, etc (for closed loop control)
        self.solver.set_f_params(iteration, joint_positions)
        self.state[iteration] = self.solver.integrate(time+timestep)
        if not self.solver.successful():
            message = (
                f'ODE not integrated properly at {iteration=}'
                f' ({self.solver.t=} < {time+timestep=} [s])'
                f'\nReturn code: {self.solver.get_return_code()=}'
            )
            print(message)

        left_activities = np.concatenate([
            self.state[iteration, self.left_body_idx],
            self.state[iteration, self.left_leg_idx],
        ])
        right_activities = np.concatenate([
            self.state[iteration, self.right_body_idx],
            self.state[iteration, self.right_leg_idx],
        ])

        M_left  = 1+np.cos(left_activities)
        M_right = 1+np.cos(right_activities)
        M_diff  = M_right-M_left
        M_sum   = M_right+M_left

        return M_diff, M_sum




