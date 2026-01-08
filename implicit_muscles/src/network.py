"""Network"""

from abc import ABC, abstractmethod
import numpy as np
from farms_core.model.data import AnimatData
from scipy.integrate._ode import ode
from scipy import integrate

class NNController(ABC):
    """NN controller"""

    def __init__(self, animat_data, n_joints, n_iterations):
        super().__init__()
        self.animat_data: AnimatData = animat_data
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
        """Step function called at each simulation iteration"""


class DummyController(NNController):
    """Dummy controller"""

    def __init__(self, animat_data, n_joints, n_iterations, config):
        super().__init__(animat_data, n_joints, n_iterations)

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



class WaveController(NNController):
    """WaveController"""

    def __init__(self, animat_data, n_joints, n_iterations, config):
        super().__init__(animat_data, n_joints, n_iterations)
        self.config = config

        self.state    = np.zeros((self.n_iterations, 2*self.n_joints))

        self.muscle_l = 2*np.arange(0, self.n_joints) # indexes of the left muscle
        self.muscle_r = self.muscle_l+1

        self.config.amplitudes_left  = self.config.amp+self.config.bias
        self.config.amplitudes_right = self.config.amp-self.config.bias

    def step(self, iteration, time, timestep):
        """Compute neural activity"""
        time     = iteration * timestep
        aux_sine = np.sin(
            2*np.pi * ( self.config.freq*time - self.config.twl*np.arange(self.n_joints)/self.n_joints )
        )
        self.state[iteration, self.muscle_l]  = self.config.amplitudes_left * (1+aux_sine)/2
        self.state[iteration, self.muscle_r]  = self.config.amplitudes_right * (1-aux_sine)/2

        M_diff     = (self.state[iteration,self.muscle_l] - self.state[iteration,self.muscle_r])
        M_sum      = (self.state[iteration,self.muscle_l] + self.state[iteration,self.muscle_r])

        return M_diff, M_sum


class POscillatorController(NNController):
    """Phase Oscillator Controller"""

    def __init__(self, animat_data, n_joints, n_iterations, config):
        super().__init__(animat_data, n_joints, n_iterations)
        self.config                       = config                                          # unused in this example (could be used to set parameters)
        # self.state                        = np.zeros((self.n_iterations, 2*self.n_joints))  # 2 state variables - phi_L first, phi_R second
        # self.state[0,:self.n_joints]      = 1                                               # introduce asym

        self.state           = np.zeros((self.n_iterations, 2*self.n_joints))  # 2 state variables - phi_L first, phi_R second
        self.state[0]        = np.random.rand(2*self.n_joints)*2*np.pi         # introduce asym
        self.phase_lag       = 2*np.pi/20
        self.weight          = 10.0
        self.freq            = 1
        self.enable_coupling = False
        self.weight_feedback = 0.3
        self.dstate          = np.zeros_like(self.state)
        self.solver          = integrate.ode(f=self.rhs)
        self.solver.set_integrator('dopri5', atol=1e-6, rtol=1e-6)
        self.solver.set_initial_value(y=self.state[0], t=0.0)

    def couple(self, output, input, weight, phase_lag):
        """Unidirectional coupling function between oscillators"""
        return weight * np.sin(input - output - phase_lag)

    def rhs(self, time, state, iteration, joint_pos):
        """Right-hand side of the ODEs for the phase oscillators"""

        self.dstate[iteration] = 2*np.pi*self.freq

        phases_l = state[:self.n_joints] # left phases
        phases_r = state[self.n_joints:] # right phases

        if self.enable_coupling:
            # down coupling
            self.dstate[iteration, 1:self.n_joints]  += self.couple(phases_l[1:],phases_l[:-1], self.weight, self.phase_lag)
            self.dstate[iteration, self.n_joints+1:] += self.couple(phases_r[1:],phases_r[:-1], self.weight, self.phase_lag)

            # up coupling
            self.dstate[iteration, :self.n_joints-1] += self.couple(phases_l[:-1],phases_l[1:], self.weight, -self.phase_lag)
            self.dstate[iteration, self.n_joints:-1] += self.couple(phases_r[:-1],phases_r[1:], self.weight, -self.phase_lag)

            # contralateral coupling
            self.dstate[iteration, :self.n_joints] += self.couple(phases_l, phases_r, self.weight, np.pi)
            self.dstate[iteration, self.n_joints:] += self.couple(phases_r, phases_l, self.weight, np.pi)

        # add stretch sensors in style of the zebrafish spinal cord
        # right_sensors = np.maximum(-joint_pos,0)
        # left_sensors  = np.maximum(joint_pos,0)

        # self.dstate[iteration, :self.n_joints-1] += self.weight_feedback * right_sensors[1:] * np.sin(phases_l[:-1])
        # self.dstate[iteration, self.n_joints:-1] += self.weight_feedback * left_sensors[1:] * np.sin(phases_r[:-1])

        # self.dstate[iteration, 1:self.n_joints] += self.weight_feedback * right_sensors[:-1] * np.sin(phases_l[1:])
        # self.dstate[iteration, 1+self.n_joints:] += self.weight_feedback * left_sensors[:-1] * np.sin(phases_r[1:])

        # self.dstate[iteration, :self.n_joints] += self.weight_feedback * (right_sensors-left_sensors) * np.sin(phases_l)
        # self.dstate[iteration, self.n_joints:] += self.weight_feedback * (left_sensors-right_sensors) * np.sin(phases_r)

        # self.dstate[iteration, :self.n_joints] += self.weight_feedback * (right_sensors) * np.sin(phases_l)
        # self.dstate[iteration, self.n_joints:] += self.weight_feedback * (left_sensors) * np.sin(phases_r)

        self.dstate[iteration, 1:self.n_joints]  -= self.weight_feedback * joint_pos[:-1] * np.sin(phases_l[1:])
        self.dstate[iteration, self.n_joints+1:] += self.weight_feedback * joint_pos[:-1] * np.sin(phases_r[1:])

        # self.dstate[iteration, :self.n_joints-1] -= self.weight_feedback * joint_pos[1:] * np.sin(phases_l[:self.n_joints-1])
        # self.dstate[iteration, self.n_joints:-1] += self.weight_feedback * joint_pos[1:] * np.sin(phases_r[:self.n_joints-1])

        # self.dstate[iteration, :self.n_joints] -= self.weight_feedback * joint_pos * np.sin(phases_l)
        # self.dstate[iteration, self.n_joints:] += self.weight_feedback * joint_pos * np.sin(phases_r)

        return self.dstate[iteration]

    def step(self, iteration, time, timestep):
        """Compute neural activity"""

        # set inputs to the ODEs - i.e. feedback from joints, forces, etc (for closed loop control)
        joint_positions = np.array(self.animat_data.sensors.joints.positions(iteration))
        self.solver.set_f_params(iteration, joint_positions)

        # integrate ODEs
        self.state[iteration] = self.solver.integrate(time+timestep)

        if not self.solver.successful():
            message = (
                f'ODE not integrated properly at {iteration=}'
                f' ({self.solver.t=} < {time+timestep=} [s])'
                f'\nReturn code: {self.solver.get_return_code()=}'
            )
            print(message)

        M_left  = 1+np.cos(self.state[iteration, :self.n_joints])
        M_right = 1+np.cos(self.state[iteration, self.n_joints:])
        M_diff  = M_right-M_left
        M_sum   = M_right+M_left

        # M_diff[:]=1
        # M_sum[:]=0

        return M_diff, M_sum




