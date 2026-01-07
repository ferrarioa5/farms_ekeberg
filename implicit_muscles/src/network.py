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

        self.config.amplitudes_left = self.config.amp+self.config.bias
        self.config.amplitudes_right = self.config.amp-self.config.bias


    def initialize_episode(self):
        """Initialize episode"""
        pass


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


class OscillatorController(NNController):
    """OscillatorController"""

    def __init__(self, animat_data, n_joints, n_iterations, config):
        super().__init__(animat_data, n_joints, n_iterations)
        self.config = config
        self.state = np.zeros_like(self.n_iterations, 4*self.n_joints) # 4 state variables - phi_L, phi_R, A_L, A_R
        self.dstate = np.zeros_like(self.state)
        
    def initialize_episode(self):
        """Initialize episode"""
        self.solver: ode = integrate.ode(f=self.rhs)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state[0], t=0.0)

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
