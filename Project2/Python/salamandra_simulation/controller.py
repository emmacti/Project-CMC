"""Network controller"""

import numpy as np
import farms_core.pylog as pylog
from farms_core.model.control import AnimatController
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.experiment.options import ExperimentOptions
from farms_amphibious.control.amphibious import AmphibiousController


class SalamandraController(AmphibiousController):

    def before_step(self, task, action, physics):
        """Before step"""

        time = physics.time()/task.units.seconds
        timestep = physics.timestep()/task.units.seconds
        iteration = task.iteration
        # contact_sens = np.asarray(
        #     self.animat_data.sensors.contacts.array[iteration, :, 2]
        # )

        self.network.robot_parameters.step(
            time=time,
            iteration=iteration,
            salamandra_data=self.animat_data,
        )

        # super().before_step(task, action, physics)
        index = task.iteration % task.buffer_size
        # if self.drive is not None:
        #     self.drive.step(iteration, time, timestep)
        if self.network is not None:
            self.network.step(
                iteration,
                time,
                timestep,
                None,
                contact_sens=None)
        for net2joints in self.network2joints.values():
            if net2joints is not None:
                net2joints.step(iteration)

        # time = physics.time()/task.units.seconds
        # timestep = physics.timestep()/task.units.seconds
        index = task.iteration % task.buffer_size
        self.animat_data.state.array[index,
                                     :] = self.network.state.array[index, :]

        # Limb offsets
        joints_offsets_index = 4*16+8
        # Forelimbs joints
        self.animat_data.state.array[index,
                                     joints_offsets_index+1::2] = 0.2*np.pi
        # Hindlimbs joints
        self.animat_data.state.array[index,
                                     joints_offsets_index+4+1::2] = 0.3*np.pi
