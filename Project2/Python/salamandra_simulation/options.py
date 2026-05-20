"""Animat options"""

from typing import List
from farms_core.options import Options
from farms_core.model.options import (
    AnimatOptions,
    MorphologyOptions,
    LinkOptions,
    JointOptions,
    SpawnOptions,
    ControlOptions,
    MotorOptions,
    SensorsOptions,
)


class SalamandraOptions(AnimatOptions):
    """Simulation options"""

    def __init__(self, sdf: str, **kwargs):
        super().__init__(
            sdf=sdf,
            spawn=SpawnOptions(**kwargs.pop('spawn')),
            morphology=SalamandraMorphologyOptions(**kwargs.pop('morphology')),
            control=SalamandraControlOptions(**kwargs.pop('control')),
        )
        self.name = kwargs.pop('name')
        self.physics = SalamandraPhysicsOptions(**kwargs.pop('physics'))
        self.mujoco = kwargs.pop('mujoco')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class SalamandraMorphologyOptions(MorphologyOptions):
    """Salamandra morphology options"""

    def __init__(self, **kwargs):
        super().__init__(
            links=[
                SalamandraLinkOptions(**link)
                for link in kwargs.pop('links')
            ],
            self_collisions=kwargs.pop('self_collisions'),
            joints=[
                JointOptions(**joint)
                for joint in kwargs.pop('joints')
            ],
        )
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class SalamandraLinkOptions(LinkOptions):
    """Salamandra link options"""

    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.pop('name'),
            collisions=kwargs.pop('collisions'),
            friction=kwargs.pop('friction'),
            extras=kwargs.pop('extras', {}),
        )
        self.density = kwargs.pop('density')
        self.swimming = kwargs.pop('swimming')
        self.drag_coefficients = kwargs.pop('drag_coefficients')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class SalamandraPhysicsOptions(Options):
    """Salamandra physics options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.drag = kwargs.pop('drag')
        self.sph = kwargs.pop('sph')
        self.buoyancy = kwargs.pop('buoyancy')
        self.viscosity = kwargs.pop('viscosity')
        self.water_height = kwargs.pop('water_height')
        self.water_density = kwargs.pop('water_density')
        self.water_velocity = kwargs.pop('water_velocity')
        self.water_maps = kwargs.pop('water_maps')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class SalamandraControlOptions(ControlOptions):
    """Salamandra control options"""

    def __init__(self, **kwargs):
        super().__init__(
            sensors=(SalamandraSensorsOptions(**kwargs.pop('sensors'))),
            motors=[
                MotorOptions(**motor)
                for motor in kwargs.pop('motors')
            ],
        )
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class SalamandraSensorsOptions(SensorsOptions):
    """Salamandra sensors options"""

    def __init__(self, **kwargs):
        super().__init__(
            links=kwargs.pop('links'),
            joints=kwargs.pop('joints'),
            contacts=kwargs.pop('contacts'),
            xfrc=kwargs.pop('xfrc'),
        )
        assert not kwargs, f'Unknown kwargs: {kwargs}'

