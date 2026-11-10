import numpy as np
from target_gym.registry import REGISTRY


def get_environment(
    problem: str = None,
    *args,
    **kwargs,
):
    assert problem in REGISTRY.keys(), "Unknown get_environment"
    specs = REGISTRY[problem]
    return specs.make_env(), specs.params_cls()


def code(control_instance):
    if isinstance(control_instance, np.ndarray):
        control_instance = control_instance.flatten()
    control_instance_tuple = ()
    for element in control_instance:
        if not isinstance(element, (float, int)):
            element = element.tolist()
        control_instance_tuple += (round(element, 3),)
    return control_instance_tuple
