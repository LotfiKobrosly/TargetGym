import numpy as np

from classes.gaussian_kernel import GaussianKernel
from utils.models import code
from utils.constants import RELEVANCE_THRESHOLD, RANDOM_STATE


class ContinuousGaussianDictionary(dict):

    def __init__(self, kernel_radius: float):
        dict.__init__(self)
        self.kernel_radius = kernel_radius

    def __getitem__(self, key):
        position = code(key)
        if len(self) == 0:
            raise KeyError("Dict is empty")
        if position in self.keys():
            return dict.__getitem__(self, position)
        else:
            kernel = GaussianKernel(position, self.kernel_radius)
            values, weights = list(), list()
            n_relevant_elements = 0
            for other_key in self.keys():
                action_shape = len(
                    np.reshape(list(dict.__getitem__(self, other_key)), (-1, 1))
                )
                if (
                    np.linalg.norm(np.array(other_key) - np.array(position))
                    <= RELEVANCE_THRESHOLD
                ):
                    values.append(
                        np.asarray(list(dict.__getitem__(self, other_key))).flatten()
                    )
                    weights.append(kernel.pdf(other_key))
                    n_relevant_elements += 1

            if weights:
                try:
                    values = np.array(values)
                except ValueError:
                    print("Issue here:", values)
                    print("Relevant elements:", n_relevant_elements)
                    raise ValueError
                weights = np.array(weights)
                # assert not (np.any(np.isnan(weights))), "NaN in weights " + str(weights)
                # assert not (np.any(np.isnan(values))), "NaN in values " + str(np.where(np.isnan(values)))
                weights /= np.sum(weights)
                weights = np.reshape(weights, (values.shape[0], 1))

                returned_action = np.dot(weights.T, values)
                # print(returned_action.shape)

                return returned_action  # .flatten()
            else:
                return RANDOM_STATE.uniform(-1, 1, size=action_shape)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def predict(self, key, *args, **kwargs):
        return (
            self[key],
            None,
        )  # None is added to cope with the predict function of the pc-gym package


class ContinuousByRegionDictionary(dict):
    def __init__(self):
        dict.__init__(self)

    def __getitem__(self, key):
        region = None
        # key = code(key)
        if key in self.keys():
            return dict.__getitem__(self, key)
        for existing_key in self.keys():
            if np.all(
                (np.array(list(key)) >= np.array(list(existing_key[0])))
                & (np.array(list(key)) <= np.array(list(existing_key[1])))
            ):
                region = existing_key
                break
        if region is None:
            raise ValueError(
                "Element not in the covered region OR dict keys do not cover the whole action space: "
                + str(key)
            )
        return dict.__getitem__(self, region)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def predict(self, key, *args, **kwargs):
        return (
            self[key],
            None,
        )  # None is added to cope with the predict function of the pc-gym package
