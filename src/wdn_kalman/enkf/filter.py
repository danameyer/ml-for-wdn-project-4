"""EnKF setup."""

from dataclasses import dataclass
import numpy as np

@dataclass
class EnKFConfig:
    """Filter configuration."""
    ensemble_size: int = 50
    seed: int = 42

class GaussianNoiseModel:
    """Set up different covariance matrices to model uncertainty."""

    def __init__(
        self,
        state_dim,
        obs_dim,
        initial_covariance=None,
        process_covariance=None,
        measurement_covariance=None,
        seed=42,
    ):
        """initialise different noise matrices."""
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.seed = seed
        self.initial_covariance = (np.eye(state_dim) if initial_covariance is None else initial_covariance)
        self.process_covariance = (np.eye(state_dim)if process_covariance is None else process_covariance)
        self.measurement_covariance = (np.eye(obs_dim) if measurement_covariance is None else measurement_covariance)
        self.rng = np.random.default_rng(seed)

    def sample_initial_noise(self, count):
        """Sample from initial covariance matrix P."""
        return self.rng.multivariate_normal(
            mean=np.zeros(self.state_dim),
            cov=self.initial_covariance,
            size=count,
        )

    def sample_process_noise(self, count):
        """Sample from process covariance matrix Q."""
        return self.rng.multivariate_normal(
            mean=np.zeros(self.state_dim),
            cov=self.process_covariance,
            size=count
        )

    def sample_measurement_noise(self, count):
        """Sample from measurement covariance matrix R."""
        return self.rng.multivariate_normal(
            mean=np.zeros(self.obs_dim),
            cov=self.measurement_covariance,
            size=count
        )

    def reset(self):
        """Reset random number generator."""
        self.rng = np.random.default_rng(self.seed)


class EnsembleState:
    """Store ensemble members and calculate ensemble statistics."""

    def __init__(self, initial_state, ensemble_size, noise_model):
        """Initialize ensemble."""
        self.initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        self.ensemble_size = ensemble_size
        self.noise_model = noise_model
        self.members = np.empty((self.ensemble_size, self.initial_state.size), dtype=float)
        self.reset_mean()

    @property
    def mean(self):
        """return ensemble mean (best filter estimate)."""
        return self.members.mean(axis=0)

    @property
    def covariance(self):
        """Capture uncertainty and correlations of variables."""
        return np.cov(self.members, rowvar=False, ddof=1)

    def reset_mean(self):
        """regenerate the ensemble so it is centred on the initial state."""
        noise = self.noise_model.sample_initial_noise(self.ensemble_size)
        noise -= noise.mean(axis=0, keepdims=True)
        self.members = (self.initial_state.reshape(1, -1) + noise)

    def replace_members(self, new_members):
        """add new ensemble members."""
        self.members = np.asarray(new_members, dtype=float)

    def add_changes(self, changes):
        """add changes to ensemble members."""
        self.members += np.asarray(changes, dtype=float)

    def shift_mean(self, desired_mean):
        """shift ensemble mean to desired value, preserving spread."""
        desired_mean = np.asarray(desired_mean, dtype=float).reshape(-1)
        self.members += (desired_mean - self.mean).reshape(1, -1)

    def set_values(self, indices, values):
        """Shift ensemble so mean matches observed flows."""
        indices = np.asarray(indices, dtype=int)
        values = np.asarray(values, dtype=float)
        shifts = (np.asarray(values)- self.mean[indices])
        self.members[:, indices] += shifts

    def set_exact_values(self, indices, values):
        """Set selected state values exactly for all ensemble members."""
        indices = np.asarray(indices, dtype=int)
        values = np.asarray(values,dtype=float)
        self.members[:, indices] = values


class TimeVaryingEnsembleKalmanFilter:
    """Set up time-varying EnKF."""

    def __init__(
        self,
        state_dim,
        obs_dim,
        init_state,
        get_state_transition_func,
        get_measurement_func,
        init_state_uncertainty_cov=None,
        measurement_uncertainty_cov=None,
        system_uncertainty_cov=None,
        config=None,
    ):
        """set up Ensemble Kalman Filter with parameters."""
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.config = config or EnKFConfig()
        self._get_state_transition_func = get_state_transition_func
        self._get_measurement_func = get_measurement_func
        self._initial_state = np.asarray(init_state,dtype=float)

        self._noise = GaussianNoiseModel(
            state_dim=state_dim,
            obs_dim=obs_dim,
            initial_covariance=init_state_uncertainty_cov,
            process_covariance=system_uncertainty_cov,
            measurement_covariance=measurement_uncertainty_cov,
            seed=self.config.seed,
            )

        self._state = EnsembleState(
            initial_state=self._initial_state,
            ensemble_size=self.config.ensemble_size,
            noise_model=self._noise,
        )

        self._time_step = 0
        self._x = self._state.mean.copy()

    @property
    def time_step(self):
        """Return the current internal filter time index."""
        return self._time_step

    @property
    def ensemble(self):
        """return ensemble members."""
        return self._state.members.copy()

    @property
    def mean(self):
        """return ensemble mean."""
        return self._state.mean.copy()

    @property
    def covariance(self):
        """return ensemble covariance."""
        return self._state.covariance.copy()

    def step(self, observation):
        """
        Perform one prediction and correction step.

        Returns:
            corrected state mean
            corrected state covariance
        """
        self._state.shift_mean(self._x)
        transition = self._get_state_transition_func(self._time_step)
        measurement = self._get_measurement_func(self._time_step)
        self._predict(transition)
        self._correct(observation, measurement)
        self._x = self._state.mean.copy()
        self._time_step += 1

        return self._x.copy(), self._state.covariance.copy()

    def _predict(self, transition):
        """Propagate members through the surrogate model and add Q noise."""
        predicted_members = np.array([transition(member) for member in self._state.members])
        process_noise = (self._noise.sample_process_noise(self.config.ensemble_size))
        self._state.replace_members(predicted_members + process_noise)

    def _correct(self, observation, measurement):
        """Correct ensemble members using the sensor observations."""
        predicted_observations = np.array([
            measurement(member)
            for member in self._state.members
        ])
        state_anomalies = self._state.members - self._state.mean

        observation_anomalies = predicted_observations - predicted_observations.mean(axis=0)

        denominator = self.config.ensemble_size - 1

        cross_covariance = state_anomalies.T @ observation_anomalies / denominator

        observation_covariance = observation_anomalies.T @ observation_anomalies / denominator + self._noise.measurement_covariance

        kalman_gain = np.linalg.solve(observation_covariance.T, cross_covariance.T).T

        measurement_noise = self._noise.sample_measurement_noise(self.config.ensemble_size)

        perturbed_observations = np.asarray(observation) + measurement_noise

        innovations = perturbed_observations - predicted_observations

        corrections = innovations @ kalman_gain.T

        self._state.add_changes(corrections)

    def set_state_values(self, indices, values):
        """insert known state values (e.g. measured flows)."""
        self._state.set_values(indices=indices,values=values)
        self._x = self._state.mean.copy()

    def set_exact_state_values(self, indices, values):
        """Set selected state values exactly."""
        self._state.set_exact_values(indices=indices, values=values)
        self._x = self._state.mean.copy()

    def reset(self):
        """Reset ensemble, random seed and time step."""
        self._noise.reset()
        self._state.reset_mean()
        self._time_step = 0
        self._x = self._state.mean.copy()