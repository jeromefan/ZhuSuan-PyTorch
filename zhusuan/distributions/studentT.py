import torch
from zhusuan.distributions import Distribution

class StudentT(Distribution):
    """
    The class of univariate StudentT distribution
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param df: A 'float' Var. Degrees of freedom.
    :param loc: A 'float' Var. Mean of the StudentT distribution.
    :param scale: A 'float' Var. Scale of the StudentT distribution.
    """
    def __init__(self,
                dtype=torch.float32,
                is_continues=True,
                group_ndims=0,
                device=torch.device('cpu'),
                **kwargs):
        super(StudentT, self).__init__(dtype,
                                       is_continues,
                                       is_reparameterized=False, # reparameterization trick is not applied for Laplace distribution
                                       group_ndims=group_ndims,
                                       device=device,
                                       **kwargs)

        self._df = torch.as_tensor(kwargs['df'], dtype = self._dtype).to(device) if type(kwargs['df']) in [int, float] else kwargs['df'].to(device)
        self._loc = torch.as_tensor(kwargs['loc'], dtype = self._dtype).to(device) if type(kwargs['loc']) in [int, float] else kwargs['loc'].to(device)
        self._scale = torch.as_tensor(kwargs['scale'], dtype = self._dtype).to(device) if type(kwargs['scale']) in [int, float] else kwargs['scale'].to(device)

    @property
    def df(self):
        """Degrees of freedom."""
        return self._df

    @property
    def loc(self):
        """Mean of the Laplace distribution."""
        return self._loc

    @property
    def scale(self):
        """Scale of the Laplace distribution."""
        return self._scale

    def _sample(self, n_samples=1):
        if n_samples > 1:
            _shape = self._loc.shape
            _shape = torch.Size([n_samples]) + _shape
            _len = len(self._loc.shape)
            _loc = self._loc.repeat([n_samples, *_len * [1]])
            _scale = self._scale.repeat([n_samples, *_len * [1]])
            _df = self._df.repeat([n_samples, *_len * [1]])
        else:
            _shape = self._loc.shape
            _loc = torch.as_tensor(self._loc, dtype=self._dtype)
            _scale = torch.as_tensor(self._scale, dtype=self._dtype)
            _df = torch.as_tensor(self._df, dtype=self._dtype)

        _sample = torch.distributions.studentT.StudentT(_df, _loc, _scale).sample()

        return _sample

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache
        if len(sample.shape) > len(self._loc.shape):
            n_samples = sample.shape[0]
            _len = len(self._loc.shape)
            _loc = self._loc.repeat([n_samples, *_len * [1]])
            _scale = self._scale.repeat([n_samples, *_len * [1]])
            _df = self._df.repeat([n_samples, *_len * [1]])
        else:
            _loc = self._loc
            _scale = self._scale
            _df = self._df

        return torch.distributions.studentT.StudentT(_df, _loc, _scale).log_prob(sample)