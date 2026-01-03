import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.nn.functional import normalize
from torch.distributions import Beta


class MixupNormalAugmentor:
    """Mixup from normal samples with selected genes"""

    def __init__(self, df_tpm_normal, genes2mixup=[], mix_beta=0.9, n_views=1):
        """Selective mixup initialization."""
        if len(genes2mixup) == 0:
            genes2mixup = df_tpm_normal.columns

        X = torch.tensor(df_tpm_normal.values, dtype=torch.float32).clone().detach()

        self.genes2mixup = genes2mixup
        self.select_idx = df_tpm_normal.columns.get_indexer(genes2mixup)
        self.X_mix = X[:, self.select_idx]

        self.n = len(df_tpm_normal)
        self.beta = mix_beta
        self.p = Beta(torch.FloatTensor([mix_beta]), torch.FloatTensor([mix_beta]))
        self.n_views = n_views

    def _transform(self, x):
        b1 = self.beta
        m1 = self.X_mix[np.random.choice(self.n)]
        x1 = x.clone().detach()
        xs = x1[self.select_idx]
        xs = b1 * xs + (1 - b1) * m1
        x1[self.select_idx] = xs
        return x1.to(x.device)

    def augment(self, x):
        return [self._transform(x) for _ in range(self.n_views)]

    def __repr__(self):
        return f"{self.__class__.__name__}(beta={self.beta}, n_views={self.n_views})"


class RandomMaskAugmentor:
    """Random Mask Augmentation for Gene Expression Vectors."""

    def __init__(
        self,
        mask_p_prob=0.1,
        mask_a_prob=None,
        mask_n_prob=None,
        no_augment_prob=0.1,
        n_views=1,
    ):
        if mask_a_prob is None:
            mask_a_prob = mask_p_prob
        if mask_n_prob is None:
            mask_n_prob = mask_p_prob

        self.mask_p_prob = mask_p_prob
        self.mask_a_prob = mask_a_prob
        self.mask_n_prob = mask_n_prob
        self.no_augment_prob = no_augment_prob
        self.n_views = n_views

    def _transform(self, x, probability):
        if torch.rand(1).item() < self.no_augment_prob:
            return x.clone()

        random_values = torch.rand(len(x[1:]), device=x.device)
        mask = random_values < probability

        x_new = x.clone()
        x_new[1:][mask] = 0
        return x_new

    def _augment(self, x, mask_prob):
        assert len(x.shape) == 1, "augment for each sample on a vector!"
        return [self._transform(x, mask_prob) for _ in range(self.n_views)]

    def augment_p(self, p):
        return self._augment(p, self.mask_p_prob)

    def augment_a(self, a):
        return self._augment(a, self.mask_a_prob)

    def augment_n(self, n):
        return self._augment(n, self.mask_n_prob)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mask_probability=(a:{self.mask_a_prob}, "
            f"p:{self.mask_p_prob}, n:{self.mask_n_prob}), "
            f"no={self.no_augment_prob}, n_views={self.n_views})"
        )


class FeatureJitterAugmentor:
    """Feature Jittering for Gene Expression Vectors."""

    def __init__(
        self,
        jitter_p_std=0.1,
        jitter_a_std=None,
        jitter_n_std=None,
        no_augment_prob=0.1,
        n_views=1,
    ):
        if jitter_a_std is None:
            jitter_a_std = jitter_p_std
        if jitter_n_std is None:
            jitter_n_std = jitter_p_std

        self.n_views = n_views
        self.jitter_p_std = jitter_p_std
        self.jitter_a_std = jitter_a_std
        self.jitter_n_std = jitter_n_std
        self.no_augment_prob = no_augment_prob

    def _transform(self, x, jitter_std):
        if torch.rand(1) < self.no_augment_prob:
            return x.clone()

        jitter = torch.normal(
            mean=0, std=jitter_std, size=x[1:].size(), device=x.device
        )
        x_jittered = x.clone()
        x_jittered[1:] += jitter
        return x_jittered

    def _augment(self, x, jitter_std):
        assert len(x.shape) == 1, "augment for each sample on a vector!"
        return [self._transform(x, jitter_std) for _ in range(self.n_views)]

    def augment_p(self, p):
        return self._augment(p, self.jitter_p_std)

    def augment_a(self, a):
        return self._augment(a, self.jitter_a_std)

    def augment_n(self, n):
        return self._augment(n, self.jitter_n_std)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(jitter_std=(a:{self.jitter_a_std}, "
            f"p:{self.jitter_p_std}, n:{self.jitter_n_std}, "
            f"no:{self.no_augment_prob}), n_views={self.n_views})"
        )


class MaskJitterAugmentor:
    """Mixed augmentation (Masking and Jittering)."""

    def __init__(
        self,
        mask_p_prob=0.01,
        mask_a_prob=None,
        mask_n_prob=0.0,
        jitter_p_std=0.01,
        jitter_a_std=None,
        jitter_n_std=0.0,
        no_augment_prob=0.1,
        n_views=1,
    ):
        self.mask_p_prob = mask_p_prob
        self.mask_a_prob = mask_a_prob
        self.mask_n_prob = mask_n_prob
        self.jitter_p_std = jitter_p_std
        self.jitter_a_std = jitter_a_std
        self.jitter_n_std = jitter_n_std
        self.no_augment_prob = no_augment_prob
        self.n_views = n_views

        self.augmentor1 = RandomMaskAugmentor(
            mask_p_prob, mask_a_prob, mask_n_prob, no_augment_prob, n_views
        )
        self.augmentor2 = FeatureJitterAugmentor(
            jitter_p_std, jitter_a_std, jitter_n_std, no_augment_prob, n_views
        )

    def augment_p(self, p):
        augmentor = np.random.choice([self.augmentor1, self.augmentor2])
        return augmentor.augment_p(p)

    def augment_a(self, a):
        augmentor = np.random.choice([self.augmentor1, self.augmentor2])
        return augmentor.augment_a(a)

    def augment_n(self, n):
        augmentor = np.random.choice([self.augmentor1, self.augmentor2])
        return augmentor.augment_n(n)
