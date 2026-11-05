"""Tests for rating interaction terms and their integration with DataGenerator."""

import numpy as np
import pytest

from bmf.config import DGConfig
from bmf.data_generators import DataGenerator
from bmf.distributions import GaussianDistribution
from bmf.interactions import (
    BilinearInteraction,
    EuclideanInteraction,
    HigherOrderInteraction,
)


@pytest.fixture
def uv():
    """A spread of user/item ideology pairs, including negatives and zero."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, 200), rng.normal(0, 1, 200)


def make_dg_config(**overrides) -> DGConfig:
    """DGConfig with concrete distribution objects, full cartesian ratings."""
    params = dict(
        user_distribution=GaussianDistribution(mean=0.0, std=1.0),
        item_ideology_distribution=GaussianDistribution(mean=0.0, std=1.0),
        item_quality_distribution=GaussianDistribution(mean=0.0, std=1.0),
        num_users=40,
        num_items=25,
        weight=0.0,
        weight_mode='additive',
        extremize='both',
        extremize_percentile=1.0,
        select_same=1.0,
        select_opp=1.0,
        select_mode='single',
        random_rating=False,
        noise_std=0.0,
        binary_ratings=False,
    )
    params.update(overrides)
    return DGConfig(**params)


class TestInteractionMath:
    """Each kernel computes its documented formula."""

    def test_bilinear_is_product(self, uv):
        u, v = uv
        np.testing.assert_allclose(BilinearInteraction()(u, v), u * v)

    def test_euclidean_is_negative_squared_distance(self, uv):
        u, v = uv
        np.testing.assert_allclose(EuclideanInteraction()(u, v), -((u - v) ** 2))

    def test_higher_order_formula(self, uv):
        u, v = uv
        phi = 0.3
        expected = u * v + phi * (u ** 2) * (v ** 2)
        np.testing.assert_allclose(HigherOrderInteraction(phi=phi)(u, v), expected)

    def test_higher_order_reduces_to_bilinear_at_zero_phi(self, uv):
        u, v = uv
        np.testing.assert_allclose(
            HigherOrderInteraction(phi=0.0)(u, v),
            BilinearInteraction()(u, v),
        )


class TestEuclideanSign:
    """Euclidean must encode closer-is-better, not the reverse."""

    def test_closer_scores_higher(self):
        g = EuclideanInteraction()
        near = g(np.array([0.5]), np.array([0.6]))
        far = g(np.array([0.5]), np.array([2.5]))
        assert near > far

    def test_exact_match_is_maximal(self, uv):
        u, _ = uv
        g = EuclideanInteraction()
        assert np.all(g(u, u) == 0.0)
        assert np.all(g(u, u + 0.1) < 0.0)

    def test_monotone_decreasing_in_distance(self):
        g = EuclideanInteraction()
        u = np.zeros(50)
        v = np.linspace(0, 5, 50)
        values = g(u, v)
        assert np.all(np.diff(values) < 0)


class TestVectorization:
    """Kernels stay elementwise over observation arrays."""

    @pytest.mark.parametrize("interaction", [
        BilinearInteraction(),
        EuclideanInteraction(),
        HigherOrderInteraction(phi=0.2),
    ])
    def test_shape_preserved(self, interaction, uv):
        u, v = uv
        assert interaction(u, v).shape == u.shape

    @pytest.mark.parametrize("interaction", [
        BilinearInteraction(),
        EuclideanInteraction(),
        HigherOrderInteraction(phi=0.2),
    ])
    def test_elementwise_matches_scalar_calls(self, interaction, uv):
        u, v = uv
        vectorized = interaction(u, v)
        elementwise = np.array([
            interaction(np.array([ui]), np.array([vi]))[0]
            for ui, vi in zip(u[:20], v[:20])
        ])
        np.testing.assert_allclose(vectorized[:20], elementwise)


class TestBackwardCompatibility:
    """Existing configs must be unaffected by the interaction seam."""

    def test_unset_interaction_defaults_to_bilinear(self):
        """interaction=None (old configs) == explicit BilinearInteraction."""
        default = DataGenerator(make_dg_config(interaction=None)).generate(seed=7)
        explicit = DataGenerator(
            make_dg_config(interaction=BilinearInteraction())
        ).generate(seed=7)

        np.testing.assert_array_equal(
            default.ratings['rating'].values,
            explicit.ratings['rating'].values,
        )
        np.testing.assert_array_equal(
            default.ratings['user_utility'].values,
            explicit.ratings['user_utility'].values,
        )

    def test_default_reproduces_bilinear_formula(self):
        """Default ratings equal q_j + u_i * v_j exactly (noise disabled)."""
        data = DataGenerator(make_dg_config(interaction=None)).generate(seed=11)

        merged = (
            data.ratings
            .merge(data.users, on='user_id')
            .merge(data.items, on='item_id')
        )
        expected = (
            merged['true_item_quality']
            + merged['true_user_ideology'] * merged['true_item_ideology']
        )
        np.testing.assert_allclose(
            merged['rating_no_noise'].values, expected.values, rtol=1e-10
        )


class TestSeedStability:
    """Adding dg.interaction must not perturb seeds of pre-existing configs."""

    @staticmethod
    def _experiment(interaction):
        from omegaconf import OmegaConf
        from bmf.config import ExperimentConfig, ModelConfig
        from bmf.experiment import Experiment

        cfg = ExperimentConfig(
            dg=DGConfig(
                user_distribution='g',
                item_ideology_distribution='g',
                item_quality_distribution='g',
                interaction=interaction,
            ),
            model=ModelConfig(constraint='none'),
        )
        return Experiment(OmegaConf.structured(cfg))

    # dg fields added after the original experiments were run. The legacy seed
    # reference must drop all of them; each is excluded from the real hash when
    # left at its disabling default.
    ADDED_DG_FIELDS = ('interaction', 'confounder_user_std', 'confounder_item_std')

    @staticmethod
    def _seed_without_interaction_field(rep):
        """Seed as computed before any of ADDED_DG_FIELDS existed."""
        import hashlib
        import json
        from dataclasses import asdict
        from bmf.config import ExperimentConfig, ModelConfig

        cfg = ExperimentConfig(
            dg=DGConfig(
                user_distribution='g',
                item_ideology_distribution='g',
                item_quality_distribution='g',
            ),
            model=ModelConfig(constraint='none'),
        )
        as_dict = asdict(cfg)
        as_dict.pop('seed_scope', None)  # top-level field added after those runs
        # logging credentials never enter the seed (see Experiment._compute_data_seed)
        as_dict.pop('comet_api_key', None)
        as_dict.pop('comet_workspace', None)
        for field in TestSeedStability.ADDED_DG_FIELDS:
            as_dict['dg'].pop(field, None)
        digest = hashlib.md5(
            json.dumps(as_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        return cfg.random_seed + rep * 100000 + (int(digest, 16) % 100000)

    @pytest.mark.parametrize("rep", [0, 1, 5])
    @pytest.mark.parametrize("interaction", [
        None,
        {'_target_': 'bmf.interactions.BilinearInteraction', 'name': 'bilinear'},
    ])
    def test_bilinear_preserves_legacy_seed(self, interaction, rep):
        assert (
            self._experiment(interaction)._compute_data_seed(rep)
            == self._seed_without_interaction_field(rep)
        )

    def test_misspecified_gets_distinct_seed(self):
        legacy = self._seed_without_interaction_field(0)
        euclidean = self._experiment(
            {'_target_': 'bmf.interactions.EuclideanInteraction', 'name': 'euclidean'}
        )._compute_data_seed(0)
        assert euclidean != legacy

    @staticmethod
    def _experiment_with_model(seed_scope, reg_lambda):
        from omegaconf import OmegaConf
        from bmf.config import ExperimentConfig, ModelConfig
        from bmf.experiment import Experiment

        cfg = ExperimentConfig(
            dg=DGConfig(
                user_distribution='g',
                item_ideology_distribution='g',
                item_quality_distribution='g',
            ),
            model=ModelConfig(constraint='none', reg_lambda=reg_lambda),
            seed_scope=seed_scope,
        )
        return Experiment(OmegaConf.structured(cfg))

    def test_full_scope_varies_with_model_config(self):
        a = self._experiment_with_model('full', 0.01)._compute_data_seed(0)
        b = self._experiment_with_model('full', 0.5)._compute_data_seed(0)
        assert a != b

    def test_dg_scope_pairs_mechanism_arms(self):
        a = self._experiment_with_model('dg', 0.01)._compute_data_seed(0)
        b = self._experiment_with_model('dg', 0.5)._compute_data_seed(0)
        assert a == b


class TestMisspecifiedIntegration:
    """Non-bilinear interactions compose with the rest of the generator."""

    @pytest.mark.parametrize("interaction", [
        EuclideanInteraction(),
        HigherOrderInteraction(phi=0.2),
    ])
    def test_misspecified_changes_ratings(self, interaction):
        baseline = DataGenerator(make_dg_config(interaction=None)).generate(seed=3)
        misspec = DataGenerator(
            make_dg_config(interaction=interaction)
        ).generate(seed=3)

        # Same latents (same seed), different ratings
        np.testing.assert_array_equal(
            baseline.users['true_user_ideology'].values,
            misspec.users['true_user_ideology'].values,
        )
        assert not np.allclose(
            baseline.ratings['rating'].values,
            misspec.ratings['rating'].values,
        )

    def test_euclidean_reproduces_its_formula(self):
        data = DataGenerator(
            make_dg_config(interaction=EuclideanInteraction())
        ).generate(seed=5)

        merged = (
            data.ratings
            .merge(data.users, on='user_id')
            .merge(data.items, on='item_id')
        )
        expected = merged['true_item_quality'] - (
            merged['true_user_ideology'] - merged['true_item_ideology']
        ) ** 2
        np.testing.assert_allclose(
            merged['rating_no_noise'].values, expected.values, rtol=1e-10
        )

    def test_extremization_still_applies(self):
        """weight > 0 must move reported ratings away from true utilities."""
        cfg = make_dg_config(
            interaction=EuclideanInteraction(),
            weight=0.5,
            weight_mode='additive',
            extremize='both',
        )
        data = DataGenerator(cfg).generate(seed=13)

        # Extremization affects reported ratings but not underlying utilities
        assert not np.allclose(
            data.ratings['rating_no_noise'].values,
            data.ratings['user_utility'].values,
        )

    def test_partial_rating_selection_still_applies(self):
        """select_opp < 1 must drop some observations via the keep mask."""
        cfg = make_dg_config(
            interaction=EuclideanInteraction(),
            select_opp=0.2,
            select_mode='single',
        )
        data = DataGenerator(cfg).generate(seed=17)

        keep = data.ratings['keep']
        assert keep.any() and not keep.all()

    def test_random_rating_still_applies(self):
        """random_rating limits each user to random_n_items observations."""
        cfg = make_dg_config(
            interaction=EuclideanInteraction(),
            random_rating=True,
            random_n_items=6,
        )
        data = DataGenerator(cfg).generate(seed=19)

        counts = data.ratings.groupby('user_id').size()
        assert (counts == 6).all()

    def test_reproducible_with_same_seed(self):
        cfg = make_dg_config(interaction=EuclideanInteraction())
        first = DataGenerator(cfg).generate(seed=23)
        second = DataGenerator(cfg).generate(seed=23)

        np.testing.assert_array_equal(
            first.ratings['rating'].values,
            second.ratings['rating'].values,
        )


class TestConfounder:
    """Unmeasured confounder: r = q + g(u,v) + w_i*z_j + noise, d=2 truth."""

    def test_disabled_by_default(self):
        """Default config leaves ratings and the RNG stream untouched."""
        baseline = DataGenerator(make_dg_config()).generate(seed=31)
        explicit_off = DataGenerator(
            make_dg_config(confounder_user_std=0.0, confounder_item_std=0.0)
        ).generate(seed=31)

        np.testing.assert_array_equal(
            baseline.ratings['rating'].values,
            explicit_off.ratings['rating'].values,
        )
        # Latents must match too: a disabled confounder must not consume draws
        np.testing.assert_array_equal(
            baseline.users['true_user_ideology'].values,
            explicit_off.users['true_user_ideology'].values,
        )

    def test_zero_columns_when_disabled(self):
        data = DataGenerator(make_dg_config()).generate(seed=31)
        assert (data.users['true_user_confounder'] == 0).all()
        assert (data.items['true_item_confounder'] == 0).all()

    def test_one_sided_std_does_not_enable(self):
        """w*z is identically zero unless both stds are set; treat as disabled."""
        baseline = DataGenerator(make_dg_config()).generate(seed=31)
        one_sided = DataGenerator(
            make_dg_config(confounder_user_std=1.0, confounder_item_std=0.0)
        ).generate(seed=31)
        np.testing.assert_array_equal(
            baseline.ratings['rating'].values,
            one_sided.ratings['rating'].values,
        )

    def test_enabled_changes_ratings(self):
        baseline = DataGenerator(make_dg_config()).generate(seed=31)
        confounded = DataGenerator(
            make_dg_config(confounder_user_std=1.0, confounder_item_std=1.0)
        ).generate(seed=31)
        assert not np.allclose(
            baseline.ratings['rating'].values,
            confounded.ratings['rating'].values,
        )

    def test_reproduces_two_factor_formula(self):
        """rating_no_noise == q_j + u_i*v_j + w_i*z_j exactly."""
        data = DataGenerator(
            make_dg_config(confounder_user_std=1.0, confounder_item_std=1.0)
        ).generate(seed=37)

        merged = (
            data.ratings
            .merge(data.users, on='user_id')
            .merge(data.items, on='item_id')
        )
        expected = (
            merged['true_item_quality']
            + merged['true_user_ideology'] * merged['true_item_ideology']
            + merged['true_user_confounder'] * merged['true_item_confounder']
        )
        np.testing.assert_allclose(
            merged['rating_no_noise'].values, expected.values, rtol=1e-10
        )

    def test_utility_and_rating_share_one_draw(self):
        """The confounder must be sampled once, not redrawn per code path.

        With extremization off, reported ratings equal utilities; two separate
        draws would break that identity.
        """
        data = DataGenerator(
            make_dg_config(confounder_user_std=1.0, confounder_item_std=1.0, weight=0.0)
        ).generate(seed=41)
        np.testing.assert_allclose(
            data.ratings['rating_no_noise'].values,
            data.ratings['user_utility'].values,
            rtol=1e-10,
        )

    def test_composes_with_misspecified_interaction(self):
        """Confounder is additive, so it stacks on any interaction."""
        data = DataGenerator(make_dg_config(
            interaction=EuclideanInteraction(),
            confounder_user_std=1.0,
            confounder_item_std=1.0,
        )).generate(seed=43)

        merged = (
            data.ratings
            .merge(data.users, on='user_id')
            .merge(data.items, on='item_id')
        )
        expected = (
            merged['true_item_quality']
            - (merged['true_user_ideology'] - merged['true_item_ideology']) ** 2
            + merged['true_user_confounder'] * merged['true_item_confounder']
        )
        np.testing.assert_allclose(
            merged['rating_no_noise'].values, expected.values, rtol=1e-10
        )

    def test_extremization_does_not_scale_confounder(self):
        """Exaggeration applies to the ideology channel only."""
        cfg = dict(confounder_user_std=1.0, confounder_item_std=1.0,
                   weight=0.6, weight_mode='multiplicative', extremize='both')
        data = DataGenerator(make_dg_config(**cfg)).generate(seed=47)

        merged = (
            data.ratings
            .merge(data.users, on='user_id')
            .merge(data.items, on='item_id')
        )
        expected = (
            merged['true_item_quality']
            + (1 + merged['agent_weight'])
            * merged['true_user_ideology'] * merged['true_item_ideology']
            + merged['true_user_confounder'] * merged['true_item_confounder']
        )
        np.testing.assert_allclose(
            merged['rating_no_noise'].values, expected.values, rtol=1e-10
        )


class TestConfounderSeedStability:
    """A disabled confounder must not perturb pre-existing seeds."""

    @pytest.mark.parametrize("rep", [0, 3])
    def test_disabled_preserves_legacy_seed(self, rep):
        legacy = TestSeedStability._seed_without_interaction_field(rep)
        assert TestSeedStability._experiment(None)._compute_data_seed(rep) == legacy

    def test_enabled_gets_distinct_seed(self):
        from omegaconf import OmegaConf
        from bmf.config import ExperimentConfig, ModelConfig
        from bmf.experiment import Experiment

        cfg = ExperimentConfig(
            dg=DGConfig(
                user_distribution='g',
                item_ideology_distribution='g',
                item_quality_distribution='g',
                confounder_user_std=1.0,
                confounder_item_std=1.0,
            ),
            model=ModelConfig(constraint='none'),
        )
        seed = Experiment(OmegaConf.structured(cfg))._compute_data_seed(0)
        assert seed != TestSeedStability._seed_without_interaction_field(0)


class TestExperimentNameStability:
    """Inactive dg fields must not change experiment identity.

    The name is a hash of the whole config and drives _already_done and any
    join on experiment_name, so a newly added field left at its disabling
    default must not re-id existing runs.
    """

    @staticmethod
    def _name(**dg_overrides):
        from omegaconf import OmegaConf
        from bmf.config import ExperimentConfig, ModelConfig
        from bmf.experiment import Experiment

        params = dict(
            user_distribution='g',
            item_ideology_distribution='g',
            item_quality_distribution='g',
        )
        params.update(dg_overrides)
        cfg = ExperimentConfig(dg=DGConfig(**params), model=ModelConfig(constraint='none'))
        return Experiment(OmegaConf.structured(cfg))._build_experiment_name()

    def test_disabled_confounder_does_not_change_name(self):
        assert self._name() == self._name(
            confounder_user_std=0.0, confounder_item_std=0.0
        )

    def test_bilinear_interaction_does_not_change_name(self):
        assert self._name() == self._name(
            interaction={'_target_': 'bmf.interactions.BilinearInteraction',
                         'name': 'bilinear'}
        )

    def test_enabled_confounder_changes_name(self):
        assert self._name() != self._name(
            confounder_user_std=1.0, confounder_item_std=1.0
        )

    def test_misspecified_interaction_changes_name(self):
        assert self._name() != self._name(
            interaction={'_target_': 'bmf.interactions.EuclideanInteraction',
                         'name': 'euclidean'}
        )
