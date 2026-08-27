"""Alpha-adaptive NSM wrappers for AnyRank graphical recycling."""

from nscore.nonparametric_nsm import ContinuousNsmTest, MirroredContinuousNsmTest

__all__ = ["ContinuousNsmTestAlphaAdaptive", "MirroredContinuousNsmTestAlphaAdaptive"]


class ContinuousNsmTestAlphaAdaptive(ContinuousNsmTest):
    """Continuous NSM test whose alpha can be recycled by AnyRank.

    Graphical recycling is expected to increase alpha monotonically; lowering
    alpha can retract a prior decision because the decision rule compares the
    running-minimum p-value against the current alpha.
    """

    @ContinuousNsmTest.alpha.setter
    def alpha(self, value):
        self._alpha = value
        self._cutoff = 1.0 / value


class MirroredContinuousNsmTestAlphaAdaptive(MirroredContinuousNsmTest):
    """Mirrored continuous NSM test with alpha-adaptive child tests."""

    _base_class = ContinuousNsmTestAlphaAdaptive
