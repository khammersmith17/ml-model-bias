import numpy as np

class PreTrainingBias:
    @staticmethod
    def ci(facet_a: np.ndarray, facet_d: np.ndarray) -> float:
        return (facet_a.size - facet_d.size) / (facet_a.size + facet_d.size)

    @staticmethod
    def dpl(facet_a: np.ndarray, facet_d: np.ndarray) -> float:
        q_a = np.sum(facet_a) / facet_a.size
        q_d = np.sum(facet_d) / facet_d.size

        return q_a - q_d

    @staticmethod
    def kll(facet_a: np.ndarray, facet_d: np.ndarray) -> float:
        a_acceptance = np.sum(facet_a) / facet_a.size
        d_acceptance = np.sum(facet_d) / facet_d.size

        return (
            a_acceptance + np.log(a_acceptance/d_acceptance) +
            (1- a_acceptance) * np.log((1 - a_acceptance) / (1- d_acceptance))
        )

    def js(self, facet_a: np.ndarray, facet_d: np.ndarray) -> float:
        p = 1/2 * (np.sum(facet_a) / facet_d.size + np.sum(facet_d) / facet_d.size)

        return 1/2 * (self.kll(facet_a, p) + self.kll(facet_d, p))

    @staticmethod
    def lp_norm(facet_a: np.ndarray, facet_d: np.ndarray) -> float:
        a_acceptance = np.sum(facet_a) / facet_a.size
        d_acceptance = np.sum(facet_d) / facet_d.size

        return np.sqrt(np.power((a_acceptance-d_acceptance), 2) + np.power((1 - a_acceptance - 1 - d_acceptance), 2))

    @staticmethod
    def tvd(facet_a: np.ndarray, facet_d: np.ndarray) -> float:
        a_acceptance = np.sum(facet_a) / facet_a.size
        d_acceptance = np.sum(facet_d) / facet_d.size

        return np.abs(a_acceptance - d_acceptance) + np.abs((1 - a_acceptance) - (1 - d_acceptance))

    @staticmethod
    def ks(facet_a: np.ndarray, facet_d: np.ndarray) -> float:
        a_0_dist = np.sum(np.where(facet_a == 0, 1, 0)) / facet_a.size
        a_1_dist = np.sum(np.where(facet_d == 1, 1, 0)) / facet_a.size

        d_0_dist = np.sum(np.where(facet_d == 0, 1, 0))
        d_1_dist = np.sum(np.where(facet_d == 1, 1, 0))

        return np.max([np.abs(a_0_dist - d_0_dist), np.abs(a_1_dist - d_1_dist)])
