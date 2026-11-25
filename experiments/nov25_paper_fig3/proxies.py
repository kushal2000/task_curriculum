"""Proxies drawn in Fig. 3 (fold_shapes.FoldShape is the same kinematics; these add plates outside the trained pool)."""
import numpy as np

L = 0.05


class Proxy:
    def __init__(self, plates, hinges, new=()):
        self.plates = {k: np.array(v, float) * L for k, v in plates.items()}      # CCW polygons
        self.hinges = {c: (p, np.array(a, float) * L, np.array(b, float) * L) for c, (p, a, b) in hinges.items()}
        self.new = set(new)                                                       # plates the agent added a hinge for
        self.children = list(self.hinges)

    def transforms(self, ang):
        root = next(p for p in self.plates if p not in self.hinges)
        out = {root: (np.eye(3), np.zeros(3))}
        for c in self.children:  # listed parent-first
            par, a, b = self.hinges[c]
            d = (b - a) / np.linalg.norm(b - a); n = self.plates[c].mean(0) - a; n = n - n.dot(d) * d; n /= np.linalg.norm(n)
            u = np.array([n[1], -n[0], 0.0]); p = np.array([*a, 0.0]); th = ang.get(c, 0.0)
            k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            r = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k
            rp, tp = out[par]; out[c] = (rp @ r, rp @ (p - r @ p) + tp)
        return out

    def owners(self, xy):
        own = np.full(len(xy), "", dtype=object)
        for name, poly in self.plates.items():
            e = np.roll(poly, -1, 0) - poly
            ins = np.all([ee[0] * (xy[:, 1] - a[1]) - ee[1] * (xy[:, 0] - a[0]) > -1e-8 for a, ee in zip(poly, e)], axis=0)
            own[(own == "") & ins] = name
        assert (own != "").all()
        return own


W = [[-1, -1], [0, -1], [0, 1], [-1, 1]]
PROXIES = {
    "box2": Proxy({"W": W, "E": [[0, -1], [1, -1], [1, 1], [0, 1]]}, {"E": ("W", (0, -1), (0, 1))}),
    "diag": Proxy({"A": [[-1, -1], [1, 1], [-1, 1]], "B": [[-1, -1], [1, -1], [1, 1]]}, {"B": ("A", (-1, -1), (1, 1))}),
    "strip_flap": Proxy({"W": W, "E": [[0, -1], [.55, -1], [.55, 1], [0, 1]], "F": [[.55, -1], [1, -1], [1, 1], [.55, 1]]},
                        {"E": ("W", (0, -1), (0, 1)), "F": ("E", (.55, -1), (.55, 1))}, new={"F"}),
    "corner_flap": Proxy({"W": W, "E": [[0, -1], [1, -1], [1, .25], [0, 1]], "F": [[1, .25], [1, 1], [0, 1]]},
                         {"E": ("W", (0, -1), (0, 1)), "F": ("E", (1, .25), (0, 1))}, new={"F"}),
}
