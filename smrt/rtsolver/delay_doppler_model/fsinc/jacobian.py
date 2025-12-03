import numba
import numpy as np
from scipy.spatial import cKDTree, distance


def jacobian_1d(x):
    w = np.diff(x)
    w = np.append(w, w[-1])
    return w


def jacobian_1d_grad(x):
    w = np.gradient(x)
    return w


def jacobian_2d_ktree(x, y):
    d = np.vstack((x, y)).T
    t = cKDTree(d)

    (w, i) = t.query(d, [2], workers=-1)
    return w.squeeze()


@numba.njit(parallel=True, cache=True)
def jacobian_2d_bf(x, y):
    d = np.vstack((x, y)).T

    w = np.zeros((d.shape[0],))
    for i in numba.prange(d.shape[0]):
        c = d[i, :]
        s = np.sum(np.square(d[:, :] - c), axis=1)
        # print(s)
        w[i] = np.sqrt(np.min(np.delete(s, i)))

    return w


def jacobian_2d_sk(x, y):
    d = np.vstack((x, y)).T
    print("jacobian:", d.shape)
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=2, workers=-1).fit(d)
    w, ii = nbrs.kneighbors(d, 2)

    return w[:, 1].squeeze()


def jacobian_2d_pd(x, y):
    d = np.vstack((x, y)).T
    return distance.pdist(d)


def jacobian_2d_cd(x, y):
    d = np.vstack((x, y)).T
    return distance.cdist(d, d)


def jacobian_2d_es(x, y):
    X = np.vstack((x, y)).T

    XX = np.einsum("ij, ij ->i", X, X)
    D = XX[:, None] + XX - 2 * X.dot(X.T)
    np.einsum("ii->i", D)[:] = np.nan
    D = np.nanmin(D, axis=1)
    return np.sqrt(D)
    # print (D)
    # w = np.zeros((d.shape[0],))


def test_1d_diff():
    x = np.sort(np.random.uniform(0, 5, 50))
    w = jacobian_1d(x)

    xn = x + w
    np.testing.assert_allclose(x[1:], xn[:-1])

    # plt.plot(x, w)
    # plt.plot(x, np.zeros(x.shape), 'x')
    # plt.show()


def test_1d_grad():
    x = np.sort(np.random.uniform(0, 5, 50))
    jacobian_1d(x)
    w = jacobian_1d_grad(x)

    # plt.plot(x, w1, label = 'diff')
    # plt.plot(x, w, label = 'grad')
    # plt.plot(x, np.zeros(x.shape), 'x', label = 'samples')
    # plt.legend()
    # plt.show()

    xn = x + w
    xn
    # np.testing.assert_allclose(x[1:], xn[:-1])


def test_jac_2d_kdtree():
    x = np.random.uniform(0, 5, 1000)
    y = np.random.uniform(0, 5, 4000)

    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()

    w = jacobian_2d_ktree(xx, yy)
    print(w)


def test_jac_2d_ktree_diffs():
    x = [0, 0.5, 7, 8]
    y = [0.1, -4, 7.0, 8]

    w = jacobian_2d_ktree(x, y)
    print(w)

    d = np.vstack((x, y)).T
    expected = distance.cdist(d, d)
    np.fill_diagonal(expected, np.nan)
    print(expected)
    expected = np.nanmin(expected, axis=1)
    print(expected)
    np.testing.assert_equal(expected, w)

    # plt.plot(xx, yy, 'x')
    # plt.show()


# def test_2d_jac_2d_bf():
#   np.random.seed(0)
#   x = np.random.uniform(0, 5, 1000)
#   y = np.random.uniform(0, 5, 4000)

#   xx, yy = np.meshgrid(x,y)
#   xx, yy = xx.ravel(), yy.ravel()

#   w = jacobian_2d_bf(xx, yy)
#   print (w)


def test_2d_jac_2d_es():
    np.random.seed(0)
    x = np.random.uniform(0, 5, 1000)
    y = np.random.uniform(0, 5, 4000)

    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()

    w = jacobian_2d_es(xx, yy)
    print(w)


def test_2d_jac_2d_sk():
    np.random.seed(0)
    x = np.random.uniform(0, 5, 1000)
    y = np.random.uniform(0, 5, 4000)

    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()

    w = jacobian_2d_sk(xx, yy)
    print(w)


def test_2d_jac_2d_pd():
    np.random.seed(0)
    x = np.random.uniform(0, 5, 1000)
    y = np.random.uniform(0, 5, 4000)

    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()

    w = jacobian_2d_pd(xx, yy)
    print(w)


def test_2d_jac_2d_cd():
    np.random.seed(0)
    x = np.random.uniform(0, 5, 1000)
    y = np.random.uniform(0, 5, 4000)

    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()

    w = jacobian_2d_cd(xx, yy)
    print(w)
