#! /usr/bin/env python3

import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def lgwt(nx):
    """Helper for sinc1d, author: Gaute Hope, 2020"""
    xx = np.zeros((nx,))
    ww = np.zeros((nx,))
    for a in numba.prange(nx):
        _, ww[a], xx[a] = glpair(nx, a + 1)

    return xx, ww


@numba.njit(parallel=True, cache=True)
def lgwt2d(nx, ny):
    """Helper for sinc2d, author: Gaute Hope, 2020"""
    xx, wx = lgwt(nx)
    yy, wy = lgwt(ny)

    z = nx * ny
    xxx = np.zeros((z,))
    yyy = np.zeros((z,))
    www = np.zeros((z,))

    for a in numba.prange(ny):
        for b in numba.prange(nx):
            xxx[a * nx + b] = xx[b]
            yyy[a * nx + b] = yy[a]
            www[a * nx + b] = wx[b] * wy[a]

    return xxx, yyy, www


@numba.njit(parallel=True, cache=True)
def lgwt_tri(nx):
    """Helper for sincsq1d, author: Gaute Hope, 2020"""
    xx = np.zeros((2 * nx,))
    ww = np.zeros((2 * nx,))
    for a in numba.prange(nx):
        _, w, x = glpair(nx, a + 1)

        xx[a] = x - 1
        xx[a + nx] = x + 1

        ww[a] = w * (2 - np.abs(x - 1))
        ww[a + nx] = w * (2 - np.abs(x + 1))

    return xx, ww


@numba.njit(parallel=True, cache=True)
def lgwt_tri_2d(nx, ny):
    """Helper for sincsq2d, author: Gaute Hope, 2020"""
    xx, wwx = lgwt_tri(nx)
    yy, wwy = lgwt_tri(ny)

    z = 2 * nx * 2 * ny
    xxx = np.zeros((z,))
    yyy = np.zeros((z,))
    www = np.zeros((z,))

    for a in numba.prange(2 * ny):
        for b in numba.prange(2 * nx):
            xxx[a * 2 * nx + b] = xx[b]
            yyy[a * 2 * nx + b] = yy[a]
            www[a * 2 * nx + b] = wwx[b] * wwy[a]

    return xxx, yyy, www


@numba.njit(cache=True)
def besselj1squared(k):
    # *****************************************************************************80
    #
    ## BESSELJ1SQUARED computes the square of BesselJ(1, BesselZero(0,k))
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    #  Parameters:
    #
    #    Input, integer K, the index of the desired zero.
    #
    #    Output, real Z, the value of the square of the Bessel
    #    J1 function at the K-th zero of the Bessel J0 function.
    #

    j1 = np.array(
        [
            0.269514123941916926139021992911e00,
            0.115780138582203695807812836182e00,
            0.0736863511364082151406476811985e00,
            0.0540375731981162820417749182758e00,
            0.0426614290172430912655106063495e00,
            0.0352421034909961013587473033648e00,
            0.0300210701030546726750888157688e00,
            0.0261473914953080885904584675399e00,
            0.0231591218246913922652676382178e00,
            0.0207838291222678576039808057297e00,
            0.0188504506693176678161056800214e00,
            0.0172461575696650082995240053542e00,
            0.0158935181059235978027065594287e00,
            0.0147376260964721895895742982592e00,
            0.0137384651453871179182880484134e00,
            0.0128661817376151328791406637228e00,
            0.0120980515486267975471075438497e00,
            0.0114164712244916085168627222986e00,
            0.0108075927911802040115547286830e00,
            0.0102603729262807628110423992790e00,
            0.00976589713979105054059846736696e00,
        ]
    )

    if 21 < k:
        x = 1.0e00 / (k - 0.25e00)
        x2 = x * x
        z = x * (
            0.202642367284675542887758926420e00
            + x2
            * x2
            * (
                -0.303380429711290253026202643516e-03
                + x2
                * (
                    0.198924364245969295201137972743e-03
                    + x2
                    * (
                        -0.228969902772111653038747229723e-03
                        + x2
                        * (
                            0.433710719130746277915572905025e-03
                            + x2
                            * (
                                -0.123632349727175414724737657367e-02
                                + x2
                                * (
                                    0.496101423268883102872271417616e-02
                                    + x2
                                    * (
                                        -0.266837393702323757700998557826e-01
                                        + x2 * (0.185395398206345628711318848386e00)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    else:
        z = j1[k - 1]

    return z


def besselj1squared_test():
    # *****************************************************************************80
    #
    ## BESSELJ1SQUARED_TEST tests BESSELJ1SQUARED.
    #
    #  Discussion:
    #
    #    SCIPY.SPECIAL provides the built in function J1.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    import platform

    import scipy.special as sp

    print("")
    print("BESSELJ1SQUARED_TEST:")
    print("  Python version: %s" % (platform.python_version()))
    print("  BESSELJ1SQUARED returns the square of the Bessel J1(X) function")
    print("  at the K-th zero of J0(X).")
    print("")
    print("   K           X(K)                    J1(X(K))^2                 BESSELJ1SQUARED")
    print("")

    for k in range(1, 31):
        x = besseljzero(k)
        f1 = sp.j1(x) ** 2
        f2 = besselj1squared(k)
        print("  %2d  %24.16g  %24.16g  %24.16g" % (k, x, f1, f2))
    #
    #  Terminate.
    #
    print("")
    print("BESSELJ1SQUARED_TEST:")
    print("  Normal end of execution.")
    return


@numba.njit(cache=True)
def besseljzero(k):
    # *****************************************************************************80
    #
    ## BESSELJZERO computes the kth zero of the J0(X) Bessel function.
    #
    #  Discussion:
    #
    #    Note that the first 20 zeros are tabulated.  After that, they are
    #    computed.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    #  Parameters:
    #
    #    Input, integer K, the index of the desired zero.
    #    1 <= K.
    #
    #    Output, real X, the value of the zero.
    #
    # import numpy as np

    jz = np.array(
        [
            2.40482555769577276862163187933e00,
            5.52007811028631064959660411281e00,
            8.65372791291101221695419871266e00,
            11.7915344390142816137430449119e00,
            14.9309177084877859477625939974e00,
            18.0710639679109225431478829756e00,
            21.2116366298792589590783933505e00,
            24.3524715307493027370579447632e00,
            27.4934791320402547958772882346e00,
            30.6346064684319751175495789269e00,
            33.7758202135735686842385463467e00,
            36.9170983536640439797694930633e00,
            40.0584257646282392947993073740e00,
            43.1997917131767303575240727287e00,
            46.3411883716618140186857888791e00,
            49.4826098973978171736027615332e00,
            52.6240518411149960292512853804e00,
            55.7655107550199793116834927735e00,
            58.9069839260809421328344066346e00,
            62.0484691902271698828525002646e00,
        ]
    )

    if 20 < k:
        x = np.pi * (k - 0.25e00)
        r = 1.0e00 / x
        r2 = r * r
        x = x + r * (
            0.125e00
            + r2
            * (
                -0.807291666666666666666666666667e-01
                + r2
                * (
                    0.246028645833333333333333333333e00
                    + r2
                    * (
                        -0.182443876720610119047619047619e01
                        + r2
                        * (
                            0.253364147973439050099206349206e02
                            + r2
                            * (
                                -0.567644412135183381139802038240e03
                                + r2
                                * (
                                    0.186904765282320653831636345064e05
                                    + r2
                                    * (-0.849353580299148769921876983660e06 + r2 * 0.509225462402226769498681286758e08)
                                )
                            )
                        )
                    )
                )
            )
        )
    else:
        x = jz[k - 1]

    return x


def besseljzero_test():
    # *****************************************************************************80
    #
    ## BESSELJZERO_TEST tests BESSELJZERO.
    #
    #  Discussion:
    #
    #    SCIPY.SPECIAL provides the built in J0(X) function.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    import platform

    import scipy.special as sp

    print("")
    print("BESSELJZERO_TEST:")
    print("  Python version: %s" % (platform.python_version()))
    print("  BESSELJZERO returns the K-th zero of J0(X).")
    print("")
    print("   K           X(K)                  J0(X(K))")
    print("")

    for k in range(1, 31):
        x = besseljzero(k)
        j0x = sp.j0(x)
        print("  %2d  %24.16g  %24.16g" % (k, x, j0x))
    #
    #  Terminate.
    #
    print("")
    print("BESSELJZERO_TEST:")
    print("  Normal end of execution.")
    return


def fastgl_test():
    # *****************************************************************************80
    #
    ## FASTGL_TEST tests the FASTGL library.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    print("")
    print("FASTGL_TEST")
    print("  Python version: %s" % (platform.python_version()))
    print("  Test the FASTGL library.")

    besseljzero_test()
    besselj1squared_test()
    glpair_test()
    glpairs_test()
    glpairtabulated_test()
    legendre_theta_test()
    legendre_weight_test()
    #
    #  Terminate.
    #
    print("")
    print("FASTGL_TEST:")
    print("  Normal end of execution.")
    return


@numba.njit(cache=True)
def glpair(n, k):
    # *****************************************************************************80
    #
    ## GLPAIR computes the K-th pair of an N-point Gauss-Legendre rule.
    #
    #  Discussion:
    #
    #    If N <= 100, GLPAIRTABULATED is called, otherwise GLPAIR is called.
    #
    #    Theta values of the zeros are in [0,pi], and monotonically increasing.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    #  Parameters:
    #
    #    Input, integer N, the number of points in the given rule.
    #    0 < N.
    #
    #    Input, integer K, the index of the point to be returned.
    #    1 <= K <= N.
    #
    #    Output, real THETA, WEIGHT, X, the theta coordinate, weight,
    #    and x coordinate of the point.
    #
    # from sys import exit

    if n < 1:
        print("")
        print("GLPAIR - Fatal error!")
        print("  Illegal value of N.")
        # return -1

    if k < 1 or n < k:
        print("")
        print("GLPAIR - Fatal error!")
        print("  Illegal value of K.")
        # exit ( 'GLPAIR - Fatal error!' )
        # return -1

    if n < 101:
        theta, weight, x = glpairtabulated(n, k)
    else:
        theta, weight, x = glpairs(n, k)

    return theta, weight, x


def glpair_test():
    # *****************************************************************************80
    #
    ## GLPAIR_TEST tests GLPAIR.
    #
    #  Discussion:
    #
    #    Test the numerical integration of ln(x) over the range [0,1]
    #    Normally, one would not use Gauss-Legendre quadrature for this,
    #    but for the sake of having an example with l > 100, this is included.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    import platform

    import numpy as np

    print("")
    print("GLPAIR_TEST")
    print("  Python version: %s" % (platform.python_version()))
    print("  Estimate integral ( 0 <= x <= 1 ) ln(x) dx.")
    print("")
    print("    Nodes           Estimate")
    print("")

    l = 1
    for p in range(0, 7):
        q = 0.0
        for k in range(1, l + 1):
            theta, weight, x = glpair(l, k)
            q = q + 0.5 * weight * np.log(0.5 * (x + 1.0))
        print("  %7d       %24.16g" % (l, q))
        l = l * 10
    print("")
    print("    Exact        -1.0")
    #
    #  Terminate.
    #
    print("")
    print("GLPAIR_TEST:")
    print("  Normal end of execution.")
    return


@numba.njit(cache=True)
def glpairs(n, k):
    # *****************************************************************************80
    #
    ## GLPAIRS computes the K-th pair of an N-point Gauss-Legendre rule.
    #
    #  Discussion:
    #
    #    This routine is intended for cases were 100 < N.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    #  Parameters:
    #
    #    Input, integer N, the number of points in the given rule.
    #    1 <= N.
    #
    #    Input, integer K, the index of the point to be returned.
    #    1 <= K <= N.
    #
    #    Output, real THETA, WEIGHT, X, the theta coordinate, weight,
    #    and x coordinate of the point.
    #
    # import numpy as np
    # from sys import exit

    if n < 1:
        print("")
        print("GLPAIRS - Fatal error!")
        print("  Illegal value of N.")
        # exit ( 'GLPAIRS - Fatal error!' )

    if k < 1 or n < k:
        print("")
        print("GLPAIRS - Fatal error!")
        print("  Illegal value of K.")
        # exit ( 'GLPAIRS - Fatal error!' )

    if n < (2 * k - 1):
        kcopy = n - k + 1
    else:
        kcopy = k
    #
    #  Get the Bessel zero.
    #
    w = 1.0e00 / (float(n) + 0.5e00)
    nu = besseljzero(kcopy)
    theta = w * nu
    y = theta**2
    #
    #  Get the asymptotic BesselJ(1,nu) squared.
    #
    b = besselj1squared(kcopy)
    #
    #  Get the Chebyshev interpolants for the nodes.
    #
    sf1t = (
        (
            (
                (
                    (-1.29052996274280508473467968379e-12 * y + 2.40724685864330121825976175184e-10) * y
                    - 3.13148654635992041468855740012e-08
                )
                * y
                + 0.275573168962061235623801563453e-05
            )
            * y
            - 0.148809523713909147898955880165e-03
        )
        * y
        + 0.416666666665193394525296923981e-02
    ) * y - 0.416666666666662959639712457549e-01

    sf2t = (
        (
            (
                (
                    (+2.20639421781871003734786884322e-09 * y - 7.53036771373769326811030753538e-08) * y
                    + 0.161969259453836261731700382098e-05
                )
                * y
                - 0.253300326008232025914059965302e-04
            )
            * y
            + 0.282116886057560434805998583817e-03
        )
        * y
        - 0.209022248387852902722635654229e-02
    ) * y + 0.815972221772932265640401128517e-02

    sf3t = (
        (
            (
                (
                    (-2.97058225375526229899781956673e-08 * y + 5.55845330223796209655886325712e-07) * y
                    - 0.567797841356833081642185432056e-05
                )
                * y
                + 0.418498100329504574443885193835e-04
            )
            * y
            - 0.251395293283965914823026348764e-03
        )
        * y
        + 0.128654198542845137196151147483e-02
    ) * y - 0.416012165620204364833694266818e-02
    #
    #  Get the Chebyshev interpolants for the weights.
    #
    wsf1t = (
        (
            (
                (
                    (
                        (
                            (
                                (-2.20902861044616638398573427475e-14 * y + 2.30365726860377376873232578871e-12) * y
                                - 1.75257700735423807659851042318e-10
                            )
                            * y
                            + 1.03756066927916795821098009353e-08
                        )
                        * y
                        - 4.63968647553221331251529631098e-07
                    )
                    * y
                    + 0.149644593625028648361395938176e-04
                )
                * y
                - 0.326278659594412170300449074873e-03
            )
            * y
            + 0.436507936507598105249726413120e-02
        )
        * y
        - 0.305555555555553028279487898503e-01
    ) * y + 0.833333333333333302184063103900e-01

    wsf2t = (
        (
            (
                (
                    (
                        (
                            (+3.63117412152654783455929483029e-12 * y + 7.67643545069893130779501844323e-11) * y
                            - 7.12912857233642220650643150625e-09
                        )
                        * y
                        + 2.11483880685947151466370130277e-07
                    )
                    * y
                    - 0.381817918680045468483009307090e-05
                )
                * y
                + 0.465969530694968391417927388162e-04
            )
            * y
            - 0.407297185611335764191683161117e-03
        )
        * y
        + 0.268959435694729660779984493795e-02
    ) * y - 0.111111111111214923138249347172e-01

    wsf3t = (
        (
            (
                (
                    (
                        (
                            (+2.01826791256703301806643264922e-09 * y - 4.38647122520206649251063212545e-08) * y
                            + 5.08898347288671653137451093208e-07
                        )
                        * y
                        - 0.397933316519135275712977531366e-05
                    )
                    * y
                    + 0.200559326396458326778521795392e-04
                )
                * y
                - 0.422888059282921161626339411388e-04
            )
            * y
            - 0.105646050254076140548678457002e-03
        )
        * y
        - 0.947969308958577323145923317955e-04
    ) * y + 0.656966489926484797412985260842e-02
    #
    #  Refine with the paper expansions.
    #
    nuosin = nu / np.sin(theta)
    bnuosin = b * nuosin
    winvsinc = w * w * nuosin
    wis2 = winvsinc * winvsinc
    #
    #  Finally compute the node and the weight.
    #
    theta = w * (nu + theta * winvsinc * (sf1t + wis2 * (sf2t + wis2 * sf3t)))
    deno = bnuosin + bnuosin * wis2 * (wsf1t + wis2 * (wsf2t + wis2 * wsf3t))
    weight = (2.0e00 * w) / deno

    if n < (2 * k - 1):
        theta = np.pi - theta

    x = np.cos(theta)

    return theta, weight, x


def glpairs_test():
    # *****************************************************************************80
    #
    ## GLPAIRS_TEST tests GLPAIRS.
    #
    #  Discussion:
    #
    #    Test the numerical integration of cos(1000 x) over the range [-1,1]
    #    for varying number of Gauss-Legendre quadrature nodes l.
    #    The fact that only twelve digits of accuracy are obtained is due to the
    #    condition number of the summation.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    import platform

    import numpy as np

    print("")
    print("GLPAIRS_TEST:")
    print("  Python version: %s" % (platform.python_version()))
    print("  integral ( -1 <= x <= 1 ) cos(1000 x) dx")
    print("")
    print("    Nodes           Estimate")
    print("")

    for l in range(500, 620, 20):
        q = 0.0

        for k in range(1, l + 1):
            theta, weight, x = glpairs(l, k)
            q = q + weight * np.cos(1000.0 * x)

        print("  %7d  %24.16g" % (l, q))

    print("")
    print("    Exact  %24.16g" % (0.002 * np.sin(1000.0)))
    #
    #  Terminate.
    #
    print("")
    print("GLPAIRS_TEST:")
    print("  Normal end of execution.")
    return


@numba.njit(cache=True)
def glpairtabulated(l, k):
    # *****************************************************************************80
    #
    ## GLPAIRTABULATED computes the K-th pair of an N-point Gauss-Legendre rule.
    #
    #  Discussion:
    #
    #    Data is tabulated for 1 <= L <= 100.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    #  Parameters:
    #
    #    Input, integer L, the number of points in the given rule.
    #    1 <= L <= 100.
    #
    #    Input, integer K, the index of the point to be returned.
    #    1 <= K <= L.
    #
    #    Output, real THETA, WEIGHT, X, the theta coordinate, weight,
    #    and x coordinate of the point.
    #

    if l < 1 or 100 < l:
        print("")
        print("GLPAIRTABULATED - Fatal error!")
        print("  Illegal value of L.")
        # exit ( 'GLPAIRTABULATED - Fatal error!' )
        # return -1

    if k < 1 or l < k:
        print("")
        print("GLPAIRTABULATED - Fatal error!")
        print("  Illegal value of K.")
        # exit ( 'GLPAIRTABULATED - Fatal error!' )
        # return -1

    theta = legendre_theta(l, k)
    weight = legendre_weight(l, k)

    x = np.cos(theta)

    return theta, weight, x


def glpairtabulated_test():
    # *****************************************************************************80
    #
    ## GLPAIRTABULATED_TEST tests GLPAIRTABULATED.
    #
    #  Discussion:
    #
    #    Test the numerical integration of exp(x) over the range [-1,1]
    #    for varying number of Gauss-Legendre quadrature nodes l.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    import platform

    import numpy as np

    print("")
    print("GLPAIRTABULATED_TEST:")
    print("  Python version: %s" % (platform.python_version()))
    print("  integral ( -1 <= x <= 1 ) exp(x) dx")
    print("")
    print("    Nodes           Estimate")
    print("")

    for l in range(1, 10):
        q = 0.0
        for k in range(1, l + 1):
            theta, weight, x = glpairtabulated(l, k)
            q = q + weight * np.exp(x)
        print("  %7d  %24.16g" % (l, q))

    print("")
    print("    Exact  %24.16g" % (np.exp(1.0e00) - np.exp(-1.0e00)))
    #
    #  Terminate.
    #
    print("")
    print("GLPAIRTABULATED_TEST:")
    print("  Normal end of execution.")
    return


@numba.njit(cache=True)
def legendre_theta(l, k):
    # *****************************************************************************80
    #
    ## LEGENDRE_THETA returns the K-th theta coordinate in an L point rule.
    #
    #  Discussion:
    #
    #    The X coordinate is simply cos ( THETA ).
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    #  Parameters:
    #
    #    Input, integer L, the number of points in the given rule.
    #    1 <= L.
    #
    #    Input, integer K, the index of the point to be returned.
    #    1 <= K <= L.
    #
    #    Output, real THETA, the theta coordinate of the point.
    #

    EvenThetaZero1 = np.array([0.9553166181245092781638573e00])
    EvenThetaZero2 = np.array([0.1223899586470372591854100e01, 0.5332956802491269896325121e00])
    EvenThetaZero3 = np.array(
        [0.1329852612388110166006182e01, 0.8483666264874876548310910e00, 0.3696066519448289481138796e00]
    )
    EvenThetaZero4 = np.array(
        [
            0.1386317078892131346282665e01,
            0.1017455539490153431016397e01,
            0.6490365804607796100719162e00,
            0.2827570635937967783987981e00,
        ]
    )
    EvenThetaZero5 = np.array(
        [
            0.1421366498439524924081833e01,
            0.1122539327631709474018620e01,
            0.8238386589997556048023640e00,
            0.5255196555285001171749362e00,
            0.2289442988470260178701589e00,
        ]
    )
    EvenThetaZero6 = np.array(
        [
            0.1445233238471440081118261e01,
            0.1194120375947706635968399e01,
            0.9430552870605735796668951e00,
            0.6921076988818410126677201e00,
            0.4414870814893317611922530e00,
            0.1923346793046672033050762e00,
        ]
    )
    EvenThetaZero7 = np.array(
        [
            0.1462529992921481833498746e01,
            0.1246003586776677662375070e01,
            0.1029498592525136749641068e01,
            0.8130407055389454598609888e00,
            0.5966877608172733931509619e00,
            0.3806189306666775272453522e00,
            0.1658171411523664030454318e00,
        ]
    )
    EvenThetaZero8 = np.array(
        [
            0.1475640280808194256470687e01,
            0.1285331444322965257106517e01,
            0.1095033401803444343034890e01,
            0.9047575323895165085030778e00,
            0.7145252532340252146626998e00,
            0.5243866409035941583262629e00,
            0.3344986386876292124968005e00,
            0.1457246820036738335698855e00,
        ]
    )
    EvenThetaZero9 = np.array(
        [
            0.1485919440392653014379727e01,
            0.1316167494718022699851110e01,
            0.1146421481056642228295923e01,
            0.9766871104439832694094465e00,
            0.8069738930788195349918620e00,
            0.6373005058706191519531139e00,
            0.4677113145328286263205134e00,
            0.2983460782092324727528346e00,
            0.1299747364196768405406564e00,
        ]
    )
    EvenThetaZero10 = np.array(
        [
            0.1494194914310399553510039e01,
            0.1340993178589955138305166e01,
            0.1187794926634098887711586e01,
            0.1034603297590104231043189e01,
            0.8814230742890135843662021e00,
            0.7282625848696072912405713e00,
            0.5751385026314284688366450e00,
            0.4220907301111166004529037e00,
            0.2692452880289302424376614e00,
            0.1172969277059561308491253e00,
        ]
    )
    EvenThetaZero11 = np.array(
        [
            0.1501000399130816063282492e01,
            0.1361409225664372117193308e01,
            0.1221820208990359866348145e01,
            0.1082235198111836788818162e01,
            0.9426568273796608630446470e00,
            0.8030892957063359443650460e00,
            0.6635400754448062852164288e00,
            0.5240242709487281141128643e00,
            0.3845781703583910933413978e00,
            0.2453165389983612942439953e00,
            0.1068723357985259945018899e00,
        ]
    )
    EvenThetaZero12 = np.array(
        [
            0.1506695545558101030878728e01,
            0.1378494427506219143960887e01,
            0.1250294703417272987066314e01,
            0.1122097523267250692925104e01,
            0.9939044422989454674968570e00,
            0.8657177770401081355080608e00,
            0.7375413075437535618804594e00,
            0.6093818382449565759195927e00,
            0.4812531951313686873528891e00,
            0.3531886675690780704072227e00,
            0.2252936226353075734690198e00,
            0.9814932949793685067733311e-01,
        ]
    )
    EvenThetaZero13 = np.array(
        [
            0.1511531546703289231944719e01,
            0.1393002286179807923400254e01,
            0.1274473959424494104852958e01,
            0.1155947313793812040125722e01,
            0.1037423319077439147088755e01,
            0.9189033445598992550553862e00,
            0.8003894803353296871788647e00,
            0.6818851814129298518332401e00,
            0.5633967073169293284500428e00,
            0.4449368152119130526034289e00,
            0.3265362611165358134766736e00,
            0.2082924425598466358987549e00,
            0.9074274842993199730441784e-01,
        ]
    )
    EvenThetaZero14 = np.array(
        [
            0.1515689149557281132993364e01,
            0.1405475003062348722192382e01,
            0.1295261501292316172835393e01,
            0.1185049147889021579229406e01,
            0.1074838574917869281769567e01,
            0.9646306371285440922680794e00,
            0.8544265718392254369377945e00,
            0.7442282945111358297916378e00,
            0.6340389954584301734412433e00,
            0.5238644768825679339859620e00,
            0.4137165857369637683488098e00,
            0.3036239070914333637971179e00,
            0.1936769929947376175341314e00,
            0.8437551461511597225722252e-01,
        ]
    )
    EvenThetaZero15 = np.array(
        [
            0.1519301729274526620713294e01,
            0.1416312682230741743401738e01,
            0.1313324092045794720169874e01,
            0.1210336308624476413072722e01,
            0.1107349759228459143499061e01,
            0.1004365001539081003659288e01,
            0.9013828087667156388167226e00,
            0.7984043170121235411718744e00,
            0.6954313000299367256853883e00,
            0.5924667257887385542924194e00,
            0.4895160050896970092628705e00,
            0.3865901987860504829542802e00,
            0.2837160095793466884313556e00,
            0.1809780449917272162574031e00,
            0.7884320726554945051322849e-01,
        ]
    )
    EvenThetaZero16 = np.array(
        [
            0.1522469852641529230282387e01,
            0.1425817011963825344615095e01,
            0.1329164502391080681347666e01,
            0.1232512573416362994802398e01,
            0.1135861522840293704616614e01,
            0.1039211728068951568003361e01,
            0.9425636940046777101926515e00,
            0.8459181315837993237739032e00,
            0.7492760951181414487254243e00,
            0.6526392394594561548023681e00,
            0.5560103418005302722406995e00,
            0.4593944730762095704649700e00,
            0.3628020075350028174968692e00,
            0.2662579994723859636910796e00,
            0.1698418454282150179319973e00,
            0.7399171309970959768773072e-01,
        ]
    )
    EvenThetaZero17 = np.array(
        [
            0.1525270780617194430047563e01,
            0.1434219768045409606267345e01,
            0.1343169000217435981125683e01,
            0.1252118659062444379491066e01,
            0.1161068957629157748792749e01,
            0.1070020159291475075961444e01,
            0.9789726059789103169325141e00,
            0.8879267623988119819560021e00,
            0.7968832893748414870413015e00,
            0.7058431727509840105946884e00,
            0.6148079652926100198490992e00,
            0.5237802779694730663856110e00,
            0.4327648832448234459097574e00,
            0.3417715500266717765568488e00,
            0.2508238767288223767569849e00,
            0.1599966542668327644694431e00,
            0.6970264809814094464033170e-01,
        ]
    )
    EvenThetaZero18 = np.array(
        [
            0.1527764849261740485876940e01,
            0.1441701954349064743573367e01,
            0.1355639243522655042028688e01,
            0.1269576852063768424508476e01,
            0.1183514935851550608323947e01,
            0.1097453683555812711123880e01,
            0.1011393333949027021740881e01,
            0.9253342019812867059380523e00,
            0.8392767201322475821509486e00,
            0.7532215073977623159515351e00,
            0.6671694908788198522546767e00,
            0.5811221342350705406265672e00,
            0.4950819018993074588093747e00,
            0.4090533017972007314666814e00,
            0.3230455648729987995657071e00,
            0.2370809940997936908335290e00,
            0.1512302802537625099602687e00,
            0.6588357082399222649528476e-01,
        ]
    )
    EvenThetaZero19 = np.array(
        [
            0.1529999863223206659623262e01,
            0.1448406982124841835685420e01,
            0.1366814241651488684482888e01,
            0.1285221744143731581870833e01,
            0.1203629605904952775544878e01,
            0.1122037965173751996510051e01,
            0.1040446993107623345153211e01,
            0.9588569097730895525404200e00,
            0.8772680085516152329147030e00,
            0.7956806951062012653043722e00,
            0.7140955526031660805347356e00,
            0.6325134568448222221560326e00,
            0.5509357927460004487348532e00,
            0.4693648943475422765864580e00,
            0.3878050333015201414955289e00,
            0.3062649591511896679168503e00,
            0.2247658146033686460963295e00,
            0.1433746167818849555570557e00,
            0.6246124541276674097388211e-01,
        ]
    )
    EvenThetaZero20 = np.array(
        [
            0.1532014188279762793560699e01,
            0.1454449946977268522285131e01,
            0.1376885814601482670609845e01,
            0.1299321869764876494939757e01,
            0.1221758200747475205847413e01,
            0.1144194910846247537582396e01,
            0.1066632125552939823863593e01,
            0.9890700026972186303565530e00,
            0.9115087474225932692070479e00,
            0.8339486352158799520695092e00,
            0.7563900488174808348719219e00,
            0.6788335401193977027577509e00,
            0.6012799395312684623216685e00,
            0.5237305617022755897200291e00,
            0.4461876237541810478131970e00,
            0.3686551849119556335824055e00,
            0.2911415613085158758589405e00,
            0.2136668503694680525340165e00,
            0.1362947587312224822844743e00,
            0.5937690028966411906487257e-01,
        ]
    )
    EvenThetaZero21 = np.array(
        [
            0.1533838971193864306068338e01,
            0.1459924288056445029654271e01,
            0.1386009690354996919044862e01,
            0.1312095239305276612560739e01,
            0.1238181002944535867235042e01,
            0.1164267059803796726229370e01,
            0.1090353503721897748980095e01,
            0.1016440450472067349837507e01,
            0.9425280472651469176638349e00,
            0.8686164868955467866176243e00,
            0.7947060295895204342519786e00,
            0.7207970381018823842440224e00,
            0.6468900366403721167107352e00,
            0.5729858150363658839291287e00,
            0.4990856247464946058899833e00,
            0.4251915773724379089467945e00,
            0.3513075400485981451355368e00,
            0.2774414365914335857735201e00,
            0.2036124177925793565507033e00,
            0.1298811916061515892914930e00,
            0.5658282534660210272754152e-01,
        ]
    )
    EvenThetaZero22 = np.array(
        [
            0.1535499761264077326499892e01,
            0.1464906652494521470377318e01,
            0.1394313611500109323616335e01,
            0.1323720686538524176057236e01,
            0.1253127930763390908996314e01,
            0.1182535404796980113294400e01,
            0.1111943180033868679273393e01,
            0.1041351343083674290731439e01,
            0.9707600019805773720746280e00,
            0.9001692951667510715040632e00,
            0.8295794049297955988640329e00,
            0.7589905782114329186155957e00,
            0.6884031600807736268672129e00,
            0.6178176499732537480601935e00,
            0.5472348011493452159473826e00,
            0.4766558078624760377875119e00,
            0.4060826859477620301047824e00,
            0.3355191279517093844978473e00,
            0.2649727008485465487101933e00,
            0.1944616940738156405895778e00,
            0.1240440866043499301839465e00,
            0.5403988657613871827831605e-01,
        ]
    )
    EvenThetaZero23 = np.array(
        [
            0.1537017713608809830855653e01,
            0.1469460505124226636602925e01,
            0.1401903350962364703169699e01,
            0.1334346289590505369693957e01,
            0.1266789363044399933941254e01,
            0.1199232618763735058848455e01,
            0.1131676111906105521856066e01,
            0.1064119908394702657537061e01,
            0.9965640890815034701957497e00,
            0.9290087556203499065939494e00,
            0.8614540390091103102510609e00,
            0.7939001124053586164046432e00,
            0.7263472110048245091518914e00,
            0.6587956640463586742461796e00,
            0.5912459486086227271608064e00,
            0.5236987847717837556177452e00,
            0.4561553147193391989386660e00,
            0.3886174669444433167860783e00,
            0.3210887745896478259115420e00,
            0.2535764786314617292100029e00,
            0.1860980813776342452540915e00,
            0.1187090676924131329841811e00,
            0.5171568198966901682810573e-01,
        ]
    )
    EvenThetaZero24 = np.array(
        [
            0.1538410494858190444190279e01,
            0.1473638845472165977392911e01,
            0.1408867240039222913928858e01,
            0.1344095709533508756473909e01,
            0.1279324287566779722061664e01,
            0.1214553011719528935181709e01,
            0.1149781925191718586091000e01,
            0.1085011078936665906275419e01,
            0.1020240534516704208782618e01,
            0.9554703680422404498752066e00,
            0.8907006757608306209160649e00,
            0.8259315822134856671969566e00,
            0.7611632524946588128425351e00,
            0.6963959112887657683892237e00,
            0.6316298735371143844913976e00,
            0.5668655960010826255149266e00,
            0.5021037684870694065589284e00,
            0.4373454855522296089897130e00,
            0.3725925956833896735786860e00,
            0.3078484858841616878136371e00,
            0.2431200981264999375962973e00,
            0.1784242126043536701754986e00,
            0.1138140258514833068653307e00,
            0.4958315373802413441075340e-01,
        ]
    )
    EvenThetaZero25 = np.array(
        [
            0.1539692973716708504412697e01,
            0.1477486279394502338589519e01,
            0.1415279620944410339318226e01,
            0.1353073023537942666830874e01,
            0.1290866514321280958405103e01,
            0.1228660123395079609266898e01,
            0.1166453885011658611362850e01,
            0.1104247839096738022319035e01,
            0.1042042033248543055386770e01,
            0.9798365254403234947595400e00,
            0.9176313877712591840677176e00,
            0.8554267118081827231209625e00,
            0.7932226163976800550406599e00,
            0.7310192594231560707888939e00,
            0.6688168560730805146438886e00,
            0.6066157082814543103941755e00,
            0.5444162542389049922529553e00,
            0.4822191559963931133878621e00,
            0.4200254643636986308379697e00,
            0.3578369542536859435571624e00,
            0.2956568781922605524959448e00,
            0.2334919029083292837123583e00,
            0.1713581437497397360313735e00,
            0.1093066902335822942650053e00,
            0.4761952998197036029817629e-01,
        ]
    )
    EvenThetaZero26 = np.array(
        [
            0.1540877753740080417345045e01,
            0.1481040617373741365390254e01,
            0.1421203510518656600018143e01,
            0.1361366453804322852131292e01,
            0.1301529469356044341206877e01,
            0.1241692581525935716830402e01,
            0.1181855817774264617619371e01,
            0.1122019209772750368801179e01,
            0.1062182794829879659341536e01,
            0.1002346617783007482854908e01,
            0.9425107335729934538419206e00,
            0.8826752108319277463183701e00,
            0.8228401370047382776784725e00,
            0.7630056258499810562932058e00,
            0.7031718287376427885875898e00,
            0.6433389522119553277924537e00,
            0.5835072863023426715977658e00,
            0.5236772521416453354847559e00,
            0.4638494862268433259444639e00,
            0.4040249990308909882616381e00,
            0.3442054975680110060507306e00,
            0.2843941101955779333389742e00,
            0.2245972494281051799602510e00,
            0.1648304164747050021714385e00,
            0.1051427544146599992432949e00,
            0.4580550859172367960799915e-01,
        ]
    )
    EvenThetaZero27 = np.array(
        [
            0.1541975588842621898865181e01,
            0.1484334121018556567335167e01,
            0.1426692677652358867201800e01,
            0.1369051275783071487471360e01,
            0.1311409933595114953831618e01,
            0.1253768670970438091691833e01,
            0.1196127510146226323327062e01,
            0.1138486476526912406867032e01,
            0.1080845599717322003702293e01,
            0.1023204914871722785830020e01,
            0.9655644644970043364617272e00,
            0.9079243009168822510582606e00,
            0.8502844897148263889326479e00,
            0.7926451146568312828354346e00,
            0.7350062849078710810840430e00,
            0.6773681459074923011631400e00,
            0.6197308962817025162722438e00,
            0.5620948151095422609589585e00,
            0.5044603077892199488064657e00,
            0.4468279872027509013135997e00,
            0.3891988265038338944044115e00,
            0.3315744698431505326770711e00,
            0.2739579305700525818998611e00,
            0.2163553856859193758294342e00,
            0.1587817673749480300092784e00,
            0.1012844151694839452028589e00,
            0.4412462056235422293371300e-01,
        ]
    )
    EvenThetaZero28 = np.array(
        [
            0.1542995710582548837472073e01,
            0.1487394484904746766220933e01,
            0.1431793279635669382208875e01,
            0.1376192108950239363921811e01,
            0.1320590987909222553912422e01,
            0.1264989932881031519687125e01,
            0.1209388962038683919740547e01,
            0.1153788095965648154683658e01,
            0.1098187358416032947576489e01,
            0.1042586777292402877200408e01,
            0.9869863859317282394719449e00,
            0.9313862248321055503829503e00,
            0.8757863440192765677772914e00,
            0.8201868063589761051746975e00,
            0.7645876922981545448147078e00,
            0.7089891068198449136125464e00,
            0.6533911899285832425290628e00,
            0.5977941329592257586198087e00,
            0.5421982048745539015834188e00,
            0.4866037965045890355211229e00,
            0.4310114988353693539492225e00,
            0.3754222503860499120445385e00,
            0.3198376369331602148544626e00,
            0.2642605649958747239907310e00,
            0.2086969927688100977274751e00,
            0.1531613237261629042774314e00,
            0.9769922156300582041279299e-01,
            0.4256272861907242306694832e-01,
        ]
    )
    EvenThetaZero29 = np.array(
        [
            0.1543946088331101630230404e01,
            0.1490245617072432741470241e01,
            0.1436545162952171175361532e01,
            0.1382844737841275627385236e01,
            0.1329144354302189376680665e01,
            0.1275444025914442882448630e01,
            0.1221743767654456436125309e01,
            0.1168043596353244531685999e01,
            0.1114343531263457295536939e01,
            0.1060643594778787047442989e01,
            0.1006943813366184678568021e01,
            0.9532442187977767941200107e00,
            0.8995448498101763729640445e00,
            0.8458457543830885615091264e00,
            0.7921469929325243736682034e00,
            0.7384486428849507503612470e00,
            0.6847508053901545384892447e00,
            0.6310536154445759741044291e00,
            0.5773572576394624029563656e00,
            0.5236619915567428835581025e00,
            0.4699681944935857341529219e00,
            0.4162764370726533962791279e00,
            0.3625876255789859906927245e00,
            0.3089032914359211154562848e00,
            0.2552262416643531728802047e00,
            0.2015622306384971766058615e00,
            0.1479251692966707827334002e00,
            0.9435916010280739398532997e-01,
            0.4110762866287674188292735e-01,
        ]
    )
    EvenThetaZero30 = np.array(
        [
            0.1544833637851665335244669e01,
            0.1492908264756388370493025e01,
            0.1440982906138650837480037e01,
            0.1389057572001580364167786e01,
            0.1337132272892735072773304e01,
            0.1285207020157876647295968e01,
            0.1233281826234298389291217e01,
            0.1181356705000596722238457e01,
            0.1129431672204958843918638e01,
            0.1077506746001711267258715e01,
            0.1025581947637229234301640e01,
            0.9736573023432582093437126e00,
            0.9217328405213548692702866e00,
            0.8698085993416727107979968e00,
            0.8178846249414537373941032e00,
            0.7659609755086193214466010e00,
            0.7140377257012462393241274e00,
            0.6621149731355525426273686e00,
            0.6101928481720243483360470e00,
            0.5582715291407654489802101e00,
            0.5063512668959282414914789e00,
            0.4544324261262307197237056e00,
            0.4025155584642650335664553e00,
            0.3506015401168133792671488e00,
            0.2986918517703509333332016e00,
            0.2467892075469457255751440e00,
            0.1948991714956708008247732e00,
            0.1430351946011564171352354e00,
            0.9123992133264713232350199e-01,
            0.3974873026126591246235829e-01,
        ]
    )
    EvenThetaZero31 = np.array(
        [
            0.1545664389841685834178882e01,
            0.1495400520006868605194165e01,
            0.1445136662469633349524466e01,
            0.1394872825707861196682996e01,
            0.1344609018631531661347402e01,
            0.1294345250782284139500904e01,
            0.1244081532562166402923175e01,
            0.1193817875503760392032898e01,
            0.1143554292597402872188167e01,
            0.1093290798696377946301336e01,
            0.1043027411028491785799717e01,
            0.9927641498535133311947588e00,
            0.9425010393224361375194941e00,
            0.8922381086194002226900769e00,
            0.8419753935054036625982058e00,
            0.7917129384431112475049142e00,
            0.7414507995789214800057706e00,
            0.6911890490185720721582180e00,
            0.6409277811053987947460976e00,
            0.5906671218914768219060599e00,
            0.5404072438741681591850965e00,
            0.4901483897634232956856935e00,
            0.4398909124691513811974471e00,
            0.3896353458699818240468259e00,
            0.3393825380385224469051922e00,
            0.2891339221891949677776928e00,
            0.2388921255071779766209942e00,
            0.1886625339124777570188312e00,
            0.1384581678870181657476050e00,
            0.8832030722827102577102185e-01,
            0.3847679847963676404657822e-01,
        ]
    )
    EvenThetaZero32 = np.array(
        [
            0.1546443627125265521960044e01,
            0.1497738231263909315513507e01,
            0.1449032845902631477147772e01,
            0.1400327478265391242178337e01,
            0.1351622135921668846451224e01,
            0.1302916826944702448727527e01,
            0.1254211560091483702838765e01,
            0.1205506345013417018443405e01,
            0.1156801192508980685500292e01,
            0.1108096114833249453312212e01,
            0.1059391126084216587933501e01,
            0.1010686242693213908544820e01,
            0.9619814840575052973573711e00,
            0.9132768733691264344256970e00,
            0.8645724387181842642305406e00,
            0.8158682145859558652971026e00,
            0.7671642439014559105969752e00,
            0.7184605809290069459742089e00,
            0.6697572954095121564500879e00,
            0.6210544786425143220264938e00,
            0.5723522526623283741373995e00,
            0.5236507845164779831804685e00,
            0.4749503092950064087413842e00,
            0.4262511688770346357064771e00,
            0.3775538805043668894422883e00,
            0.3288592658750793954850446e00,
            0.2801687136893753887834348e00,
            0.2314847695998852605184853e00,
            0.1828126524563463299986617e00,
            0.1341649789468091132459783e00,
            0.8558174883654483804697753e-01,
            0.3728374374031613183399036e-01,
        ]
    )
    EvenThetaZero33 = np.array(
        [
            0.1547175997094614757138430e01,
            0.1499935340679181525271649e01,
            0.1452694693272706215568985e01,
            0.1405454061061768876728643e01,
            0.1358213450511184239883293e01,
            0.1310972868490444296079765e01,
            0.1263732322416537730871712e01,
            0.1216491820419724046503073e01,
            0.1169251371540540180758674e01,
            0.1122010985968754004469355e01,
            0.1074770675338453464761893e01,
            0.1027530453098431393666936e01,
            0.9802903349842005856557204e00,
            0.9330503396284544173873149e00,
            0.8858104893623263267477775e00,
            0.8385708112832335506864354e00,
            0.7913313387011139500976360e00,
            0.7440921131314510897906335e00,
            0.6968531870945337206139839e00,
            0.6496146281309018959581539e00,
            0.6023765246993705639765525e00,
            0.5551389950762090311242875e00,
            0.5079022012032895030848024e00,
            0.4606663710240282967569630e00,
            0.4134318360639670775957014e00,
            0.3661990979414348851212686e00,
            0.3189689535781378596191439e00,
            0.2717427498485401725509746e00,
            0.2245229557871702595200694e00,
            0.1773146332323969343091350e00,
            0.1301300193754780766338959e00,
            0.8300791095077070533235660e-01,
            0.3616244959900389221395842e-01,
        ]
    )
    EvenThetaZero34 = np.array(
        [
            0.1547865604457777747119921e01,
            0.1502004162357357213441384e01,
            0.1456142728021903760325049e01,
            0.1410281306774684706589738e01,
            0.1364419904164498130803254e01,
            0.1318558526067441138200403e01,
            0.1272697178801115154796514e01,
            0.1226835869256177571730448e01,
            0.1180974605051351016009903e01,
            0.1135113394719709026888693e01,
            0.1089252247936466574864114e01,
            0.1043391175801911243726755e01,
            0.9975301911979639874925565e00,
            0.9516693092438447484954432e00,
            0.9058085478865097428655118e00,
            0.8599479286766250282572181e00,
            0.8140874778035996603018790e00,
            0.7682272274981820559251592e00,
            0.7223672179660643783333797e00,
            0.6765075001043380283085699e00,
            0.6306481393987597674748178e00,
            0.5847892216487432573582268e00,
            0.5389308616059791284685642e00,
            0.4930732164176132508179420e00,
            0.4472165073094733435432890e00,
            0.4013610560689043520551232e00,
            0.3555073496130768130758891e00,
            0.3096561615434305328219637e00,
            0.2638087993597793691714182e00,
            0.2179676599607749036552390e00,
            0.1721376573496165890967450e00,
            0.1263306713881449555499955e00,
            0.8058436603519718986295825e-01,
            0.3510663068970053260227480e-01,
        ]
    )
    EvenThetaZero35 = np.array(
        [
            0.1548516088202564202943238e01,
            0.1503955613246577879586994e01,
            0.1459395145012190281751360e01,
            0.1414834688100222735099866e01,
            0.1370274247295441414922756e01,
            0.1325713827649021532002630e01,
            0.1281153434570536124285912e01,
            0.1236593073933169034954499e01,
            0.1192032752196710979323473e01,
            0.1147472476554108430135576e01,
            0.1102912255109027578275434e01,
            0.1058352097094263144928973e01,
            0.1013792013144153206047048e01,
            0.9692320156388929821870602e00,
            0.9246721191454417746654622e00,
            0.8801123409896300773149632e00,
            0.8355527020087518049947413e00,
            0.7909932275560464363973909e00,
            0.7464339488624693592395086e00,
            0.7018749049145358048463504e00,
            0.6573161450929179933243905e00,
            0.6127577329584494909986789e00,
            0.5681997518140860838771656e00,
            0.5236423130979094957496400e00,
            0.4790855694444512920982626e00,
            0.4345297357523596151738496e00,
            0.3899751246318782591316393e00,
            0.3454222091410984787772492e00,
            0.3008717408917773811461237e00,
            0.2563249902500918978614004e00,
            0.2117842860782107775954396e00,
            0.1672544029381415755198150e00,
            0.1227468836419337342946123e00,
            0.7829832364814667171382217e-01,
            0.3411071484766340151578357e-01,
        ]
    )
    EvenThetaZero36 = np.array(
        [
            0.1549130685823945998342524e01,
            0.1505799405819664254557106e01,
            0.1462468131657470292685966e01,
            0.1419136867330461353369368e01,
            0.1375805616982638895139986e01,
            0.1332474384976155365522566e01,
            0.1289143175965912901391449e01,
            0.1245811994984327181800398e01,
            0.1202480847539690438616688e01,
            0.1159149739732435788417226e01,
            0.1115818678394807971862305e01,
            0.1072487671261111519215409e01,
            0.1029156727178025494814510e01,
            0.9858258563677261466814511e00,
            0.9424950707611702085500992e00,
            0.8991643844255133860018485e00,
            0.8558338141192845596532563e00,
            0.8125033800232146117493243e00,
            0.7691731067161328174004981e00,
            0.7258430244984030733808537e00,
            0.6825131712172895509836733e00,
            0.6391835948321685576634513e00,
            0.5958543570955633038336902e00,
            0.5525255389612023677479152e00,
            0.5091972487450747080139606e00,
            0.4658696348260689008126722e00,
            0.4225429061321313393543928e00,
            0.3792173666095906812269559e00,
            0.3358934762285008809293807e00,
            0.2925719658301625547639832e00,
            0.2492540707015179370724365e00,
            0.2059420554273186332219697e00,
            0.1626405628266886976038507e00,
            0.1193608172622853851645011e00,
            0.7613840464754681957544313e-01,
            0.3316974474186058622824911e-01,
        ]
    )
    EvenThetaZero37 = np.array(
        [
            0.1549712287207882890839045e01,
            0.1507544209724862511636878e01,
            0.1465376137339015815734558e01,
            0.1423208073529702865859582e01,
            0.1381040021900765225468989e01,
            0.1338871986235691269778498e01,
            0.1296703970558498635765633e01,
            0.1254535979202491212629656e01,
            0.1212368016889500927716256e01,
            0.1170200088822853513468851e01,
            0.1128032200798161849314963e01,
            0.1085864359337236600941540e01,
            0.1043696571852037437540940e01,
            0.1001528846847853898635169e01,
            0.9593611941780778060127795e00,
            0.9171936253674231737318512e00,
            0.8750261540268988114426643e00,
            0.8328587963932301252176965e00,
            0.7906915720393251716472997e00,
            0.7485245048233193695739358e00,
            0.7063576241759074809548715e00,
            0.6641909668761970070284373e00,
            0.6220245795476036586681135e00,
            0.5798585222396645710869275e00,
            0.5376928736905555113005422e00,
            0.4955277392687366749125653e00,
            0.4533632633323484070376718e00,
            0.4111996491651493998151895e00,
            0.3690371925202636251212886e00,
            0.3268763409876008462653069e00,
            0.2847178057580674399826003e00,
            0.2425627889274157106498810e00,
            0.2004134942584602007834507e00,
            0.1582744399049656648660257e00,
            0.1161565488818554609430574e00,
            0.7409445176394481360104851e-01,
            0.3227929535095246410912398e-01,
        ]
    )
    EvenThetaZero38 = np.array(
        [
            0.1550263480064160377720298e01,
            0.1509197788083808185665328e01,
            0.1468132100566875710992083e01,
            0.1427066420556418463513913e01,
            0.1386000751198712817289420e01,
            0.1344935095788765217267069e01,
            0.1303869457820298477498722e01,
            0.1262803841041882838326682e01,
            0.1221738249521212843639205e01,
            0.1180672687719991159061894e01,
            0.1139607160582508034089119e01,
            0.1098541673641858946868449e01,
            0.1057476233148907719560749e01,
            0.1016410846230700992453501e01,
            0.9753455210872527645472818e00,
            0.9342802672387126698703291e00,
            0.8932150958393123732306518e00,
            0.8521500200807685012223049e00,
            0.8110850557169691024167180e00,
            0.7700202217553081279468270e00,
            0.7289555413804262510029339e00,
            0.6878910432074509889956044e00,
            0.6468267630110350344178276e00,
            0.6057627461556542068727688e00,
            0.5646990510834698732732127e00,
            0.5236357544389875315454201e00,
            0.4825729588028297682338108e00,
            0.4415108047277878179738561e00,
            0.4004494901533595099830119e00,
            0.3593893030723592157150581e00,
            0.3183306793460978083354355e00,
            0.2772743115465352362860883e00,
            0.2362213703174823832436869e00,
            0.1951740017836102296584907e00,
            0.1541366059551230775894261e00,
            0.1131198202589878992052369e00,
            0.7215736988593890187079586e-01,
            0.3143540438351454384152236e-01,
        ]
    )
    EvenThetaZero39 = np.array(
        [
            0.1550786588415152297375587e01,
            0.1510767112957397367780716e01,
            0.1470747641421582916022579e01,
            0.1430728176478592843861361e01,
            0.1390708720885325111445925e01,
            0.1350689277522434511387126e01,
            0.1310669849435604714836514e01,
            0.1270650439881648370588402e01,
            0.1230631052380981613091250e01,
            0.1190611690778358944744052e01,
            0.1150592359314214516523625e01,
            0.1110573062709576809284752e01,
            0.1070553806268363352417161e01,
            0.1030534596002003296175373e01,
            0.9905154387828984834423913e00,
            0.9504963425353941517573974e00,
            0.9104773164759498161192732e00,
            0.8704583714184727086854142e00,
            0.8304395201669023270865304e00,
            0.7904207780260519973626051e00,
            0.7504021634749074983118715e00,
            0.7103836990664583264642972e00,
            0.6703654126486745769832673e00,
            0.6303473390491956215820085e00,
            0.5903295224434431765765323e00,
            0.5503120197533818815098408e00,
            0.5102949056413983084126817e00,
            0.4702782800468414863285692e00,
            0.4302622799152491769326599e00,
            0.3902470981180917254123191e00,
            0.3502330152869736207185960e00,
            0.3102204561556976356809728e00,
            0.2702100956292792195263915e00,
            0.2302030745053307298726703e00,
            0.1902014842102915167005070e00,
            0.1502096126336221315300686e00,
            0.1102378261690820867329259e00,
            0.7031899075931525095025389e-01,
            0.3063451333411226493032265e-01,
        ]
    )
    EvenThetaZero40 = np.array(
        [
            0.1551283705347968314195100e01,
            0.1512258463601911009913297e01,
            0.1473233225313284690780287e01,
            0.1434207992834186122366616e01,
            0.1395182768588723275108301e01,
            0.1356157555104474252423723e01,
            0.1317132355046745793679891e01,
            0.1278107171256650000336432e01,
            0.1239082006794203284097135e01,
            0.1200056864987904389011051e01,
            0.1161031749492588664002624e01,
            0.1122006664357811755961100e01,
            0.1082981614109627397900573e01,
            0.1043956603849447575483550e01,
            0.1004931639374790125389322e01,
            0.9659067273282460489273148e00,
            0.9268818753831082867718635e00,
            0.8878570924770502938457708e00,
            0.8488323891094102606331406e00,
            0.8098077777236123075833052e00,
            0.7707832732049530424809748e00,
            0.7317588935368492604710264e00,
            0.6927346606780251833003950e00,
            0.6537106017528970872810663e00,
            0.6146867506941756306797580e00,
            0.5756631505519364744300804e00,
            0.5366398568077528417370132e00,
            0.4976169422443344500752625e00,
            0.4585945042946725387136724e00,
            0.4195726764797194195007418e00,
            0.3805516468579533335376469e00,
            0.3415316890685593880011997e00,
            0.3025132172735989410463832e00,
            0.2634968895917008761291809e00,
            0.2244838184598823563259898e00,
            0.1854760433267094750424413e00,
            0.1464777455344068532549101e00,
            0.1074990339130794792907032e00,
            0.6857195785426972961368108e-01,
            0.2987341732561906608807860e-01,
        ]
    )
    EvenThetaZero41 = np.array(
        [
            0.1551756721003315464043007e01,
            0.1513677510435354867644006e01,
            0.1475598302924814895692182e01,
            0.1437519100549654116408972e01,
            0.1399439905448387106945081e01,
            0.1361360719846430407096351e01,
            0.1323281546084682430842605e01,
            0.1285202386651141609385598e01,
            0.1247123244216506877361870e01,
            0.1209044121674894401873626e01,
            0.1170965022191058363946285e01,
            0.1132885949255841486220662e01,
            0.1094806906752030657845562e01,
            0.1056727899033393535018723e01,
            0.1018648931020478788570327e01,
            0.9805700083178549567966928e00,
            0.9424911373589552049711100e00,
            0.9044123255867553868253384e00,
            0.8663335816813894348633149e00,
            0.8282549158498738099497389e00,
            0.7901763401989443875774432e00,
            0.7520978692204962458482329e00,
            0.7140195204316730003387055e00,
            0.6759413152305656820841666e00,
            0.6378632800575392064866756e00,
            0.5997854479978337579981629e00,
            0.5617078610344953281799357e00,
            0.5236305732820186728652802e00,
            0.4855536557378012985520074e00,
            0.4474772034530068342865487e00,
            0.4094013466928584958758982e00,
            0.3713262689388439070717808e00,
            0.3332522371792479009733062e00,
            0.2951796555193184134657530e00,
            0.2571091661074227554417865e00,
            0.2190418543971735546480404e00,
            0.1809797103814301725822348e00,
            0.1429268140230164119614409e00,
            0.1048930290780323497410212e00,
            0.6690962797843649866645769e-01,
            0.2914922224685900914817542e-01,
        ]
    )
    EvenThetaZero42 = np.array(
        [
            0.1552207346590136182648920e01,
            0.1515029387081184115266415e01,
            0.1477851430283927973458023e01,
            0.1440673478039699629370259e01,
            0.1403495532240969264030648e01,
            0.1366317594853508812224152e01,
            0.1329139667940348087929429e01,
            0.1291961753688162615428688e01,
            0.1254783854436838464182091e01,
            0.1217605972713102930414639e01,
            0.1180428111269300876868432e01,
            0.1143250273128649048802100e01,
            0.1106072461638634327789036e01,
            0.1068894680534663975270023e01,
            0.1031716934016664760314029e01,
            0.9945392268421176498894610e00,
            0.9573615644400829018748874e00,
            0.9201839530522288586731642e00,
            0.8830063999088902711516820e00,
            0.8458289134509915302518266e00,
            0.8086515036126424512848147e00,
            0.7714741821849085841225787e00,
            0.7342969632895448309937051e00,
            0.6971198640037406540069491e00,
            0.6599429051953912854163132e00,
            0.6227661126567800124770610e00,
            0.5855895186691062254659102e00,
            0.5484131642019636734351025e00,
            0.5112371020703309674589504e00,
            0.4740614015734592960802666e00,
            0.4368861554959151187817336e00,
            0.3997114910036376358365916e00,
            0.3625375872199777754435892e00,
            0.3253647047992267079974806e00,
            0.2881932382678453273830096e00,
            0.2510238145617968753500674e00,
            0.2138574934303919974438356e00,
            0.1766962177535783269128215e00,
            0.1395439709154010255199071e00,
            0.1024103832005221866954023e00,
            0.6532598686141261097119747e-01,
            0.2845930797694291389393445e-01,
        ]
    )
    EvenThetaZero43 = np.array(
        [
            0.1552637135069155811491072e01,
            0.1516318752418798211357541e01,
            0.1480000372180291690418989e01,
            0.1443681995989991700140976e01,
            0.1407363625527612735973164e01,
            0.1371045262534953065860219e01,
            0.1334726908836065747097909e01,
            0.1298408566359386697763653e01,
            0.1262090237162411913706886e01,
            0.1225771923459625279363960e01,
            0.1189453627654523146514386e01,
            0.1153135352376772077918208e01,
            0.1116817100525785826106551e01,
            0.1080498875322336017099434e01,
            0.1044180680370244915946738e01,
            0.1007862519730785566833872e01,
            0.9715443980131875264637689e00,
            0.9352263204856910439167915e00,
            0.8989082932130182550456316e00,
            0.8625903232280967802521182e00,
            0.8262724187486163930514201e00,
            0.7899545894528804342126058e00,
            0.7536368468349768085155075e00,
            0.7173192046673890278545072e00,
            0.6810016796111441673128480e00,
            0.6446842920316340773745262e00,
            0.6083670671059611518530899e00,
            0.5720500363511797523369558e00,
            0.5357332397728172506411618e00,
            0.4994167289487775362163415e00,
            0.4631005715608865274454686e00,
            0.4267848582339839676363509e00,
            0.3904697131799790288672503e00,
            0.3541553113674441557740819e00,
            0.3178419074113077198829473e00,
            0.2815298867038369044519273e00,
            0.2452198616736214006194288e00,
            0.2089128675558041239775998e00,
            0.1726108022974787183994402e00,
            0.1363175571713249458600521e00,
            0.1000425397881322914313825e00,
            0.6381557644960651200944222e-01,
            0.2780129671121636039734655e-01,
        ]
    )
    EvenThetaZero44 = np.array(
        [
            0.1553047499032218401181962e01,
            0.1517549844221432542461907e01,
            0.1482052191561582448658478e01,
            0.1446554542510861055782865e01,
            0.1411056898564365493121105e01,
            0.1375559261269981001734724e01,
            0.1340061632245437638964436e01,
            0.1304564013196950335363525e01,
            0.1269066405939915513649123e01,
            0.1233568812422221364483924e01,
            0.1198071234750839346739124e01,
            0.1162573675222508872274463e01,
            0.1127076136359515473862368e01,
            0.1091578620951808778363231e01,
            0.1056081132107029235444226e01,
            0.1020583673310438024843461e01,
            0.9850862484973095869616622e00,
            0.9495888621411026369897815e00,
            0.9140915193617473526041913e00,
            0.8785942260597805964360395e00,
            0.8430969890839839780181254e00,
            0.8075998164428632814935249e00,
            0.7721027175741014967901450e00,
            0.7366057036915554827257553e00,
            0.7011087882372792641869964e00,
            0.6656119874777629720186974e00,
            0.6301153213012084608241887e00,
            0.5946188142997514629085459e00,
            0.5591224972630766104664894e00,
            0.5236264092783024624074546e00,
            0.4881306007441175888503326e00,
            0.4526351377998500905914452e00,
            0.4171401090099414677462070e00,
            0.3816456357674021470057899e00,
            0.3461518890753412856675063e00,
            0.3106591177837409768492156e00,
            0.2751676985649013361686770e00,
            0.2396782299970584002479842e00,
            0.2041917239104339765549482e00,
            0.1687100353513348647833163e00,
            0.1332369676454340307348264e00,
            0.9778171579501174586520881e-01,
            0.6237343205901608270979365e-01,
            0.2717302558182235133513210e-01,
        ]
    )
    EvenThetaZero45 = np.array(
        [
            0.1553439726211153891540573e01,
            0.1518726525682668668950427e01,
            0.1484013327077361052080319e01,
            0.1449300131698066374929113e01,
            0.1414586940879145218883617e01,
            0.1379873756000009717714844e01,
            0.1345160578499605494109603e01,
            0.1310447409892181029407508e01,
            0.1275734251784724823396464e01,
            0.1241021105896515467487132e01,
            0.1206307974081314658309029e01,
            0.1171594858352843571506531e01,
            0.1136881760914326165420300e01,
            0.1102168684193068774494217e01,
            0.1067455630881287279906518e01,
            0.1032742603984709761582283e01,
            0.9980296068808995413713835e00,
            0.9633166433897968474836258e00,
            0.9286037178597176902839922e00,
            0.8938908352730483454962679e00,
            0.8591780013772376740585140e00,
            0.8244652228485703565016715e00,
            0.7897525074988288740747291e00,
            0.7550398645386622329842600e00,
            0.7203273049167972965433221e00,
            0.6856148417619669061621766e00,
            0.6509024909658764678789680e00,
            0.6161902719627732109904446e00,
            0.5814782087876726421060849e00,
            0.5467663315368932859708410e00,
            0.5120546784214694424751802e00,
            0.4773432987146161851453875e00,
            0.4426322570828636775769209e00,
            0.4079216401227574252826633e00,
            0.3732115665343573673240355e00,
            0.3385022035318641142927744e00,
            0.3037937944563405612019789e00,
            0.2690867076466992914990193e00,
            0.2343815284441088285495466e00,
            0.1996792463094099012688324e00,
            0.1649816752853099621072722e00,
            0.1302925346385956500837770e00,
            0.9562081616094948269905207e-01,
            0.6099502786102040135198395e-01,
            0.2657252290854776665952679e-01,
        ]
    )
    EvenThetaZero46 = np.array(
        [
            0.1553814992974904767594241e01,
            0.1519852325907741898557817e01,
            0.1485889660564341242674032e01,
            0.1451926998111647785152899e01,
            0.1417964339743630985906479e01,
            0.1384001686692845945859686e01,
            0.1350039040242776872946770e01,
            0.1316076401741232369348729e01,
            0.1282113772615099921371445e01,
            0.1248151154386817288949698e01,
            0.1214188548692984168143550e01,
            0.1180225957305622474388020e01,
            0.1146263382156703179022046e01,
            0.1112300825366698998613230e01,
            0.1078338289278105103916832e01,
            0.1044375776495107627552926e01,
            0.1010413289930890288650173e01,
            0.9764508328644780886953041e00,
            0.9424884090095589354132202e00,
            0.9085260225984488659490189e00,
            0.8745636784853451215455853e00,
            0.8406013822743460048537475e00,
            0.8066391404795569177534715e00,
            0.7726769607271702244889884e00,
            0.7387148520130367387469271e00,
            0.7047528250344497011443004e00,
            0.6707908926224332706815892e00,
            0.6368290703120276715090693e00,
            0.6028673771049329733376093e00,
            0.5689058365047911420524623e00,
            0.5349444779460832748774921e00,
            0.5009833388030907720537138e00,
            0.4670224672735823328060142e00,
            0.4330619266162571710985599e00,
            0.3991018015460700850326972e00,
            0.3651422081877256344485503e00,
            0.3311833101314466311103548e00,
            0.2972253454486352538763297e00,
            0.2632686745061683534910424e00,
            0.2293138699815081215985284e00,
            0.1953618999343470689252174e00,
            0.1614145391777897730914718e00,
            0.1274754265555317105245073e00,
            0.9355335943686297111639257e-01,
            0.5967622944002585907962555e-01,
            0.2599798753052849047032580e-01,
        ]
    )
    EvenThetaZero47 = np.array(
        [
            0.1554174376112911655131098e01,
            0.1520930475263640362170511e01,
            0.1487686575963027013435604e01,
            0.1454442679258803180913942e01,
            0.1421198786221944168258440e01,
            0.1387954897956585365296993e01,
            0.1354711015610581847809736e01,
            0.1321467140386931222410853e01,
            0.1288223273556309404505081e01,
            0.1254979416471008337267759e01,
            0.1221735570580615776412743e01,
            0.1188491737449843097435062e01,
            0.1155247918778991542491874e01,
            0.1122004116427655660730083e01,
            0.1088760332442401967089102e01,
            0.1055516569089340593777585e01,
            0.1022272828892740925095715e01,
            0.9890291146811467076264609e00,
            0.9557854296428465678959260e00,
            0.9225417773930866874628226e00,
            0.8892981620561221868061383e00,
            0.8560545883661619186153440e00,
            0.8228110617925680415850631e00,
            0.7895675886964734656602191e00,
            0.7563241765284943282959446e00,
            0.7230808340807681383862155e00,
            0.6898375718116413059811978e00,
            0.6565944022687408111136058e00,
            0.6233513406471279431598408e00,
            0.5901084055357449335782332e00,
            0.5568656199307345199294838e00,
            0.5236230126340485109018232e00,
            0.4903806202198476810807501e00,
            0.4571384898571183050552302e00,
            0.4238966834573972483152713e00,
            0.3906552839347125500730013e00,
            0.3574144049483910279156003e00,
            0.3241742066189948531421192e00,
            0.2909349219721993995636414e00,
            0.2576969037411283384416169e00,
            0.2244607124763750082606152e00,
            0.1912272957431274569912962e00,
            0.1579983907861406744991899e00,
            0.1247775594308675650267811e00,
            0.9157341285433675818728635e-01,
            0.5841325237532701385812948e-01,
            0.2544777076240816313972829e-01,
        ]
    )
    EvenThetaZero48 = np.array(
        [
            0.1554518863153354618809409e01,
            0.1521963936333782670214978e01,
            0.1489409010908686292228052e01,
            0.1456854087820918568482631e01,
            0.1424299168033388494075931e01,
            0.1391744252537595165009714e01,
            0.1359189342362693116905575e01,
            0.1326634438585269225516707e01,
            0.1294079542340034988016159e01,
            0.1261524654831668904330407e01,
            0.1228969777348083696352705e01,
            0.1196414911275444418157033e01,
            0.1163860058115329026827193e01,
            0.1131305219504506571098859e01,
            0.1098750397237914982841550e01,
            0.1066195593295557461150055e01,
            0.1033640809874212986016967e01,
            0.1001086049425085324651032e01,
            0.9685313146988134601280153e00,
            0.9359766087996588330547245e00,
            0.9034219352512048766203636e00,
            0.8708672980765996496647291e00,
            0.8383127018973108640833295e00,
            0.8057581520556423644789438e00,
            0.7732036547680256450048242e00,
            0.7406492173185620802637676e00,
            0.7080948483057714882525616e00,
            0.6755405579604902654406567e00,
            0.6429863585601198571817691e00,
            0.6104322649751623629236805e00,
            0.5778782954001507969801437e00,
            0.5453244723459250134170285e00,
            0.5127708240092147734858477e00,
            0.4802173861982495342372455e00,
            0.4476642050968422792532389e00,
            0.4151113413261211455132671e00,
            0.3825588760747025757978563e00,
            0.3500069206395502661556462e00,
            0.3174556318161704671189642e00,
            0.2849052377944113082878058e00,
            0.2523560839907875626097181e00,
            0.2198087193323827316322426e00,
            0.1872640717400572601243546e00,
            0.1547238424480887172335593e00,
            0.1221915194567498709631299e00,
            0.8967553546914315204781840e-01,
            0.5720262597323678474637133e-01,
            0.2492036059421555107245208e-01,
        ]
    )
    EvenThetaZero49 = np.array(
        [
            0.1554849361424470843090118e01,
            0.1522955431101933730645303e01,
            0.1491061502037751976297424e01,
            0.1459167575082261894770634e01,
            0.1427273651103158170602525e01,
            0.1395379730992862183093282e01,
            0.1363485815676330697886480e01,
            0.1331591906119453530640248e01,
            0.1299698003338207337770238e01,
            0.1267804108408757103237650e01,
            0.1235910222478728395590872e01,
            0.1204016346779913703159712e01,
            0.1172122482642727288439229e01,
            0.1140228631512787910483320e01,
            0.1108334794970091261912531e01,
            0.1076440974751339138680154e01,
            0.1044547172776127017814204e01,
            0.1012653391177865049408482e01,
            0.9807596323405319627306720e00,
            0.9488658989426541583823449e00,
            0.9169721940102869797082899e00,
            0.8850785209812848825432963e00,
            0.8531848837838285971960304e00,
            0.8212912869330969580404013e00,
            0.7893977356512249782795147e00,
            0.7575042360174185552669765e00,
            0.7256107951575083863622461e00,
            0.6937174214856350398887716e00,
            0.6618241250156435481462849e00,
            0.6299309177668761147121611e00,
            0.5980378142995696297245189e00,
            0.5661448324309071185385071e00,
            0.5342519942071113461815355e00,
            0.5023593272451872220760104e00,
            0.4704668666194035162003700e00,
            0.4385746575692260390945883e00,
            0.4066827594785525726660483e00,
            0.3747912518813925812276922e00,
            0.3429002438089823350625543e00,
            0.3110098888674705209106637e00,
            0.2791204106078991711912441e00,
            0.2472321474279120600810915e00,
            0.2153456371036966567922014e00,
            0.1834617887100953140198272e00,
            0.1515822689338083535939382e00,
            0.1197104949484175660714864e00,
            0.8785472823121690639967810e-01,
            0.5604116141749524467628553e-01,
            0.2441436781606819510490200e-01,
        ]
    )
    EvenThetaZero50 = np.array(
        [
            0.1555166706034023842787706e01,
            0.1523907464890582273398300e01,
            0.1492648224885016483409279e01,
            0.1461388986785839210767990e01,
            0.1430129751376631035251350e01,
            0.1398870519462421720845393e01,
            0.1367611291876438076975682e01,
            0.1336352069487341263827064e01,
            0.1305092853207091240256110e01,
            0.1273833643999595441027728e01,
            0.1242574442890323705495464e01,
            0.1211315250977103200801981e01,
            0.1180056069442347222927677e01,
            0.1148796899567022469820701e01,
            0.1117537742746723499546780e01,
            0.1086278600510304367969825e01,
            0.1055019474541620880304106e01,
            0.1023760366705069175705639e01,
            0.9925012790757765081567637e00,
            0.9612422139755203374385844e00,
            0.9299831740157389822411293e00,
            0.8987241621493743180375722e00,
            0.8674651817337867269827651e00,
            0.8362062366076504452859926e00,
            0.8049473311856388413561914e00,
            0.7736884705759381993127359e00,
            0.7424296607273230664538510e00,
            0.7111709086148904840060001e00,
            0.6799122224768919982385331e00,
            0.6486536121198915829209739e00,
            0.6173950893164463798129595e00,
            0.5861366683298159921278400e00,
            0.5548783666157332634604655e00,
            0.5236202057751242467455922e00,
            0.4923622128691229579358494e00,
            0.4611044222679868504944176e00,
            0.4298468783051183048132298e00,
            0.3985896391770900735176252e00,
            0.3673327828297899556279530e00,
            0.3360764161195064368209114e00,
            0.3048206895905456571224703e00,
            0.2735658223403245791263072e00,
            0.2423121460275046288225596e00,
            0.2110601877217048587999889e00,
            0.1798108384023314561549010e00,
            0.1485657315840060835766576e00,
            0.1173282164330337207824850e00,
            0.8610639001623934211634967e-01,
            0.5492592372249737419414775e-01,
            0.2392851379957687254895331e-01,
        ]
    )

    OddThetaZero1 = np.array([0.6847192030022829138880982e00])
    OddThetaZero2 = np.array([0.1002176803643121641749915e01, 0.4366349492255221620374655e00])
    OddThetaZero3 = np.array(
        [0.1152892953722227341986065e01, 0.7354466143229520469385622e00, 0.3204050902900619825355950e00]
    )
    OddThetaZero4 = np.array(
        [
            0.1240573923404363422789550e01,
            0.9104740292261473250358755e00,
            0.5807869795060065580284919e00,
            0.2530224166119306882187233e00,
        ]
    )
    OddThetaZero5 = np.array(
        [
            0.1297877729331450368298142e01,
            0.1025003226369574843297844e01,
            0.7522519395990821317003373e00,
            0.4798534223256743217333579e00,
            0.2090492874137409414071522e00,
        ]
    )
    OddThetaZero6 = np.array(
        [
            0.1338247676100454369194835e01,
            0.1105718066248490075175419e01,
            0.8732366099401630367220948e00,
            0.6408663264733867770811230e00,
            0.4088002373420211722955679e00,
            0.1780944581262765470585931e00,
        ]
    )
    OddThetaZero7 = np.array(
        [
            0.1368219536992351783359098e01,
            0.1165652065603030148723847e01,
            0.9631067821301481995711685e00,
            0.7606069572889918619145483e00,
            0.5582062109125313357140248e00,
            0.3560718303314725022788878e00,
            0.1551231069747375098418591e00,
        ]
    )
    OddThetaZero8 = np.array(
        [
            0.1391350647015287461874435e01,
            0.1211909966211469688151240e01,
            0.1032480728417239563449772e01,
            0.8530732514258505686069670e00,
            0.6737074594242522259878462e00,
            0.4944303818194983217354808e00,
            0.3153898594929282395996014e00,
            0.1373998952992547671039022e00,
        ]
    )
    OddThetaZero9 = np.array(
        [
            0.1409742336767428999667236e01,
            0.1248691224331339221187704e01,
            0.1087646521650454938943641e01,
            0.9266134127998189551499083e00,
            0.7656007620508340547558669e00,
            0.6046261769405451549818494e00,
            0.4437316659960951760051408e00,
            0.2830497588453068048261493e00,
            0.1233108673082312764916251e00,
        ]
    )
    OddThetaZero10 = np.array(
        [
            0.1424715475176742734932665e01,
            0.1278636375242898727771561e01,
            0.1132561101012537613667002e01,
            0.9864925055883793730483278e00,
            0.8404350520135058972624775e00,
            0.6943966110110701016065380e00,
            0.5483930281810389839680525e00,
            0.4024623099018152227701990e00,
            0.2567245837448891192759858e00,
            0.1118422651428890834760883e00,
        ]
    )
    OddThetaZero11 = np.array(
        [
            0.1437141935303526306632113e01,
            0.1303488659735581140681362e01,
            0.1169837785762829821262819e01,
            0.1036190996404462300207004e01,
            0.9025507517347875930425807e00,
            0.7689210263823624893974324e00,
            0.6353089402976822861185532e00,
            0.5017289283414202278167583e00,
            0.3682157131008289798868520e00,
            0.2348791589702580223688923e00,
            0.1023252788872632487579640e00,
        ]
    )
    OddThetaZero12 = np.array(
        [
            0.1447620393135667144403507e01,
            0.1324445197736386798102445e01,
            0.1201271573324181312770120e01,
            0.1078100568411879956441542e01,
            0.9549336362382321811515336e00,
            0.8317729718814276781352878e00,
            0.7086221837538611370849622e00,
            0.5854877911108011727748238e00,
            0.4623830630132757357909198e00,
            0.3393399712563371486343129e00,
            0.2164597408964339264361902e00,
            0.9430083986305519349231898e-01,
        ]
    )
    OddThetaZero13 = np.array(
        [
            0.1456575541704195839944967e01,
            0.1342355260834552126304154e01,
            0.1228136043468909663499174e01,
            0.1113918572282611841378549e01,
            0.9997037539874953933323299e00,
            0.8854928869950799998575862e00,
            0.7712879690777516856072467e00,
            0.6570923167092416238233585e00,
            0.5429119513798658239789812e00,
            0.4287591577660783587509129e00,
            0.3146635662674373982102762e00,
            0.2007190266590380629766487e00,
            0.8744338280630300217927750e-01,
        ]
    )
    OddThetaZero14 = np.array(
        [
            0.1464317002991565219979113e01,
            0.1357838033080061766980173e01,
            0.1251359804334884770836945e01,
            0.1144882777708662655968171e01,
            0.1038407544520296695714932e01,
            0.9319349156915986836657782e00,
            0.8254660749671546663859351e00,
            0.7190028636037068047812305e00,
            0.6125483562383020473196681e00,
            0.5061081521562999836102547e00,
            0.3996936914666951732317457e00,
            0.2933325857619472952507468e00,
            0.1871123137498061864373407e00,
            0.8151560650977882057817999e-01,
        ]
    )
    OddThetaZero15 = np.array(
        [
            0.1471075823713997440657641e01,
            0.1371355574944658989649887e01,
            0.1271635855736122280723838e01,
            0.1171916986981363820797100e01,
            0.1072199368669106404814915e01,
            0.9724835301003496870596165e00,
            0.8727702114891848603047954e00,
            0.7730605060747958359120755e00,
            0.6733561257504194406005404e00,
            0.5736599396529727772420934e00,
            0.4739771829190733570809765e00,
            0.3743185619229329461021810e00,
            0.2747099287638327553949437e00,
            0.1752332025619508475799133e00,
            0.7634046205384429302353073e-01,
        ]
    )
    OddThetaZero16 = np.array(
        [
            0.1477027911291552393547878e01,
            0.1383259682348271685979143e01,
            0.1289491840051302622319481e01,
            0.1195724613675799550484673e01,
            0.1101958282220461402990667e01,
            0.1008193204014774090964219e01,
            0.9144298626454031699590564e00,
            0.8206689427646120483710056e00,
            0.7269114630504563073034288e00,
            0.6331590254855162126233733e00,
            0.5394143214244183829842424e00,
            0.4456822679082866369288652e00,
            0.3519729273095236644049666e00,
            0.2583106041071417718760275e00,
            0.1647723231643112502628240e00,
            0.7178317184275122449502857e-01,
        ]
    )
    OddThetaZero17 = np.array(
        [
            0.1482309554825692463999299e01,
            0.1393822922226542123661077e01,
            0.1305336577335833571381699e01,
            0.1216850687682353365944624e01,
            0.1128365453024608460982204e01,
            0.1039881123511957522668140e01,
            0.9513980267579228357946521e00,
            0.8629166105524045911461307e00,
            0.7744375139383604902604254e00,
            0.6859616923374368587817328e00,
            0.5974906525247623278123711e00,
            0.5090269299866796725116786e00,
            0.4205751610647263669405267e00,
            0.3321448379994943116084719e00,
            0.2437588931448048912587688e00,
            0.1554900095178924564386865e00,
            0.6773932498157585698088354e-01,
        ]
    )
    OddThetaZero18 = np.array(
        [
            0.1487027983239550912222135e01,
            0.1403259745496922270264564e01,
            0.1319491725464661433609663e01,
            0.1235724047968681189212364e01,
            0.1151956859289811446164825e01,
            0.1068190338689553494802072e01,
            0.9844247150109837231349622e00,
            0.9006602918737365182850484e00,
            0.8168974877846821404275069e00,
            0.7331369031796229223580227e00,
            0.6493794386888650054486281e00,
            0.5656265174356596757139537e00,
            0.4818805368222631487731579e00,
            0.3981458834052590173509113e00,
            0.3144315409387123154212535e00,
            0.2307592167302372059759857e00,
            0.1471977156945989772472748e00,
            0.6412678117309944052403703e-01,
        ]
    )
    OddThetaZero19 = np.array(
        [
            0.1491268718102344688271411e01,
            0.1411741190914640487505771e01,
            0.1332213830951015404441941e01,
            0.1252686732830809999680267e01,
            0.1173160005794509313174730e01,
            0.1093633781237958896879965e01,
            0.1014108223243148393065201e01,
            0.9345835440325075907377330e00,
            0.8550600276575269107773349e00,
            0.7755380679025248517258532e00,
            0.6960182317959841585145109e00,
            0.6165013717819833504477346e00,
            0.5369888366794912945318079e00,
            0.4574829005269902932408889e00,
            0.3779877260196973978940863e00,
            0.2985118404618624984946326e00,
            0.2190758506462427957069113e00,
            0.1397450765119767349146353e00,
            0.6088003363863534825005464e-01,
        ]
    )
    OddThetaZero20 = np.array(
        [
            0.1495100801651051409999732e01,
            0.1419405340110198552778393e01,
            0.1343710008748627892724810e01,
            0.1268014880389353000310414e01,
            0.1192320038028903827079750e01,
            0.1116625579891689469044026e01,
            0.1040931626310454794079799e01,
            0.9652383295306942866661884e00,
            0.8895458882533946571137358e00,
            0.8138545700535261740447950e00,
            0.7381647473570304814395029e00,
            0.6624769578126105498149624e00,
            0.5867920109947446493391737e00,
            0.5111111891461744489290992e00,
            0.4354366553151050147918632e00,
            0.3597723703299625354660452e00,
            0.2841264494060559943920389e00,
            0.2085185052177154996230005e00,
            0.1330107089065635461375419e00,
            0.5794620170990797798650123e-01,
        ]
    )
    OddThetaZero21 = np.array(
        [
            0.1498580583401444174317386e01,
            0.1426364890228584522673414e01,
            0.1354149299629923281192036e01,
            0.1281933868420423988034246e01,
            0.1209718660626713399048551e01,
            0.1137503750956414845248481e01,
            0.1065289229411733880607916e01,
            0.9930752076949068878557126e00,
            0.9208618284397049456535757e00,
            0.8486492789905562098591586e00,
            0.7764378127156926158031943e00,
            0.7042277832708635930867344e00,
            0.6320197021480767602848178e00,
            0.5598143404345395912377042e00,
            0.4876129202946139420188428e00,
            0.4154175043169533365541148e00,
            0.3432318703096418027524597e00,
            0.2710637595435203246492797e00,
            0.1989318822110657561806962e00,
            0.1268955503926593166308254e00,
            0.5528212871240371048241379e-01,
        ]
    )
    OddThetaZero22 = np.array(
        [
            0.1501754508594837337089856e01,
            0.1432712730475143340404518e01,
            0.1363671034069754274950592e01,
            0.1294629464249430679064317e01,
            0.1225588071083248538559259e01,
            0.1156546912269029268686830e01,
            0.1087506056298747798071893e01,
            0.1018465586752840651469411e01,
            0.9494256083335850798964741e00,
            0.8803862556198167553278643e00,
            0.8113477061841624760598814e00,
            0.7423102009244498727845341e00,
            0.6732740767851639064676858e00,
            0.6042398217472142478598295e00,
            0.5352081720899522889584566e00,
            0.4661802954366277026594659e00,
            0.3971581629712621730826920e00,
            0.3281453857685808451825081e00,
            0.2591493642052661979197670e00,
            0.1901879854885491785792565e00,
            0.1213179541186130699071317e00,
            0.5285224511635143601147552e-01,
        ]
    )
    OddThetaZero23 = np.array(
        [
            0.1504661202517196460191540e01,
            0.1438526110541037227495230e01,
            0.1372391084315255737540026e01,
            0.1306256159670931796771616e01,
            0.1240121376243315949825014e01,
            0.1173986779205849344923421e01,
            0.1107852421486856229076325e01,
            0.1041718366715156747157745e01,
            0.9755846932657442605621389e00,
            0.9094514999854931965227238e00,
            0.8433189145364798253029042e00,
            0.7771871059265138564989363e00,
            0.7110563039566125173946002e00,
            0.6449268305419475123120585e00,
            0.5787991523675322133651034e00,
            0.5126739740395088296453592e00,
            0.4465524134105889084933393e00,
            0.3804363581140941600870992e00,
            0.3143292666717729726674543e00,
            0.2482382273986418438740754e00,
            0.1821803739336923550363257e00,
            0.1162100228791666307841708e00,
            0.5062697144246344520692308e-01,
        ]
    )
    OddThetaZero24 = np.array(
        [
            0.1507333049739684406957329e01,
            0.1443869798951040686809862e01,
            0.1380406601553595646811530e01,
            0.1316943486448336467960940e01,
            0.1253480485358734060913055e01,
            0.1190017634088428795118215e01,
            0.1126554974102287077081806e01,
            0.1063092554588577221978254e01,
            0.9996304352342330000643921e00,
            0.9361686900661624628632729e00,
            0.8727074129127595264965883e00,
            0.8092467253835331800652228e00,
            0.7457867888716805068068402e00,
            0.6823278231980088937854296e00,
            0.6188701366516795329577182e00,
            0.5554141765061554178407906e00,
            0.4919606183965743300387332e00,
            0.4285105345527885639657014e00,
            0.3650657359209552112046854e00,
            0.3016295408979540017854803e00,
            0.2382087510453128743250072e00,
            0.1748198074104535338147956e00,
            0.1115148317291502081079519e00,
            0.4858150828905663931389750e-01,
        ]
    )
    OddThetaZero25 = np.array(
        [
            0.1509797405521643600800862e01,
            0.1448798505784201776188819e01,
            0.1387799649767640868379247e01,
            0.1326800860997572277878513e01,
            0.1265802165120213614545418e01,
            0.1204803590828283748583827e01,
            0.1143805171007496028164312e01,
            0.1082806944206958485218487e01,
            0.1021808956582037259849130e01,
            0.9608112645303606832220554e00,
            0.8998139383584991342974664e00,
            0.8388170675106567024157190e00,
            0.7778207682214244793380700e00,
            0.7168251950382156442798800e00,
            0.6558305587295081487906238e00,
            0.5948371551492265376377962e00,
            0.5338454137827292925154468e00,
            0.4728559836463229599006206e00,
            0.4118698949811841042358258e00,
            0.3508888880839026413717319e00,
            0.2899161521835467942607342e00,
            0.2289582244272697168835150e00,
            0.1680309071251709912058722e00,
            0.1071842976730454709494914e00,
            0.4669490825917857848258897e-01,
        ]
    )
    OddThetaZero26 = np.array(
        [
            0.1512077535592702651885542e01,
            0.1453358762182399391553360e01,
            0.1394640024852448295479492e01,
            0.1335921342914185177270306e01,
            0.1277202737290683500323248e01,
            0.1218484231207691826029908e01,
            0.1159765851037557179133987e01,
            0.1101047627365156083369632e01,
            0.1042329596373083545617043e01,
            0.9836118016874520301049009e00,
            0.9248942968954766185908511e00,
            0.8661771490588063053774554e00,
            0.8074604437333368789787031e00,
            0.7487442923247565105494255e00,
            0.6900288431709550365296138e00,
            0.6313142987730108226833704e00,
            0.5726009435739572428629866e00,
            0.5138891906843943809444838e00,
            0.4551796645660731149033106e00,
            0.3964733566771858874923011e00,
            0.3377719420068963817561906e00,
            0.2790784903284342592940125e00,
            0.2203992941938221111139898e00,
            0.1617495649772923108686624e00,
            0.1031775271253784724197264e00,
            0.4494935602951385601335598e-01,
        ]
    )
    OddThetaZero27 = np.array(
        [
            0.1514193352804819997509006e01,
            0.1457590393617468793209691e01,
            0.1400987464419153080392546e01,
            0.1344384581184662080889348e01,
            0.1287781761126833878758488e01,
            0.1231179023218584237510462e01,
            0.1174576388822640925688125e01,
            0.1117973882475943676285829e01,
            0.1061371532893653466992815e01,
            0.1004769374285310770780417e01,
            0.9481674481184788854172919e00,
            0.8915658055327279211293483e00,
            0.8349645107156934027761499e00,
            0.7783636457331086848148917e00,
            0.7217633176118399859733190e00,
            0.6651636690166557413471029e00,
            0.6085648948549621671933311e00,
            0.5519672690500084950513985e00,
            0.4953711895788266953367288e00,
            0.4387772581729219934583483e00,
            0.3821864303519236078766179e00,
            0.3256003205491779498477363e00,
            0.2690218877324958059454348e00,
            0.2124571975249336244841297e00,
            0.1559209129891515317090843e00,
            0.9945952063842375053227931e-01,
            0.4332960406341033436157524e-01,
        ]
    )
    OddThetaZero28 = np.array(
        [
            0.1516162000094549207021851e01,
            0.1461527685790782385188426e01,
            0.1406893396579229558427657e01,
            0.1352259145769086826235918e01,
            0.1297624947629923059740243e01,
            0.1242990817790597917328601e01,
            0.1188356773715062198539162e01,
            0.1133722835287525783953663e01,
            0.1079089025551156002698850e01,
            0.1024455371662101389801169e01,
            0.9698219061474760364582928e00,
            0.9151886685974009713577537e00,
            0.8605557079864861100238346e00,
            0.8059230859253162466918892e00,
            0.7512908813164713594661588e00,
            0.6966591971861112012613682e00,
            0.6420281709850565965229799e00,
            0.5873979906122764301937499e00,
            0.5327689202536826556885353e00,
            0.4781413438508069051295597e00,
            0.4235158420269503798571552e00,
            0.3688933369002844229314675e00,
            0.3142753865947702189467806e00,
            0.2596648470121556361200229e00,
            0.2050675726616484232653526e00,
            0.1504977164639767777858359e00,
            0.9600014792058154736462106e-01,
            0.4182252607645932321862773e-01,
        ]
    )
    OddThetaZero29 = np.array(
        [
            0.1517998315905975681819213e01,
            0.1465200315462026532129551e01,
            0.1412402336143180968579639e01,
            0.1359604389111228213837104e01,
            0.1306806486279734731351497e01,
            0.1254008640622089183072742e01,
            0.1201210866535131048800458e01,
            0.1148413180281179970113571e01,
            0.1095615600538999408768381e01,
            0.1042818149105710558651372e01,
            0.9900208518088600875617620e00,
            0.9372237397138955502862203e00,
            0.8844268507524555199840381e00,
            0.8316302319600398731649744e00,
            0.7788339426133210890795576e00,
            0.7260380587255163256281298e00,
            0.6732426796448045921910045e00,
            0.6204479380061240544867289e00,
            0.5676540152134466427854705e00,
            0.5148611664077887834613451e00,
            0.4620697624728053757183766e00,
            0.4092803643735033357684553e00,
            0.3564938631002461237979451e00,
            0.3037117642790043703921396e00,
            0.2509368276982060978106092e00,
            0.1981747109679032915697317e00,
            0.1454390911823840643137232e00,
            0.9277332955453467429763451e-01,
            0.4041676055113025684436480e-01,
        ]
    )
    OddThetaZero30 = np.array(
        [
            0.1519715208823086817411929e01,
            0.1468634099702062550682430e01,
            0.1417553008469014674939490e01,
            0.1366471944542347269659860e01,
            0.1315390917933946912760115e01,
            0.1264309939489363760018555e01,
            0.1213229021168654147755139e01,
            0.1162148176384137345494752e01,
            0.1111067420416500738111992e01,
            0.1059986770938296676746064e01,
            0.1008906248685091746434581e01,
            0.9578258783312407255956784e00,
            0.9067456896525242756445150e00,
            0.8556657190967860708153477e00,
            0.8045860119448479090873824e00,
            0.7535066253423996943740445e00,
            0.7024276326462752642452137e00,
            0.6513491298057893513225544e00,
            0.6002712449887427739045163e00,
            0.5491941535583390603837715e00,
            0.4981181022276018128369963e00,
            0.4470434496975185070560821e00,
            0.3959707385770101868486847e00,
            0.3449008307748737032772825e00,
            0.2938351828535981363494671e00,
            0.2427764647581323719392653e00,
            0.1917301500230701193408602e00,
            0.1407094708800750523796875e00,
            0.8975637836633630394302762e-01,
            0.3910242380354419363081899e-01,
        ]
    )
    OddThetaZero31 = np.array(
        [
            0.1521323961422700444944464e01,
            0.1471851603590422118622546e01,
            0.1422379260986849454727777e01,
            0.1372906941604798453293218e01,
            0.1323434653909307929892118e01,
            0.1273962407026590487708892e01,
            0.1224490210963055761921526e01,
            0.1175018076866133593082748e01,
            0.1125546017342156230227131e01,
            0.1076074046851682267877939e01,
            0.1026602182210094558879809e01,
            0.9771304432322302018639612e00,
            0.9276588535760335871906045e00,
            0.8781874418647315968408864e00,
            0.8287162432047307488040550e00,
            0.7792453012756761070555010e00,
            0.7297746712644485550469075e00,
            0.6803044240724808212528033e00,
            0.6308346524943159683026367e00,
            0.5813654805388740483542438e00,
            0.5318970779332963132260134e00,
            0.4824296835154055410257004e00,
            0.4329636445908698102350729e00,
            0.3834994865870752458854056e00,
            0.3340380441799942088370002e00,
            0.2845807279748544733570760e00,
            0.2351301237470960623526672e00,
            0.1856915325646991222655151e00,
            0.1362777698319134965765757e00,
            0.8692946525012054120187353e-01,
            0.3787087726949234365520114e-01,
        ]
    )
    OddThetaZero32 = np.array(
        [
            0.1522834478472358672931947e01,
            0.1474872636605138418026177e01,
            0.1426910807768284322082436e01,
            0.1378948998781055367310047e01,
            0.1330987216841224680164684e01,
            0.1283025469674968454386883e01,
            0.1235063765709222885799986e01,
            0.1187102114275073728898860e01,
            0.1139140525853183114841234e01,
            0.1091179012375759666645271e01,
            0.1043217587604604879578741e01,
            0.9952562676120370548458597e00,
            0.9472950714021223337048082e00,
            0.8993340217254078241758816e00,
            0.8513731461641338285808219e00,
            0.8034124786014693431693904e00,
            0.7554520612457602368887930e00,
            0.7074919474732165281510693e00,
            0.6595322059052657580628641e00,
            0.6115729263971504325174172e00,
            0.5636142290734363894767612e00,
            0.5156562783879918167717991e00,
            0.4676993058012953469089537e00,
            0.4197436479350834076514896e00,
            0.3717898140987174444032373e00,
            0.3238386134116156886828960e00,
            0.2758914133405791810724762e00,
            0.2279507206431424610498769e00,
            0.1800216744637006612298520e00,
            0.1321166988439841543825694e00,
            0.8427518284958235696897899e-01,
            0.3671453742186897322954009e-01,
        ]
    )
    OddThetaZero33 = np.array(
        [
            0.1524255491013576804195881e01,
            0.1477714660784952783237945e01,
            0.1431173841758652772349485e01,
            0.1384633039781787069436630e01,
            0.1338092261006965672253841e01,
            0.1291551512012124788593875e01,
            0.1245010799937299944123195e01,
            0.1198470132644670409416924e01,
            0.1151929518909907204916554e01,
            0.1105388968655282680015213e01,
            0.1058848493238442193822372e01,
            0.1012308105815651361079674e01,
            0.9657678218054126734684090e00,
            0.9192276594886802366068293e00,
            0.8726876407972167893294764e00,
            0.8261477923647281669131478e00,
            0.7796081469509049827753598e00,
            0.7330687454042532567721262e00,
            0.6865296394193009886613469e00,
            0.6399908954920466591029822e00,
            0.5934526007301573325059582e00,
            0.5469148716199143611697357e00,
            0.5003778676688561814362271e00,
            0.4538418134105091550464446e00,
            0.4073070354279485829740435e00,
            0.3607740278788822846227453e00,
            0.3142435758510728338330843e00,
            0.2677170062389944640113953e00,
            0.2211967514739567668334169e00,
            0.1746877983807874325844051e00,
            0.1282022028383479964348629e00,
            0.8177818680168764430245080e-01,
            0.3562671947817428176226631e-01,
        ]
    )
    OddThetaZero34 = np.array(
        [
            0.1525594725214770881206476e01,
            0.1480393128432045740356817e01,
            0.1435191541323085582529217e01,
            0.1389989968924959812091252e01,
            0.1344788416522907866060817e01,
            0.1299586889746827997174554e01,
            0.1254385394680661996389736e01,
            0.1209183937989395175829969e01,
            0.1163982527069600127515982e01,
            0.1118781170231154762473596e01,
            0.1073579876920155012130433e01,
            0.1028378657996412636748477e01,
            0.9831775260837211038023103e00,
            0.9379764960179657076015136e00,
            0.8927755854282048597997986e00,
            0.8475748155007347757967789e00,
            0.8023742119985848209905761e00,
            0.7571738066433708695662393e00,
            0.7119736390205872251796930e00,
            0.6667737592565460745639184e00,
            0.6215742318591892056934095e00,
            0.5763751413603713322640298e00,
            0.5311766008298875656047892e00,
            0.4859787651249621588538330e00,
            0.4407818522612533891543536e00,
            0.3955861793705505114602136e00,
            0.3503922263398633798966312e00,
            0.3052007556167344348049303e00,
            0.2600130558662051177480644e00,
            0.2148314894784555841956251e00,
            0.1696608997322034095150907e00,
            0.1245129955389270002683579e00,
            0.7942489891978153749097006e-01,
            0.3460150809198016850782325e-01,
        ]
    )
    OddThetaZero35 = np.array(
        [
            0.1526859042890589526378487e01,
            0.1482921763148403842276533e01,
            0.1438984491795164536567108e01,
            0.1395047233189252525231459e01,
            0.1351109991891878034957302e01,
            0.1307172772745304669260382e01,
            0.1263235580960968906699379e01,
            0.1219298422221050703835127e01,
            0.1175361302797916700875697e01,
            0.1131424229697065895207730e01,
            0.1087487210830887883186060e01,
            0.1043550255232887174273672e01,
            0.9996133733253190253881393e00,
            0.9556765772578535710715874e00,
            0.9117398813415957196221754e00,
            0.8678033026125661948850687e00,
            0.8238668615732247310812836e00,
            0.7799305831824293601507400e00,
            0.7359944981977457886183921e00,
            0.6920586450266629333858465e00,
            0.6481230723279649663476697e00,
            0.6041878427445000852182582e00,
            0.5602530383870993537615272e00,
            0.5163187691099712879003757e00,
            0.4723851853891499571797438e00,
            0.4284524990953311047063058e00,
            0.3845210184454249891341793e00,
            0.3405912098612419399584605e00,
            0.2966638144233703899038032e00,
            0.2527400847124078576715667e00,
            0.2088223170017057788708674e00,
            0.1649152190599722827308055e00,
            0.1210301722471160155498167e00,
            0.7720326018898817828206987e-01,
            0.3363364974516995102167462e-01,
        ]
    )
    OddThetaZero36 = np.array(
        [
            0.1528054559083405137047563e01,
            0.1485312794997097705446883e01,
            0.1442571038214470776579613e01,
            0.1399829292522320013570493e01,
            0.1357087561874166765548658e01,
            0.1314345850454078759228779e01,
            0.1271604162748143389685638e01,
            0.1228862503626296926524085e01,
            0.1186120878437839368895715e01,
            0.1143379293124832099340074e01,
            0.1100637754358770795248912e01,
            0.1057896269707576280860569e01,
            0.1015154847842238156769126e01,
            0.9724134987956584339974640e00,
            0.9296722342907946818431047e00,
            0.8869310681617368097575324e00,
            0.8441900169008687429295884e00,
            0.8014491003793523040286325e00,
            0.7587083428093935362576859e00,
            0.7159677740493625646700975e00,
            0.6732274314040501413860867e00,
            0.6304873621547357085895928e00,
            0.5877476271899241333221832e00,
            0.5450083063396327078463020e00,
            0.5022695064252395155059223e00,
            0.4595313737871711838065652e00,
            0.4167941144922007176438387e00,
            0.3740580283336802289311736e00,
            0.3313235690067746553700419e00,
            0.2885914573933480330041531e00,
            0.2458629119584249278750153e00,
            0.2031401664615301668533461e00,
            0.1604278005405711652039491e00,
            0.1177368858339244458607172e00,
            0.7510252408650086658441596e-01,
            0.3271846270775478856070884e-01,
        ]
    )
    OddThetaZero37 = np.array(
        [
            0.1529186740959505109653289e01,
            0.1487577158293388707508111e01,
            0.1445967582009979387718202e01,
            0.1404358015412336440816745e01,
            0.1362748461941311565399969e01,
            0.1321138925227929972823825e01,
            0.1279529409151733277951100e01,
            0.1237919917907156982173977e01,
            0.1196310456080472987418488e01,
            0.1154701028740456700905269e01,
            0.1113091641546798022997704e01,
            0.1071482300881451340842721e01,
            0.1029873014009735917989475e01,
            0.9882637892802373569245916e00,
            0.9466546363756944528310758e00,
            0.9050455666314926033869497e00,
            0.8634365934447506520344540e00,
            0.8218277328062565148449433e00,
            0.7802190040012226850703573e00,
            0.7386104305454957746359112e00,
            0.6970020414556031913861513e00,
            0.6553938730008771105113861e00,
            0.6137859711661063914322283e00,
            0.5721783951857430999179669e00,
            0.5305712227365694155165922e00,
            0.4889645577740232855661796e00,
            0.4473585427277744403333139e00,
            0.4057533781735039172217875e00,
            0.3641493559322687127223795e00,
            0.3225469176515179389138545e00,
            0.2809467650889194227770571e00,
            0.2393500844055270891104500e00,
            0.1977590501629603151642330e00,
            0.1561781206604067112364815e00,
            0.1146180742271483316267615e00,
            0.7311308274978660184520447e-01,
            0.3185176130791400787169333e-01,
        ]
    )
    OddThetaZero38 = np.array(
        [
            0.1530260491394766313570510e01,
            0.1489724658775115137266557e01,
            0.1449188831753177403184250e01,
            0.1408653013220734918131897e01,
            0.1368117206184034069757975e01,
            0.1327581413807020172726043e01,
            0.1287045639459248683416470e01,
            0.1246509886770075360771330e01,
            0.1205974159691064843702080e01,
            0.1165438462569017837684371e01,
            0.1124902800232641860108690e01,
            0.1084367178096737546333982e01,
            0.1043831602288925654461738e01,
            0.1003296079805520648394589e01,
            0.9627606187053432591598275e00,
            0.9222252283533212928777294e00,
            0.8816899197300536544215117e00,
            0.8411547058297167490229198e00,
            0.8006196021777239861105672e00,
            0.7600846275129127849719743e00,
            0.7195498046991648764002008e00,
            0.6790151619622966197448464e00,
            0.6384807345966275863946969e00,
            0.5979465673637771458800754e00,
            0.5574127179353942966481713e00,
            0.5168792619515766918187819e00,
            0.4763463006547495614450433e00,
            0.4358139727703203523144583e00,
            0.3952824736706231680817472e00,
            0.3547520876199503791717895e00,
            0.3142232448436673832093046e00,
            0.2736966289659020439688229e00,
            0.2331733955144496369946707e00,
            0.1926556629116315949109922e00,
            0.1521477743835989472840536e00,
            0.1116602300918232453371161e00,
            0.7122632005925390425640031e-01,
            0.3102979192734513847869512e-01,
        ]
    )
    OddThetaZero39 = np.array(
        [
            0.1531280219945530918862887e01,
            0.1491764115543711582608611e01,
            0.1452248016067723206269747e01,
            0.1412731924058150689920340e01,
            0.1373215842151127100608219e01,
            0.1333699773114208203180680e01,
            0.1294183719885939986844436e01,
            0.1254667685620366764206205e01,
            0.1215151673737978313041594e01,
            0.1175635687984935008823566e01,
            0.1136119732502868295328130e01,
            0.1096603811912170337964094e01,
            0.1057087931412518476253055e01,
            0.1017572096905509113863513e01,
            0.9780563151458206514020694e00,
            0.9385405939294598498464432e00,
            0.8990249423306286659734381e00,
            0.8595093710029676968794279e00,
            0.8199938925669823205931282e00,
            0.7804785221142635001130339e00,
            0.7409632778721439753645470e00,
            0.7014481820920565808095373e00,
            0.6619332622550151160558599e00,
            0.6224185527349885679868616e00,
            0.5829040971371158016902326e00,
            0.5433899516536147244946347e00,
            0.5038761899947552937603140e00,
            0.4643629108305196256509391e00,
            0.4248502493722176391609139e00,
            0.3853383960541810555628366e00,
            0.3458276279674767760058527e00,
            0.3063183644932167228808922e00,
            0.2668112720373341483108662e00,
            0.2273074770384765519559169e00,
            0.1878090446069578429818381e00,
            0.1483202086882449059764783e00,
            0.1088512052741322662621244e00,
            0.6943448689600673838300180e-01,
            0.3024917865720923179577363e-01,
        ]
    )
    OddThetaZero40 = np.array(
        [
            0.1532249903371281818085917e01,
            0.1493703482108998740614827e01,
            0.1455157065195200346809599e01,
            0.1416610654869340270223431e01,
            0.1378064253450997606022340e01,
            0.1339517863369794055919890e01,
            0.1300971487198245305453001e01,
            0.1262425127688525048786896e01,
            0.1223878787814308166546501e01,
            0.1185332470819113339105535e01,
            0.1146786180272904774996439e01,
            0.1108239920139165765487189e01,
            0.1069693694855262920629379e01,
            0.1031147509429735196850587e01,
            0.9926013695612463310882198e00,
            0.9540552817854489715123326e00,
            0.9155092536580933534978986e00,
            0.8769632939856246206308699e00,
            0.8384174131186299393233148e00,
            0.7998716233293992826192237e00,
            0.7613259393034545837300323e00,
            0.7227803787876118667749166e00,
            0.6842349634562860931661901e00,
            0.6456897200871628101751519e00,
            0.6071446821835496233813653e00,
            0.5685998922550279415939221e00,
            0.5300554050908430047380815e00,
            0.4915112925697217767572364e00,
            0.4529676509187802579380104e00,
            0.4144246120108054286121629e00,
            0.3758823615873930314093573e00,
            0.3373411699211847213003570e00,
            0.2988014460838619282843952e00,
            0.2602638401106843145315994e00,
            0.2217294507811754336425535e00,
            0.1832002925124018168986342e00,
            0.1446804953347050655563166e00,
            0.1061800440374660771048480e00,
            0.6773059476567831336488402e-01,
            0.2950687695527422224851832e-01,
        ]
    )
    OddThetaZero41 = np.array(
        [
            0.1533173137460634461235066e01,
            0.1495549950040734249895393e01,
            0.1457926766471340970762709e01,
            0.1420303588732694442267846e01,
            0.1382680418872520759663065e01,
            0.1345057259031099988676433e01,
            0.1307434111468678960501903e01,
            0.1269810978596001635506341e01,
            0.1232187863008871596257323e01,
            0.1194564767527851916244615e01,
            0.1156941695244461089553108e01,
            0.1119318649575559636172662e01,
            0.1081695634328067979412755e01,
            0.1044072653776750930510111e01,
            0.1006449712758602402124214e01,
            0.9688268167884441336521039e00,
            0.9312039722018274751677424e00,
            0.8935811863333633591901354e00,
            0.8559584677414483220357356e00,
            0.8183358264943738874932307e00,
            0.7807132745385688213392421e00,
            0.7430908261781095392995681e00,
            0.7054684987070400358784448e00,
            0.6678463132547297820965882e00,
            0.6302242959332082940279826e00,
            0.5926024794204980683238426e00,
            0.5549809051864955237054951e00,
            0.5173596266878257139037738e00,
            0.4797387140623364241772428e00,
            0.4421182612140318859822955e00,
            0.4044983968396638104610711e00,
            0.3668793022152994411560368e00,
            0.3292612411240570212440856e00,
            0.2916446128242035199998930e00,
            0.2540300517665934607689814e00,
            0.2164186303985620085027010e00,
            0.1788123148742007754778852e00,
            0.1412151362884411752920148e00,
            0.1036368402634645114775150e00,
            0.6610832470916409695729856e-01,
            0.2880013396280840229218334e-01,
        ]
    )
    OddThetaZero42 = np.array(
        [
            0.1534053181584449084854269e01,
            0.1497310038074501005766978e01,
            0.1460566897984002644464183e01,
            0.1423823763069232867789940e01,
            0.1387080635143547965139117e01,
            0.1350337516098480646600889e01,
            0.1313594407926723776707731e01,
            0.1276851312747612578778902e01,
            0.1240108232835827171448969e01,
            0.1203365170654181674634873e01,
            0.1166622128891556900019450e01,
            0.1129879110507284864268762e01,
            0.1093136118783624474731954e01,
            0.1056393157388405786003046e01,
            0.1019650230450503114577901e01,
            0.9829073426515786199715451e00,
            0.9461644993385942743541994e00,
            0.9094217066630320811486021e00,
            0.8726789717547518480094350e00,
            0.8359363029411928866147184e00,
            0.7991937100265521402467844e00,
            0.7624512046511992388808212e00,
            0.7257088007597790982554974e00,
            0.6889665152185689899094400e00,
            0.6522243686409073299342467e00,
            0.6154823865075504175164916e00,
            0.5787406007128420496638175e00,
            0.5419990517384125648087865e00,
            0.5052577917731960988645056e00,
            0.4685168892980173635234519e00,
            0.4317764360047099160222576e00,
            0.3950365575646972937604113e00,
            0.3582974309994310205507555e00,
            0.3215593139080007759227897e00,
            0.2848225961961619069649047e00,
            0.2480878974611689122432227e00,
            0.2113562650517154915467591e00,
            0.1746296191183898065201571e00,
            0.1379118964507339113271975e00,
            0.1012126146941469342401701e00,
            0.6456194899726137278760257e-01,
            0.2812645439079299219187419e-01,
        ]
    )
    OddThetaZero43 = np.array(
        [
            0.1534892997139557227614279e01,
            0.1498989668998897501276994e01,
            0.1463086343903285773505644e01,
            0.1427183023414814244376429e01,
            0.1391279709144040438287602e01,
            0.1355376402767821814937864e01,
            0.1319473106048673173924451e01,
            0.1283569820856137399848247e01,
            0.1247666549190742942502495e01,
            0.1211763293211231530413995e01,
            0.1175860055265884319525693e01,
            0.1139956837928964066704190e01,
            0.1104053644043538840797350e01,
            0.1068150476772278342447444e01,
            0.1032247339658243608912598e01,
            0.9963442366982618328585493e00,
            0.9604411724322426430586723e00,
            0.9245381520528253319358567e00,
            0.8886351815411563067305273e00,
            0.8527322678365406966379400e00,
            0.8168294190504262166761188e00,
            0.7809266447390142664026016e00,
            0.7450239562542930157586944e00,
            0.7091213672012904985883029e00,
            0.6732188940411854039055226e00,
            0.6373165568977466866717861e00,
            0.6014143806519714388063208e00,
            0.5655123964528129857845238e00,
            0.5296106438411039715966193e00,
            0.4937091737981756577948229e00,
            0.4578080532255782153525438e00,
            0.4219073717059785387344039e00,
            0.3860072520255396095683859e00,
            0.3501078671472635990145335e00,
            0.3142094687704932495909488e00,
            0.2783124378775218384923333e00,
            0.2424173798924625361874772e00,
            0.2065253182141071551836492e00,
            0.1706381290938671641708352e00,
            0.1347596593282315198612592e00,
            0.9889920900871122533586553e-01,
            0.6308626356388784057588631e-01,
            0.2748357108440508277394892e-01,
        ]
    )
    OddThetaZero44 = np.array(
        [
            0.1535695280838629983064694e01,
            0.1500594236235067817656313e01,
            0.1465493194350303789230585e01,
            0.1430392156577492526495371e01,
            0.1395291124351096810858349e01,
            0.1360190099162024176252063e01,
            0.1325089082574000089379322e01,
            0.1289988076241572027256558e01,
            0.1254887081930202663795858e01,
            0.1219786101538994920367859e01,
            0.1184685137126702182946916e01,
            0.1149584190941820846004092e01,
            0.1114483265457749469332035e01,
            0.1079382363414242848588494e01,
            0.1044281487866708939888712e01,
            0.1009180642245317812316634e01,
            0.9740798304264509422659935e00,
            0.9389790568197674899837757e00,
            0.9038783264751749171405213e00,
            0.8687776452153701566068907e00,
            0.8336770198015192229270720e00,
            0.7985764581422970742698971e00,
            0.7634759695602610192254430e00,
            0.7283755651349081311799055e00,
            0.6932752581495918103871962e00,
            0.6581750646810479477537782e00,
            0.6230750043877162525265513e00,
            0.5879751015798285374141491e00,
            0.5528753866962970290878822e00,
            0.5177758983811020490994086e00,
            0.4826766864637186565865902e00,
            0.4475778163386701445336738e00,
            0.4124793755752883735361361e00,
            0.3773814842049053432591527e00,
            0.3422843113148581639684411e00,
            0.3071881029697497767338606e00,
            0.2720932316284942932084102e00,
            0.2370002891767127567222407e00,
            0.2019102761348421810146637e00,
            0.1668250268181992892198073e00,
            0.1317483020532982541977987e00,
            0.9668919410176593344830717e-01,
            0.6167652949817792358742135e-01,
            0.2686941953400762687915995e-01,
        ]
    )
    OddThetaZero45 = np.array(
        [
            0.1536462493634653558154673e01,
            0.1502128661685489464262068e01,
            0.1467794832169950298839286e01,
            0.1433461006333747476463744e01,
            0.1399127185457927306909792e01,
            0.1364793370871767472755746e01,
            0.1330459563966682507229700e01,
            0.1296125766211456950047804e01,
            0.1261791979169174443914592e01,
            0.1227458204516276373551266e01,
            0.1193124444064268679881601e01,
            0.1158790699784705540565424e01,
            0.1124456973838220917338613e01,
            0.1090123268608563332983681e01,
            0.1055789586742829027889885e01,
            0.1021455931199402224481903e01,
            0.9871223053055240306873772e00,
            0.9527887128269590857605072e00,
            0.9184551580529615732874848e00,
            0.8841216459007313197085517e00,
            0.8497881820448998068986446e00,
            0.8154547730794463756650640e00,
            0.7811214267220410983907210e00,
            0.7467881520744805288579630e00,
            0.7124549599581423848996086e00,
            0.6781218633510390484774053e00,
            0.6437888779643722276961833e00,
            0.6094560230135452763170614e00,
            0.5751233222647905281576305e00,
            0.5407908054797110156395425e00,
            0.5064585104462232763044121e00,
            0.4721264858937837018545325e00,
            0.4377947957771643018072936e00,
            0.4034635257416918872885646e00,
            0.3691327931855416440777167e00,
            0.3348027634909946151567752e00,
            0.3004736773353657517478163e00,
            0.2661458990278703974149616e00,
            0.2318200075085118064771005e00,
            0.1974969814205034596217949e00,
            0.1631786149772797106698111e00,
            0.1288685867945150272796250e00,
            0.9457579039019365184018477e-01,
            0.6032842220945916819748797e-01,
            0.2628211572883546008386342e-01,
        ]
    )
    OddThetaZero46 = np.array(
        [
            0.1537196885933572311910085e01,
            0.1503597446159129663218426e01,
            0.1469998008568304160871417e01,
            0.1436398574277729377094190e01,
            0.1402799144434368418084898e01,
            0.1369199720226542342210552e01,
            0.1335600302895785666466344e01,
            0.1302000893749787833197270e01,
            0.1268401494176718217187838e01,
            0.1234802105661283063077015e01,
            0.1201202729802928570677208e01,
            0.1167603368336689119731474e01,
            0.1134004023157288628212183e01,
            0.1100404696347243386055709e01,
            0.1066805390209896030700280e01,
            0.1033206107308545748870827e01,
            0.9996068505131472996246808e00,
            0.9660076230564559666956844e00,
            0.9324084286020318648375284e00,
            0.8988092713272342656758064e00,
            0.8652101560253048754543914e00,
            0.8316110882319595313175680e00,
            0.7980120743837286767240920e00,
            0.7644131220178278512956878e00,
            0.7308142400269308414762795e00,
            0.6972154389873656296775555e00,
            0.6636167315867435235683334e00,
            0.6300181331881122315700717e00,
            0.5964196625844131090385918e00,
            0.5628213430226633060114218e00,
            0.5292232036175455988770345e00,
            0.4956252813388603291268380e00,
            0.4620276238643496856689332e00,
            0.4284302937718012609061760e00,
            0.3948333748659561479104270e00,
            0.3612369820255324850503899e00,
            0.3276412770872600895283016e00,
            0.2940464955725917238672137e00,
            0.2604529939906062681054034e00,
            0.2268613388903867245835696e00,
            0.1932724879746856807613294e00,
            0.1596881970714452359218090e00,
            0.1261120661032951679792394e00,
            0.9255279834764232165670211e-01,
            0.5903798711627596210077655e-01,
            0.2571993685288741305807485e-01,
        ]
    )
    OddThetaZero47 = np.array(
        [
            0.1537900519639177351485509e01,
            0.1505004713461118831562885e01,
            0.1472108909246876714959093e01,
            0.1439213107999753788389740e01,
            0.1406317310749171844429399e01,
            0.1373421518560135827011261e01,
            0.1340525732543378926238078e01,
            0.1307629953866399941713193e01,
            0.1274734183765634621652739e01,
            0.1241838423560042429086765e01,
            0.1208942674666441461574345e01,
            0.1176046938616989970936899e01,
            0.1143151217079296988032361e01,
            0.1110255511879752164142939e01,
            0.1077359825030803093319347e01,
            0.1044464158763086533573607e01,
            0.1011568515563550961534623e01,
            0.9786728982210094134846821e00,
            0.9457773098809579873452675e00,
            0.9128817541120207886160886e00,
            0.8799862349870845994262022e00,
            0.8470907571831347751008168e00,
            0.8141953261050969543263697e00,
            0.7812999480407721032432037e00,
            0.7484046303564402266425896e00,
            0.7155093817462244075628281e00,
            0.6826142125533466396520346e00,
            0.6497191351887403950357432e00,
            0.6168241646833332909207560e00,
            0.5839293194266532352434130e00,
            0.5510346221695150324949297e00,
            0.5181401014079633610544426e00,
            0.4852457933290632607369490e00,
            0.4523517446039431388003505e00,
            0.4194580164920722423612656e00,
            0.3865646910356375140534892e00,
            0.3536718807003189971969294e00,
            0.3207797439266498255416416e00,
            0.2878885112969848452450724e00,
            0.2549985318477515756044100e00,
            0.2221103602568508117102717e00,
            0.1892249341643785313168465e00,
            0.1563439726212394862316010e00,
            0.1234710001537068179882843e00,
            0.9061453776736619019094845e-01,
            0.5780160090309369034797044e-01,
            0.2518130440638251656980999e-01,
        ]
    )
    OddThetaZero48 = np.array(
        [
            0.1538575287485045780713568e01,
            0.1506354249056545799167351e01,
            0.1474133212398093554231315e01,
            0.1441912178413208451704314e01,
            0.1409691148027973881079186e01,
            0.1377470122199186272616473e01,
            0.1345249101923067139210221e01,
            0.1313028088244711409410919e01,
            0.1280807082268469343020428e01,
            0.1248586085169490583375238e01,
            0.1216365098206699074213627e01,
            0.1184144122737518830558069e01,
            0.1151923160234735793613503e01,
            0.1119702212305964062886069e01,
            0.1087481280716290811591462e01,
            0.1055260367414810028339009e01,
            0.1023039474565930165787482e01,
            0.9908186045865674272211987e00,
            0.9585977601906320722299056e00,
            0.9263769444426036830464570e00,
            0.8941561608225061952846411e00,
            0.8619354133052817812042663e00,
            0.8297147064584916186566054e00,
            0.7974940455635382827549679e00,
            0.7652734367673509855003551e00,
            0.7330528872739117793257283e00,
            0.7008324055884451343305450e00,
            0.6686120018320298047193041e00,
            0.6363916881515750209117372e00,
            0.6041714792607289968809847e00,
            0.5719513931632926368357825e00,
            0.5397314521353000325496229e00,
            0.5075116840805377280486923e00,
            0.4752921244363891783961832e00,
            0.4430728189095547215892704e00,
            0.4108538274961112658763390e00,
            0.3786352305487998074788803e00,
            0.3464171382200184643128623e00,
            0.3141997056941599156198233e00,
            0.2819831588178046655599196e00,
            0.2497678394619649260592757e00,
            0.2175542909210219972765731e00,
            0.1853434315961135904158300e00,
            0.1531369452704970394027659e00,
            0.1209382841678252589048669e00,
            0.8875579450016283173293810e-01,
            0.5661593754525190873771522e-01,
            0.2466476940450737058975552e-01,
        ]
    )
    OddThetaZero49 = np.array(
        [
            0.1539222930035210331902410e01,
            0.1507649534071729882214386e01,
            0.1476076139707032453353232e01,
            0.1444502747756546556830706e01,
            0.1412929359055252480197337e01,
            0.1381355974464721552102928e01,
            0.1349782594880622732927647e01,
            0.1318209221240839295255046e01,
            0.1286635854534357387243172e01,
            0.1255062495811112994872428e01,
            0.1223489146193015470717893e01,
            0.1191915806886406014313715e01,
            0.1160342479196260434661502e01,
            0.1128769164542510055952304e01,
            0.1097195864478936528824546e01,
            0.1065622580715200621234508e01,
            0.1034049315142698534418744e01,
            0.1002476069865111021377467e01,
            0.9709028472347329448081481e00,
            0.9393296498959608456620406e00,
            0.9077564808376970380335442e00,
            0.8761833434569334264096395e00,
            0.8446102416364528348063321e00,
            0.8130371798404960344077378e00,
            0.7814641632334840064334645e00,
            0.7498911978285964532098456e00,
            0.7183182906753955596314298e00,
            0.6867454500990591398232408e00,
            0.6551726860086246453663390e00,
            0.6236000102986843027011345e00,
            0.5920274373793840619034224e00,
            0.5604549848852622707385612e00,
            0.5288826746375584896948472e00,
            0.4973105339724571989307663e00,
            0.4657385976085971045914307e00,
            0.4341669103277770901346174e00,
            0.4025955309141879357899857e00,
            0.3710245380997234377015025e00,
            0.3394540398171456073906403e00,
            0.3078841881262277508367562e00,
            0.2763152043287541015913350e00,
            0.2447474234189502677044064e00,
            0.2131813777658572006989977e00,
            0.1816179673056091210434906e00,
            0.1500588419721174291665790e00,
            0.1185073845935281602210493e00,
            0.8697177361567243680812898e-01,
            0.5547793843128156580348541e-01,
            0.2416899936118312040170588e-01,
        ]
    )

    if l < 1 or 100 < l:
        print("")
        print("LEGENDRE_THETA - Fatal error!")
        print("  1 <= L <= 100 is required.")
        # exit ( 'LEGENDRE_THETA - Fatal error!' )
        # return -1

    lhalf = (l + 1) // 2

    if (l % 2) == 1:
        if lhalf < k:
            kcopy = k - lhalf
        elif lhalf == k:
            kcopy = lhalf
        else:
            kcopy = lhalf - k
    else:
        if lhalf < k:
            kcopy = k - lhalf
        else:
            kcopy = lhalf + 1 - k

    if kcopy < 1 or lhalf < kcopy:
        print("")
        print("LEGENDRE_THETA - Fatal error!")
        print("  1 <= K <= (L+1)/2 is required.")
        # exit ( 'LEGENDRE_THETA - Fatal error!' )
        # return -1
    #
    #  If L is odd, and K = ( L - 1 ) / 2, then it's easy.
    #
    if (l % 2) == 1 and kcopy == lhalf:
        theta = np.pi / 2.0
    elif l == 2:
        theta = EvenThetaZero1[kcopy - 1]
    elif l == 3:
        theta = OddThetaZero1[kcopy - 1]
    elif l == 4:
        theta = EvenThetaZero2[kcopy - 1]
    elif l == 5:
        theta = OddThetaZero2[kcopy - 1]
    elif l == 6:
        theta = EvenThetaZero3[kcopy - 1]
    elif l == 7:
        theta = OddThetaZero3[kcopy - 1]
    elif l == 8:
        theta = EvenThetaZero4[kcopy - 1]
    elif l == 9:
        theta = OddThetaZero4[kcopy - 1]
    elif l == 10:
        theta = EvenThetaZero5[kcopy - 1]
    elif l == 11:
        theta = OddThetaZero5[kcopy - 1]
    elif l == 12:
        theta = EvenThetaZero6[kcopy - 1]
    elif l == 13:
        theta = OddThetaZero6[kcopy - 1]
    elif l == 14:
        theta = EvenThetaZero7[kcopy - 1]
    elif l == 15:
        theta = OddThetaZero7[kcopy - 1]
    elif l == 16:
        theta = EvenThetaZero8[kcopy - 1]
    elif l == 17:
        theta = OddThetaZero8[kcopy - 1]
    elif l == 18:
        theta = EvenThetaZero9[kcopy - 1]
    elif l == 19:
        theta = OddThetaZero9[kcopy - 1]
    elif l == 20:
        theta = EvenThetaZero10[kcopy - 1]
    elif l == 21:
        theta = OddThetaZero10[kcopy - 1]
    elif l == 22:
        theta = EvenThetaZero11[kcopy - 1]
    elif l == 23:
        theta = OddThetaZero11[kcopy - 1]
    elif l == 24:
        theta = EvenThetaZero12[kcopy - 1]
    elif l == 25:
        theta = OddThetaZero12[kcopy - 1]
    elif l == 26:
        theta = EvenThetaZero13[kcopy - 1]
    elif l == 27:
        theta = OddThetaZero13[kcopy - 1]
    elif l == 28:
        theta = EvenThetaZero14[kcopy - 1]
    elif l == 29:
        theta = OddThetaZero14[kcopy - 1]
    elif l == 30:
        theta = EvenThetaZero15[kcopy - 1]
    elif l == 31:
        theta = OddThetaZero15[kcopy - 1]
    elif l == 32:
        theta = EvenThetaZero16[kcopy - 1]
    elif l == 33:
        theta = OddThetaZero16[kcopy - 1]
    elif l == 34:
        theta = EvenThetaZero17[kcopy - 1]
    elif l == 35:
        theta = OddThetaZero17[kcopy - 1]
    elif l == 36:
        theta = EvenThetaZero18[kcopy - 1]
    elif l == 37:
        theta = OddThetaZero18[kcopy - 1]
    elif l == 38:
        theta = EvenThetaZero19[kcopy - 1]
    elif l == 39:
        theta = OddThetaZero19[kcopy - 1]
    elif l == 40:
        theta = EvenThetaZero20[kcopy - 1]
    elif l == 41:
        theta = OddThetaZero20[kcopy - 1]
    elif l == 42:
        theta = EvenThetaZero21[kcopy - 1]
    elif l == 43:
        theta = OddThetaZero21[kcopy - 1]
    elif l == 44:
        theta = EvenThetaZero22[kcopy - 1]
    elif l == 45:
        theta = OddThetaZero22[kcopy - 1]
    elif l == 46:
        theta = EvenThetaZero23[kcopy - 1]
    elif l == 47:
        theta = OddThetaZero23[kcopy - 1]
    elif l == 48:
        theta = EvenThetaZero24[kcopy - 1]
    elif l == 49:
        theta = OddThetaZero24[kcopy - 1]
    elif l == 50:
        theta = EvenThetaZero25[kcopy - 1]
    elif l == 51:
        theta = OddThetaZero25[kcopy - 1]
    elif l == 52:
        theta = EvenThetaZero26[kcopy - 1]
    elif l == 53:
        theta = OddThetaZero26[kcopy - 1]
    elif l == 54:
        theta = EvenThetaZero27[kcopy - 1]
    elif l == 55:
        theta = OddThetaZero27[kcopy - 1]
    elif l == 56:
        theta = EvenThetaZero28[kcopy - 1]
    elif l == 57:
        theta = OddThetaZero28[kcopy - 1]
    elif l == 58:
        theta = EvenThetaZero29[kcopy - 1]
    elif l == 59:
        theta = OddThetaZero29[kcopy - 1]
    elif l == 60:
        theta = EvenThetaZero30[kcopy - 1]
    elif l == 61:
        theta = OddThetaZero30[kcopy - 1]
    elif l == 62:
        theta = EvenThetaZero31[kcopy - 1]
    elif l == 63:
        theta = OddThetaZero31[kcopy - 1]
    elif l == 64:
        theta = EvenThetaZero32[kcopy - 1]
    elif l == 65:
        theta = OddThetaZero32[kcopy - 1]
    elif l == 66:
        theta = EvenThetaZero33[kcopy - 1]
    elif l == 67:
        theta = OddThetaZero33[kcopy - 1]
    elif l == 68:
        theta = EvenThetaZero34[kcopy - 1]
    elif l == 69:
        theta = OddThetaZero34[kcopy - 1]
    elif l == 70:
        theta = EvenThetaZero35[kcopy - 1]
    elif l == 71:
        theta = OddThetaZero35[kcopy - 1]
    elif l == 72:
        theta = EvenThetaZero36[kcopy - 1]
    elif l == 73:
        theta = OddThetaZero36[kcopy - 1]
    elif l == 74:
        theta = EvenThetaZero37[kcopy - 1]
    elif l == 75:
        theta = OddThetaZero37[kcopy - 1]
    elif l == 76:
        theta = EvenThetaZero38[kcopy - 1]
    elif l == 77:
        theta = OddThetaZero38[kcopy - 1]
    elif l == 78:
        theta = EvenThetaZero39[kcopy - 1]
    elif l == 79:
        theta = OddThetaZero39[kcopy - 1]
    elif l == 80:
        theta = EvenThetaZero40[kcopy - 1]
    elif l == 81:
        theta = OddThetaZero40[kcopy - 1]
    elif l == 82:
        theta = EvenThetaZero41[kcopy - 1]
    elif l == 83:
        theta = OddThetaZero41[kcopy - 1]
    elif l == 84:
        theta = EvenThetaZero42[kcopy - 1]
    elif l == 85:
        theta = OddThetaZero42[kcopy - 1]
    elif l == 86:
        theta = EvenThetaZero43[kcopy - 1]
    elif l == 87:
        theta = OddThetaZero43[kcopy - 1]
    elif l == 88:
        theta = EvenThetaZero44[kcopy - 1]
    elif l == 89:
        theta = OddThetaZero44[kcopy - 1]
    elif l == 90:
        theta = EvenThetaZero45[kcopy - 1]
    elif l == 91:
        theta = OddThetaZero45[kcopy - 1]
    elif l == 92:
        theta = EvenThetaZero46[kcopy - 1]
    elif l == 93:
        theta = OddThetaZero46[kcopy - 1]
    elif l == 94:
        theta = EvenThetaZero47[kcopy - 1]
    elif l == 95:
        theta = OddThetaZero47[kcopy - 1]
    elif l == 96:
        theta = EvenThetaZero48[kcopy - 1]
    elif l == 97:
        theta = OddThetaZero48[kcopy - 1]
    elif l == 98:
        theta = EvenThetaZero49[kcopy - 1]
    elif l == 99:
        theta = OddThetaZero49[kcopy - 1]
    elif l == 100:
        theta = EvenThetaZero50[kcopy - 1]

    if (2 * k - 1) <= l:
        theta = np.pi - theta

    return theta


def legendre_theta_test():
    # *****************************************************************************80
    #
    ## LEGENDRE_THETA_TEST tests LEGENDRE_THETA.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    import platform

    print("")
    print("LEGENDRE_THETA_TEST:")
    print("  Python version: %s" % (platform.python_version()))
    print("  LEGENDRE_THETA returns the K-th theta value for")
    print("  a Gauss Legendre rule of order L.")

    for l in range(1, 11):
        print("")
        print("  Gauss Legendre rule of order %d" % (l))
        print("")
        print("   K       Theta      Cos(Theta)")
        print("")
        for k in range(1, l + 1):
            theta = legendre_theta(l, k)
            print("  %2d  %14.6g  %14.6g" % (k, theta, np.cos(theta)))
    #
    #  Terminate.
    #
    print("")
    print("LEGENDRE_THETA_TEST:")
    print("  Normal end of execution.")
    return


@numba.njit(cache=True)
def legendre_weight(l, k):
    # *****************************************************************************80
    #
    ## LEGENDRE_WEIGHT returns the K-th weight in an L-point Legendre rule.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    Original C++ version by Ignace Bogaert.
    #    Python version by John Burkardt.
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    #  Parameters:
    #
    #    Input, integer L, the number of points in the given rule.
    #    1 <= L.
    #
    #    Input, integer K, the index of the point to be returned.
    #    1 <= K <= L.
    #
    #    Output, real WEIGHT, the weight of the point.
    #

    cl = np.array(
        [
            1.0e00,
            1.0e00,
            -0.5000000000000000000000000e00,
            -0.1500000000000000000000000e01,
            0.3750000000000000000000000e00,
            0.1875000000000000000000000e01,
            -0.3125000000000000000000000e00,
            -0.2187500000000000000000000e01,
            0.2734375000000000000000000e00,
            0.2460937500000000000000000e01,
            -0.2460937500000000000000000e00,
            -0.2707031250000000000000000e01,
            0.2255859375000000000000000e00,
            0.2932617187500000000000000e01,
            -0.2094726562500000000000000e00,
            -0.3142089843750000000000000e01,
            0.1963806152343750000000000e00,
            0.3338470458984375000000000e01,
            -0.1854705810546875000000000e00,
            -0.3523941040039062500000000e01,
            0.1761970520019531250000000e00,
            0.3700138092041015625000000e01,
            -0.1681880950927734375000000e00,
            -0.3868326187133789062500000e01,
            0.1611802577972412109375000e00,
            0.4029506444931030273437500e01,
            -0.1549810171127319335937500e00,
            -0.4184487462043762207031250e01,
            0.1494459807872772216796875e00,
            0.4333933442831039428710938e01,
            -0.1444644480943679809570312e00,
            -0.4478397890925407409667969e01,
            0.1399499340914189815521240e00,
            0.4618347825016826391220093e01,
            -0.1358337595593184232711792e00,
            -0.4754181584576144814491272e01,
            0.1320605995715595781803131e00,
            0.4886242184147704392671585e01,
            -0.1285853206354659050703049e00,
            -0.5014827504783170297741890e01,
            0.1253706876195792574435472e00,
            0.5140198192402749555185437e01,
            -0.1223856712476845132187009e00,
            -0.5262583863650434068404138e01,
            0.1196041787193280470091850e00,
            0.5382188042369762115413323e01,
            -0.1170040878776035242481157e00,
            -0.5499192130247365639661439e01,
            0.1145665027134867841596133e00,
            0.5613758632960852423821052e01,
            -0.1122751726592170484764210e00,
            -0.5726033805620069472297473e01,
            0.1101160347234628744672591e00,
            0.5836149840343532346764732e01,
            -0.1080768488952505990141617e00,
            -0.5944226689238782945778894e01,
            0.1061469051649782668889088e00,
            0.6050373594403761212667803e01,
            -0.1043167861104096760804794e00,
            -0.6154690380514170888748282e01,
            0.1025781730085695148124714e00,
            0.6257268553522740403560753e01,
            -0.1009236863471409742509799e00,
            -0.6358192239869881377811733e01,
            0.9934675374796689652830833e-01,
            0.6457538993617848274340042e01,
            -0.9784149990330073142939457e-01,
            -0.6555380493521149005769436e01,
            0.9640265431648748537896230e-01,
            0.6651783147837636491148399e01,
            -0.9502547354053766415926284e-01,
            -0.6746808621378174155307661e01,
            0.9370567529691908549038419e-01,
            0.6840514296675093240798046e01,
            -0.9243938238750126001078440e-01,
            -0.6932953679062594500808830e01,
            0.9122307472450782237906355e-01,
            0.7024176753787102323187894e01,
            -0.9005354812547567081010120e-01,
            -0.7114230301912577993997995e01,
            0.8892787877390722492497493e-01,
            0.7203158180686485218922970e01,
            -0.8784339244739616120637768e-01,
            -0.7291001573133881380129347e01,
            0.8679763777540334976344461e-01,
            0.7377799210909284729892792e01,
            -0.8578836291754982244061386e-01,
            -0.7463587573826834552333406e01,
            0.8481349515712311991287961e-01,
            0.7548401068983957672246285e01,
            -0.8387112298871064080273650e-01,
            -0.7632272191972668313049022e01,
            0.8295948034752900340270676e-01,
            0.7715231672320197316451729e01,
            -0.8207693268425741826012477e-01,
            -0.7797308605004454734711853e01,
            0.8122196463546307015324847e-01,
            0.7878530569639917804865102e01,
            -0.8039316907795834494760308e-01,
            -0.7958923738717876149812705e01,
            0.7958923738717876149812705e-01,
        ]
    )
    EvenW1 = np.array([1.0e00])
    EvenW2 = np.array([0.6521451548625461426269364e00, 0.3478548451374538573730642e00])
    EvenW3 = np.array([0.4679139345726910473898704e00, 0.3607615730481386075698336e00, 0.1713244923791703450402969e00])
    EvenW4 = np.array(
        [
            0.3626837833783619829651504e00,
            0.3137066458778872873379622e00,
            0.2223810344533744705443556e00,
            0.1012285362903762591525320e00,
        ]
    )
    EvenW5 = np.array(
        [
            0.2955242247147528701738930e00,
            0.2692667193099963550912268e00,
            0.2190863625159820439955350e00,
            0.1494513491505805931457764e00,
            0.6667134430868813759356850e-01,
        ]
    )
    EvenW6 = np.array(
        [
            0.2491470458134027850005624e00,
            0.2334925365383548087608498e00,
            0.2031674267230659217490644e00,
            0.1600783285433462263346522e00,
            0.1069393259953184309602552e00,
            0.4717533638651182719461626e-01,
        ]
    )
    EvenW7 = np.array(
        [
            0.2152638534631577901958766e00,
            0.2051984637212956039659240e00,
            0.1855383974779378137417164e00,
            0.1572031671581935345696019e00,
            0.1215185706879031846894145e00,
            0.8015808715976020980563266e-01,
            0.3511946033175186303183410e-01,
        ]
    )
    EvenW8 = np.array(
        [
            0.1894506104550684962853967e00,
            0.1826034150449235888667636e00,
            0.1691565193950025381893119e00,
            0.1495959888165767320815019e00,
            0.1246289712555338720524763e00,
            0.9515851168249278480992520e-01,
            0.6225352393864789286284360e-01,
            0.2715245941175409485178166e-01,
        ]
    )
    EvenW9 = np.array(
        [
            0.1691423829631435918406565e00,
            0.1642764837458327229860538e00,
            0.1546846751262652449254180e00,
            0.1406429146706506512047311e00,
            0.1225552067114784601845192e00,
            0.1009420441062871655628144e00,
            0.7642573025488905652912984e-01,
            0.4971454889496979645333512e-01,
            0.2161601352648331031334248e-01,
        ]
    )
    EvenW10 = np.array(
        [
            0.1527533871307258506980843e00,
            0.1491729864726037467878288e00,
            0.1420961093183820513292985e00,
            0.1316886384491766268984948e00,
            0.1181945319615184173123774e00,
            0.1019301198172404350367504e00,
            0.8327674157670474872475850e-01,
            0.6267204833410906356950596e-01,
            0.4060142980038694133103928e-01,
            0.1761400713915211831186249e-01,
        ]
    )
    EvenW11 = np.array(
        [
            0.1392518728556319933754102e00,
            0.1365414983460151713525738e00,
            0.1311735047870623707329649e00,
            0.1232523768105124242855609e00,
            0.1129322960805392183934005e00,
            0.1004141444428809649320786e00,
            0.8594160621706772741444398e-01,
            0.6979646842452048809496104e-01,
            0.5229333515268328594031142e-01,
            0.3377490158481415479330258e-01,
            0.1462799529827220068498987e-01,
        ]
    )
    EvenW12 = np.array(
        [
            0.1279381953467521569740562e00,
            0.1258374563468282961213754e00,
            0.1216704729278033912044631e00,
            0.1155056680537256013533445e00,
            0.1074442701159656347825772e00,
            0.9761865210411388826988072e-01,
            0.8619016153195327591718514e-01,
            0.7334648141108030573403386e-01,
            0.5929858491543678074636724e-01,
            0.4427743881741980616860272e-01,
            0.2853138862893366318130802e-01,
            0.1234122979998719954680507e-01,
        ]
    )
    EvenW13 = np.array(
        [
            0.1183214152792622765163711e00,
            0.1166604434852965820446624e00,
            0.1133618165463196665494407e00,
            0.1084718405285765906565795e00,
            0.1020591610944254232384142e00,
            0.9421380035591414846366474e-01,
            0.8504589431348523921044770e-01,
            0.7468414976565974588707538e-01,
            0.6327404632957483553945402e-01,
            0.5097582529714781199831990e-01,
            0.3796238329436276395030342e-01,
            0.2441785109263190878961718e-01,
            0.1055137261734300715565387e-01,
        ]
    )
    EvenW14 = np.array(
        [
            0.1100470130164751962823763e00,
            0.1087111922582941352535716e00,
            0.1060557659228464179104165e00,
            0.1021129675780607698142166e00,
            0.9693065799792991585048880e-01,
            0.9057174439303284094218612e-01,
            0.8311341722890121839039666e-01,
            0.7464621423456877902393178e-01,
            0.6527292396699959579339794e-01,
            0.5510734567571674543148330e-01,
            0.4427293475900422783958756e-01,
            0.3290142778230437997763004e-01,
            0.2113211259277125975149896e-01,
            0.9124282593094517738816778e-02,
        ]
    )
    EvenW15 = np.array(
        [
            0.1028526528935588403412856e00,
            0.1017623897484055045964290e00,
            0.9959342058679526706278018e-01,
            0.9636873717464425963946864e-01,
            0.9212252223778612871763266e-01,
            0.8689978720108297980238752e-01,
            0.8075589522942021535469516e-01,
            0.7375597473770520626824384e-01,
            0.6597422988218049512812820e-01,
            0.5749315621761906648172152e-01,
            0.4840267283059405290293858e-01,
            0.3879919256962704959680230e-01,
            0.2878470788332336934971862e-01,
            0.1846646831109095914230276e-01,
            0.7968192496166605615469690e-02,
        ]
    )
    EvenW16 = np.array(
        [
            0.9654008851472780056676488e-01,
            0.9563872007927485941908208e-01,
            0.9384439908080456563918026e-01,
            0.9117387869576388471286854e-01,
            0.8765209300440381114277140e-01,
            0.8331192422694675522219922e-01,
            0.7819389578707030647174106e-01,
            0.7234579410884850622539954e-01,
            0.6582222277636184683765034e-01,
            0.5868409347853554714528360e-01,
            0.5099805926237617619616316e-01,
            0.4283589802222668065687810e-01,
            0.3427386291302143310268716e-01,
            0.2539206530926205945575196e-01,
            0.1627439473090567060516896e-01,
            0.7018610009470096600404748e-02,
        ]
    )
    EvenW17 = np.array(
        [
            0.9095674033025987361533764e-01,
            0.9020304437064072957394216e-01,
            0.8870189783569386928707642e-01,
            0.8646573974703574978424688e-01,
            0.8351309969984565518702044e-01,
            0.7986844433977184473881888e-01,
            0.7556197466003193127083398e-01,
            0.7062937581425572499903896e-01,
            0.6511152155407641137854442e-01,
            0.5905413582752449319396124e-01,
            0.5250741457267810616824590e-01,
            0.4552561152335327245382266e-01,
            0.3816659379638751632176606e-01,
            0.3049138063844613180944194e-01,
            0.2256372198549497008409476e-01,
            0.1445016274859503541520101e-01,
            0.6229140555908684718603220e-02,
        ]
    )
    EvenW18 = np.array(
        [
            0.8598327567039474749008516e-01,
            0.8534668573933862749185052e-01,
            0.8407821897966193493345756e-01,
            0.8218726670433970951722338e-01,
            0.7968782891207160190872470e-01,
            0.7659841064587067452875784e-01,
            0.7294188500565306135387342e-01,
            0.6874532383573644261368974e-01,
            0.6403979735501548955638454e-01,
            0.5886014424532481730967550e-01,
            0.5324471397775991909202590e-01,
            0.4723508349026597841661708e-01,
            0.4087575092364489547411412e-01,
            0.3421381077030722992124474e-01,
            0.2729862149856877909441690e-01,
            0.2018151529773547153209770e-01,
            0.1291594728406557440450307e-01,
            0.5565719664245045361251818e-02,
        ]
    )
    EvenW19 = np.array(
        [
            0.8152502928038578669921876e-01,
            0.8098249377059710062326952e-01,
            0.7990103324352782158602774e-01,
            0.7828784465821094807537540e-01,
            0.7615366354844639606599344e-01,
            0.7351269258474345714520658e-01,
            0.7038250706689895473928292e-01,
            0.6678393797914041193504612e-01,
            0.6274093339213305405296984e-01,
            0.5828039914699720602230556e-01,
            0.5343201991033231997375704e-01,
            0.4822806186075868337435238e-01,
            0.4270315850467443423587832e-01,
            0.3689408159402473816493978e-01,
            0.3083950054517505465873166e-01,
            0.2457973973823237589520214e-01,
            0.1815657770961323689887502e-01,
            0.1161344471646867417766868e-01,
            0.5002880749639345675901886e-02,
        ]
    )
    EvenW20 = np.array(
        [
            0.7750594797842481126372404e-01,
            0.7703981816424796558830758e-01,
            0.7611036190062624237155810e-01,
            0.7472316905796826420018930e-01,
            0.7288658239580405906051074e-01,
            0.7061164739128677969548346e-01,
            0.6791204581523390382569024e-01,
            0.6480401345660103807455446e-01,
            0.6130624249292893916653822e-01,
            0.5743976909939155136661768e-01,
            0.5322784698393682435499678e-01,
            0.4869580763507223206143380e-01,
            0.4387090818567327199167442e-01,
            0.3878216797447201763997196e-01,
            0.3346019528254784739267780e-01,
            0.2793700698002340109848970e-01,
            0.2224584919416695726150432e-01,
            0.1642105838190788871286396e-01,
            0.1049828453115281361474434e-01,
            0.4521277098533191258471490e-02,
        ]
    )
    EvenW21 = np.array(
        [
            0.7386423423217287999638556e-01,
            0.7346081345346752826402828e-01,
            0.7265617524380410488790570e-01,
            0.7145471426517098292181042e-01,
            0.6986299249259415976615480e-01,
            0.6788970337652194485536350e-01,
            0.6554562436490897892700504e-01,
            0.6284355804500257640931846e-01,
            0.5979826222758665431283142e-01,
            0.5642636935801838164642686e-01,
            0.5274629569917407034394234e-01,
            0.4877814079280324502744954e-01,
            0.4454357777196587787431674e-01,
            0.4006573518069226176059618e-01,
            0.3536907109759211083266214e-01,
            0.3047924069960346836290502e-01,
            0.2542295952611304788674188e-01,
            0.2022786956905264475705664e-01,
            0.1492244369735749414467869e-01,
            0.9536220301748502411822340e-02,
            0.4105998604649084610599928e-02,
        ]
    )
    EvenW22 = np.array(
        [
            0.7054915778935406881133824e-01,
            0.7019768547355821258714200e-01,
            0.6949649186157257803708402e-01,
            0.6844907026936666098545864e-01,
            0.6706063890629365239570506e-01,
            0.6533811487918143498424096e-01,
            0.6329007973320385495013890e-01,
            0.6092673670156196803855800e-01,
            0.5825985987759549533421064e-01,
            0.5530273556372805254874660e-01,
            0.5207009609170446188123162e-01,
            0.4857804644835203752763920e-01,
            0.4484398408197003144624282e-01,
            0.4088651231034621890844686e-01,
            0.3672534781380887364290888e-01,
            0.3238122281206982088084682e-01,
            0.2787578282128101008111450e-01,
            0.2323148190201921062895910e-01,
            0.1847148173681474917204335e-01,
            0.1361958675557998552020491e-01,
            0.8700481367524844122565470e-02,
            0.3745404803112777515171456e-02,
        ]
    )
    EvenW23 = np.array(
        [
            0.6751868584903645882021418e-01,
            0.6721061360067817586237416e-01,
            0.6659587476845488737576196e-01,
            0.6567727426778120737875756e-01,
            0.6445900346713906958827948e-01,
            0.6294662106439450817895206e-01,
            0.6114702772465048101535670e-01,
            0.5906843459554631480755080e-01,
            0.5672032584399123581687444e-01,
            0.5411341538585675449163752e-01,
            0.5125959800714302133536554e-01,
            0.4817189510171220053046892e-01,
            0.4486439527731812676709458e-01,
            0.4135219010967872970421980e-01,
            0.3765130535738607132766076e-01,
            0.3377862799910689652060416e-01,
            0.2975182955220275579905234e-01,
            0.2558928639713001063470016e-01,
            0.2130999875413650105447862e-01,
            0.1693351400783623804623151e-01,
            0.1247988377098868420673525e-01,
            0.7969898229724622451610710e-02,
            0.3430300868107048286016700e-02,
        ]
    )
    EvenW24 = np.array(
        [
            0.6473769681268392250302496e-01,
            0.6446616443595008220650418e-01,
            0.6392423858464818662390622e-01,
            0.6311419228625402565712596e-01,
            0.6203942315989266390419786e-01,
            0.6070443916589388005296916e-01,
            0.5911483969839563574647484e-01,
            0.5727729210040321570515042e-01,
            0.5519950369998416286820356e-01,
            0.5289018948519366709550490e-01,
            0.5035903555385447495780746e-01,
            0.4761665849249047482590674e-01,
            0.4467456085669428041944838e-01,
            0.4154508294346474921405856e-01,
            0.3824135106583070631721688e-01,
            0.3477722256477043889254814e-01,
            0.3116722783279808890206628e-01,
            0.2742650970835694820007336e-01,
            0.2357076083932437914051962e-01,
            0.1961616045735552781446139e-01,
            0.1557931572294384872817736e-01,
            0.1147723457923453948959265e-01,
            0.7327553901276262102386656e-02,
            0.3153346052305838632678320e-02,
        ]
    )
    EvenW25 = np.array(
        [
            0.6217661665534726232103316e-01,
            0.6193606742068324338408750e-01,
            0.6145589959031666375640678e-01,
            0.6073797084177021603175000e-01,
            0.5978505870426545750957640e-01,
            0.5860084981322244583512250e-01,
            0.5718992564772838372302946e-01,
            0.5555774480621251762356746e-01,
            0.5371062188899624652345868e-01,
            0.5165570306958113848990528e-01,
            0.4940093844946631492124360e-01,
            0.4695505130394843296563322e-01,
            0.4432750433880327549202254e-01,
            0.4152846309014769742241230e-01,
            0.3856875661258767524477018e-01,
            0.3545983561514615416073452e-01,
            0.3221372822357801664816538e-01,
            0.2884299358053519802990658e-01,
            0.2536067357001239044019428e-01,
            0.2178024317012479298159128e-01,
            0.1811556071348939035125903e-01,
            0.1438082276148557441937880e-01,
            0.1059054838365096926356876e-01,
            0.6759799195745401502778824e-02,
            0.2908622553155140958394976e-02,
        ]
    )
    EvenW26 = np.array(
        [
            0.5981036574529186024778538e-01,
            0.5959626017124815825831088e-01,
            0.5916881546604297036933200e-01,
            0.5852956177181386855029062e-01,
            0.5768078745252682765393200e-01,
            0.5662553090236859719080832e-01,
            0.5536756966930265254904124e-01,
            0.5391140693275726475083694e-01,
            0.5226225538390699303439404e-01,
            0.5042601856634237721821144e-01,
            0.4840926974407489685396032e-01,
            0.4621922837278479350764582e-01,
            0.4386373425900040799512978e-01,
            0.4135121950056027167904044e-01,
            0.3869067831042397898510146e-01,
            0.3589163483509723294194276e-01,
            0.3296410908971879791501014e-01,
            0.2991858114714394664128188e-01,
            0.2676595374650401344949324e-01,
            0.2351751355398446159032286e-01,
            0.2018489150798079220298930e-01,
            0.1678002339630073567792252e-01,
            0.1331511498234096065660116e-01,
            0.9802634579462752061952706e-02,
            0.6255523962973276899717754e-02,
            0.2691316950047111118946698e-02,
        ]
    )
    EvenW27 = np.array(
        [
            0.5761753670714702467237616e-01,
            0.5742613705411211485929010e-01,
            0.5704397355879459856782852e-01,
            0.5647231573062596503104434e-01,
            0.5571306256058998768336982e-01,
            0.5476873621305798630622270e-01,
            0.5364247364755361127210060e-01,
            0.5233801619829874466558872e-01,
            0.5085969714618814431970910e-01,
            0.4921242732452888606879048e-01,
            0.4740167880644499105857626e-01,
            0.4543346672827671397485208e-01,
            0.4331432930959701544192564e-01,
            0.4105130613664497422171834e-01,
            0.3865191478210251683685736e-01,
            0.3612412584038355258288694e-01,
            0.3347633646437264571604038e-01,
            0.3071734249787067605400450e-01,
            0.2785630931059587028700164e-01,
            0.2490274146720877305005456e-01,
            0.2186645142285308594551102e-01,
            0.1875752762146937791200757e-01,
            0.1558630303592413170296832e-01,
            0.1236332812884764416646861e-01,
            0.9099369455509396948032734e-02,
            0.5805611015239984878826112e-02,
            0.2497481835761585775945054e-02,
        ]
    )
    EvenW28 = np.array(
        [
            0.5557974630651439584627342e-01,
            0.5540795250324512321779340e-01,
            0.5506489590176242579630464e-01,
            0.5455163687088942106175058e-01,
            0.5386976186571448570895448e-01,
            0.5302137852401076396799152e-01,
            0.5200910915174139984305222e-01,
            0.5083608261779848056012412e-01,
            0.4950592468304757891996610e-01,
            0.4802274679360025812073550e-01,
            0.4639113337300189676219012e-01,
            0.4461612765269228321341510e-01,
            0.4270321608466708651103858e-01,
            0.4065831138474451788012514e-01,
            0.3848773425924766248682568e-01,
            0.3619819387231518603588452e-01,
            0.3379676711561176129542654e-01,
            0.3129087674731044786783572e-01,
            0.2868826847382274172988602e-01,
            0.2599698705839195219181960e-01,
            0.2322535156256531693725830e-01,
            0.2038192988240257263480560e-01,
            0.1747551291140094650495930e-01,
            0.1451508927802147180777130e-01,
            0.1150982434038338217377419e-01,
            0.8469063163307887661628584e-02,
            0.5402522246015337761313780e-02,
            0.2323855375773215501098716e-02,
        ]
    )
    EvenW29 = np.array(
        [
            0.5368111986333484886390600e-01,
            0.5352634330405825210061082e-01,
            0.5321723644657901410348096e-01,
            0.5275469052637083342964580e-01,
            0.5214003918366981897126058e-01,
            0.5137505461828572547451486e-01,
            0.5046194247995312529765992e-01,
            0.4940333550896239286651076e-01,
            0.4820228594541774840657052e-01,
            0.4686225672902634691841818e-01,
            0.4538711151481980250398048e-01,
            0.4378110353364025103902560e-01,
            0.4204886332958212599457020e-01,
            0.4019538540986779688807676e-01,
            0.3822601384585843322945902e-01,
            0.3614642686708727054078062e-01,
            0.3396262049341601079772722e-01,
            0.3168089125380932732029244e-01,
            0.2930781804416049071839382e-01,
            0.2685024318198186847590714e-01,
            0.2431525272496395254025850e-01,
            0.2171015614014623576691612e-01,
            0.1904246546189340865578709e-01,
            0.1631987423497096505212063e-01,
            0.1355023711298881214517933e-01,
            0.1074155353287877411685532e-01,
            0.7901973849998674754018608e-02,
            0.5039981612650243085015810e-02,
            0.2167723249627449943047768e-02,
        ]
    )
    EvenW30 = np.array(
        [
            0.5190787763122063973286496e-01,
            0.5176794317491018754380368e-01,
            0.5148845150098093399504444e-01,
            0.5107015606985562740454910e-01,
            0.5051418453250937459823872e-01,
            0.4982203569055018101115930e-01,
            0.4899557545575683538947578e-01,
            0.4803703181997118096366674e-01,
            0.4694898884891220484701330e-01,
            0.4573437971611448664719662e-01,
            0.4439647879578711332778398e-01,
            0.4293889283593564195423128e-01,
            0.4136555123558475561316394e-01,
            0.3968069545238079947012286e-01,
            0.3788886756924344403094056e-01,
            0.3599489805108450306657888e-01,
            0.3400389272494642283491466e-01,
            0.3192121901929632894945890e-01,
            0.2975249150078894524083642e-01,
            0.2750355674992479163522324e-01,
            0.2518047762152124837957096e-01,
            0.2278951694399781986378308e-01,
            0.2033712072945728677503268e-01,
            0.1782990101420772026039605e-01,
            0.1527461859678479930672510e-01,
            0.1267816647681596013149540e-01,
            0.1004755718228798435788578e-01,
            0.7389931163345455531517530e-02,
            0.4712729926953568640893942e-02,
            0.2026811968873758496433874e-02,
        ]
    )
    EvenW31 = np.array(
        [
            0.5024800037525628168840300e-01,
            0.5012106956904328807480410e-01,
            0.4986752859495239424476130e-01,
            0.4948801791969929252786578e-01,
            0.4898349622051783710485112e-01,
            0.4835523796347767283480314e-01,
            0.4760483018410123227045008e-01,
            0.4673416847841552480220700e-01,
            0.4574545221457018077723242e-01,
            0.4464117897712441429364478e-01,
            0.4342413825804741958006920e-01,
            0.4209740441038509664302268e-01,
            0.4066432888241744096828524e-01,
            0.3912853175196308412331100e-01,
            0.3749389258228002998561838e-01,
            0.3576454062276814128558760e-01,
            0.3394484437941054509111762e-01,
            0.3203940058162467810633926e-01,
            0.3005302257398987007700934e-01,
            0.2799072816331463754123820e-01,
            0.2585772695402469802709536e-01,
            0.2365940720868279257451652e-01,
            0.2140132227766996884117906e-01,
            0.1908917665857319873250324e-01,
            0.1672881179017731628855027e-01,
            0.1432619182380651776740340e-01,
            0.1188739011701050194481938e-01,
            0.9418579428420387637936636e-02,
            0.6926041901830960871704530e-02,
            0.4416333456930904813271960e-02,
            0.1899205679513690480402948e-02,
        ]
    )
    EvenW32 = np.array(
        [
            0.4869095700913972038336538e-01,
            0.4857546744150342693479908e-01,
            0.4834476223480295716976954e-01,
            0.4799938859645830772812614e-01,
            0.4754016571483030866228214e-01,
            0.4696818281621001732532634e-01,
            0.4628479658131441729595326e-01,
            0.4549162792741814447977098e-01,
            0.4459055816375656306013478e-01,
            0.4358372452932345337682780e-01,
            0.4247351512365358900733972e-01,
            0.4126256324262352861015628e-01,
            0.3995374113272034138665686e-01,
            0.3855015317861562912896262e-01,
            0.3705512854024004604041492e-01,
            0.3547221325688238381069330e-01,
            0.3380516183714160939156536e-01,
            0.3205792835485155358546770e-01,
            0.3023465707240247886797386e-01,
            0.2833967261425948322751098e-01,
            0.2637746971505465867169136e-01,
            0.2435270256871087333817770e-01,
            0.2227017380838325415929788e-01,
            0.2013482315353020937234076e-01,
            0.1795171577569734308504602e-01,
            0.1572603047602471932196614e-01,
            0.1346304789671864259806029e-01,
            0.1116813946013112881859029e-01,
            0.8846759826363947723030856e-02,
            0.6504457968978362856118112e-02,
            0.4147033260562467635287472e-02,
            0.1783280721696432947292054e-02,
        ]
    )
    EvenW33 = np.array(
        [
            0.4722748126299855484563332e-01,
            0.4712209828764473218544518e-01,
            0.4691156748762082774625404e-01,
            0.4659635863958410362582412e-01,
            0.4617717509791597547166640e-01,
            0.4565495222527305612043888e-01,
            0.4503085530544150021519278e-01,
            0.4430627694315316190460328e-01,
            0.4348283395666747864757528e-01,
            0.4256236377005571631890662e-01,
            0.4154692031324188131773448e-01,
            0.4043876943895497912586836e-01,
            0.3924038386682833018781280e-01,
            0.3795443766594162094913028e-01,
            0.3658380028813909441368980e-01,
            0.3513153016547255590064132e-01,
            0.3360086788611223267034862e-01,
            0.3199522896404688727128174e-01,
            0.3031819621886851919364104e-01,
            0.2857351178293187118282268e-01,
            0.2676506875425000190879332e-01,
            0.2489690251475737263773110e-01,
            0.2297318173532665591809836e-01,
            0.2099819909186462577733052e-01,
            0.1897636172277132593486659e-01,
            0.1691218147224521718035102e-01,
            0.1481026500273396017364296e-01,
            0.1267530398126168187644599e-01,
            0.1051206598770575465737803e-01,
            0.8325388765990901416725080e-02,
            0.6120192018447936365568516e-02,
            0.3901625641744248259228942e-02,
            0.1677653744007238599334225e-02,
        ]
    )
    EvenW34 = np.array(
        [
            0.4584938738725097468656398e-01,
            0.4575296541606795051900614e-01,
            0.4556032425064828598070770e-01,
            0.4527186901844377786941174e-01,
            0.4488820634542666782635216e-01,
            0.4441014308035275590934876e-01,
            0.4383868459795605201060492e-01,
            0.4317503268464422322584344e-01,
            0.4242058301114249930061428e-01,
            0.4157692219740291648457550e-01,
            0.4064582447595407614088174e-01,
            0.3962924796071230802540652e-01,
            0.3852933052910671449325372e-01,
            0.3734838532618666771607896e-01,
            0.3608889590017987071497568e-01,
            0.3475351097975151316679320e-01,
            0.3334503890398068790314300e-01,
            0.3186644171682106493934736e-01,
            0.3032082893855398034157906e-01,
            0.2871145102748499071080394e-01,
            0.2704169254590396155797848e-01,
            0.2531506504517639832390244e-01,
            0.2353519968587633336129308e-01,
            0.2170583961037807980146532e-01,
            0.1983083208795549829102926e-01,
            0.1791412045792315248940600e-01,
            0.1595973590961380007213420e-01,
            0.1397178917445765581596455e-01,
            0.1195446231976944210322336e-01,
            0.9912001251585937209131520e-02,
            0.7848711393177167415052160e-02,
            0.5768969918729952021468320e-02,
            0.3677366595011730633570254e-02,
            0.1581140256372912939103728e-02,
        ]
    )
    EvenW35 = np.array(
        [
            0.4454941715975466720216750e-01,
            0.4446096841724637082355728e-01,
            0.4428424653905540677579966e-01,
            0.4401960239018345875735580e-01,
            0.4366756139720144025254848e-01,
            0.4322882250506869978939520e-01,
            0.4270425678944977776996576e-01,
            0.4209490572728440602098398e-01,
            0.4140197912904520863822652e-01,
            0.4062685273678961635122600e-01,
            0.3977106549277656747784952e-01,
            0.3883631648407340397900292e-01,
            0.3782446156922281719727230e-01,
            0.3673750969367269534804046e-01,
            0.3557761890129238053276980e-01,
            0.3434709204990653756854510e-01,
            0.3304837223937242047087430e-01,
            0.3168403796130848173465310e-01,
            0.3025679798015423781653688e-01,
            0.2876948595580828066131070e-01,
            0.2722505481866441715910742e-01,
            0.2562657090846848279898494e-01,
            0.2397720788910029227868640e-01,
            0.2228024045225659583389064e-01,
            0.2053903782432645338449270e-01,
            0.1875705709313342341545081e-01,
            0.1693783637630293253183738e-01,
            0.1508498786544312768229492e-01,
            0.1320219081467674762507440e-01,
            0.1129318464993153764963015e-01,
            0.9361762769699026811498692e-02,
            0.7411769363190210362109460e-02,
            0.5447111874217218312821680e-02,
            0.3471894893078143254999524e-02,
            0.1492721288844515731042666e-02,
        ]
    )
    EvenW36 = np.array(
        [
            0.4332111216548653707639384e-01,
            0.4323978130522261748526514e-01,
            0.4307727227491369974525036e-01,
            0.4283389016833881366683982e-01,
            0.4251009191005772007780078e-01,
            0.4210648539758646414658732e-01,
            0.4162382836013859820760788e-01,
            0.4106302693607506110193610e-01,
            0.4042513397173397004332898e-01,
            0.3971134704483490178239872e-01,
            0.3892300621616966379996300e-01,
            0.3806159151380216383437540e-01,
            0.3712872015450289946055536e-01,
            0.3612614350763799298563092e-01,
            0.3505574380721787043413848e-01,
            0.3391953061828605949719618e-01,
            0.3271963706429384670431246e-01,
            0.3145831582256181397777608e-01,
            0.3013793489537547929298290e-01,
            0.2876097316470176109512506e-01,
            0.2733001573895093443379638e-01,
            0.2584774910065589028389804e-01,
            0.2431695606441916432634724e-01,
            0.2274051055503575445593134e-01,
            0.2112137221644055350981986e-01,
            0.1946258086329427804301667e-01,
            0.1776725078920065359435915e-01,
            0.1603856495028515521816122e-01,
            0.1427976905455419326655572e-01,
            0.1249416561987375776778277e-01,
            0.1068510816535189715895734e-01,
            0.8855996073706153383956510e-02,
            0.7010272321861863296081600e-02,
            0.5151436018790886908248502e-02,
            0.3283169774667495801897558e-02,
            0.1411516393973434135715864e-02,
        ]
    )
    EvenW37 = np.array(
        [
            0.4215870660994342212223066e-01,
            0.4208374996915697247489576e-01,
            0.4193396995777702146995522e-01,
            0.4170963287924075437870998e-01,
            0.4141113759675351082006810e-01,
            0.4103901482412726684741876e-01,
            0.4059392618219472805807676e-01,
            0.4007666302247696675915112e-01,
            0.3948814502019646832363280e-01,
            0.3882941853913770775808220e-01,
            0.3810165477126324889635168e-01,
            0.3730614765439415573370658e-01,
            0.3644431157165856448181076e-01,
            0.3551767883680095992585374e-01,
            0.3452789696982646100333388e-01,
            0.3347672576782876626372244e-01,
            0.3236603417621699952527994e-01,
            0.3119779696591542603337254e-01,
            0.2997409122246118733996502e-01,
            0.2869709265326987534209508e-01,
            0.2736907171967935230243778e-01,
            0.2599238960072378786677346e-01,
            0.2456949399594276724564910e-01,
            0.2310291477491582303093246e-01,
            0.2159525948167588896969968e-01,
            0.2004920870279494425273506e-01,
            0.1846751130897987978285368e-01,
            0.1685297958202485358484807e-01,
            0.1520848424340123480887426e-01,
            0.1353694941178749434105245e-01,
            0.1184134754749966732316814e-01,
            0.1012469453828730542112095e-01,
            0.8390045433971397064089364e-02,
            0.6640492909114357634760192e-02,
            0.4879179758594144584288316e-02,
            0.3109420149896754678673688e-02,
            0.1336761650069883550325931e-02,
        ]
    )
    EvenW38 = np.array(
        [
            0.4105703691622942259325972e-01,
            0.4098780546479395154130842e-01,
            0.4084945930182849228039176e-01,
            0.4064223171029473877745496e-01,
            0.4036647212284402315409558e-01,
            0.4002264553259682611646172e-01,
            0.3961133170906205842314674e-01,
            0.3913322422051844076750754e-01,
            0.3858912926450673834292118e-01,
            0.3797996430840528319523540e-01,
            0.3730675654238160982756716e-01,
            0.3657064114732961700724404e-01,
            0.3577285938071394752777924e-01,
            0.3491475648355076744412550e-01,
            0.3399777941205638084674262e-01,
            0.3302347439779174100654158e-01,
            0.3199348434042160006853510e-01,
            0.3090954603749159538993714e-01,
            0.2977348725590504095670750e-01,
            0.2858722365005400377397500e-01,
            0.2735275553182752167415270e-01,
            0.2607216449798598352427480e-01,
            0.2474760992065967164326474e-01,
            0.2338132530701118662247962e-01,
            0.2197561453441624916801320e-01,
            0.2053284796790802109297466e-01,
            0.1905545846719058280680223e-01,
            0.1754593729147423095419928e-01,
            0.1600682991224857088850986e-01,
            0.1444073174827667993988980e-01,
            0.1285028384751014494492467e-01,
            0.1123816856966768723967455e-01,
            0.9607105414713754082404616e-02,
            0.7959847477239734621118374e-02,
            0.6299180497328445866575096e-02,
            0.4627935228037421326126844e-02,
            0.2949102953642474900394994e-02,
            0.1267791634085359663272804e-02,
        ]
    )
    EvenW39 = np.array(
        [
            0.4001146511842048298877858e-01,
            0.3994739036908802487930490e-01,
            0.3981934348036408922503176e-01,
            0.3962752950781054295639346e-01,
            0.3937225562423312193722022e-01,
            0.3905393062777341314731136e-01,
            0.3867306428725767400389548e-01,
            0.3823026652585098764962036e-01,
            0.3772624644432424786429014e-01,
            0.3716181118549838685067108e-01,
            0.3653786464168470064819248e-01,
            0.3585540600719169544500572e-01,
            0.3511552817821718947488010e-01,
            0.3431941600268909029029166e-01,
            0.3346834438285897797298150e-01,
            0.3256367623368904440805548e-01,
            0.3160686030030479773888294e-01,
            0.3059942883801304528943330e-01,
            0.2954299515860694641162030e-01,
            0.2843925104689751626239046e-01,
            0.2728996405162436486456432e-01,
            0.2609697465510883502983394e-01,
            0.2486219332622245076144308e-01,
            0.2358759746145747209645146e-01,
            0.2227522821911388676305032e-01,
            0.2092718725187772678537816e-01,
            0.1954563334339992337791787e-01,
            0.1813277895498232864440684e-01,
            0.1669088668934389186621294e-01,
            0.1522226568017845169331591e-01,
            0.1372926792014414839372596e-01,
            0.1221428454978988639768250e-01,
            0.1067974215748111335351669e-01,
            0.9128099227255087276943326e-02,
            0.7561843189439718826977318e-02,
            0.5983489944440407989648850e-02,
            0.4395596039460346742737866e-02,
            0.2800868811838630411609396e-02,
            0.1204024566067353280336448e-02,
        ]
    )
    EvenW40 = np.array(
        [
            0.3901781365630665481128044e-01,
            0.3895839596276953119862554e-01,
            0.3883965105905196893177418e-01,
            0.3866175977407646332707712e-01,
            0.3842499300695942318521238e-01,
            0.3812971131447763834420674e-01,
            0.3777636436200139748977496e-01,
            0.3736549023873049002670538e-01,
            0.3689771463827600883915092e-01,
            0.3637374990583597804396502e-01,
            0.3579439395341605460286146e-01,
            0.3516052904474759349552658e-01,
            0.3447312045175392879436434e-01,
            0.3373321498461152281667534e-01,
            0.3294193939764540138283636e-01,
            0.3210049867348777314805654e-01,
            0.3121017418811470164244288e-01,
            0.3027232175955798066122008e-01,
            0.2928836958326784769276746e-01,
            0.2825981605727686239675312e-01,
            0.2718822750048638067441898e-01,
            0.2607523576756511790296854e-01,
            0.2492253576411549110511808e-01,
            0.2373188286593010129319242e-01,
            0.2250509024633246192622164e-01,
            0.2124402611578200638871032e-01,
            0.1995061087814199892889169e-01,
            0.1862681420829903142873492e-01,
            0.1727465205626930635858456e-01,
            0.1589618358372568804490352e-01,
            0.1449350804050907611696272e-01,
            0.1306876159240133929378674e-01,
            0.1162411412079782691646643e-01,
            0.1016176604110306452083288e-01,
            0.8683945269260858426408640e-02,
            0.7192904768117312752674654e-02,
            0.5690922451403198649270494e-02,
            0.4180313124694895236739096e-02,
            0.2663533589512681669292770e-02,
            0.1144950003186941534544369e-02,
        ]
    )
    EvenW41 = np.array(
        [
            0.3807230964014187120769602e-01,
            0.3801710843143526990530278e-01,
            0.3790678605050578477946422e-01,
            0.3774150245427586967153708e-01,
            0.3752149728818502087157412e-01,
            0.3724708953872766418784006e-01,
            0.3691867707095445699853162e-01,
            0.3653673605160765284219780e-01,
            0.3610182025872702307569544e-01,
            0.3561456027872747268049598e-01,
            0.3507566259211269038478042e-01,
            0.3448590854915070550737888e-01,
            0.3384615323699685874463648e-01,
            0.3315732423990721132775848e-01,
            0.3242042029434060507783656e-01,
            0.3163650984090024553762352e-01,
            0.3080672947521562981366802e-01,
            0.2993228230001272463508596e-01,
            0.2901443618076440396145302e-01,
            0.2805452190745423047171398e-01,
            0.2705393126512477151978662e-01,
            0.2601411501601702375386842e-01,
            0.2493658079624075515577230e-01,
            0.2382289093004782634222678e-01,
            0.2267466016491410310244200e-01,
            0.2149355333077484404348958e-01,
            0.2028128292691215890157032e-01,
            0.1903960664017892507303976e-01,
            0.1777032479849840714698234e-01,
            0.1647527776398370889101217e-01,
            0.1515634327076256178846848e-01,
            0.1381543371412645938772740e-01,
            0.1245449340114210467973318e-01,
            0.1107549578175989632022419e-01,
            0.9680440704371073736965104e-02,
            0.8271351818383685604431294e-02,
            0.6850274534183526184325356e-02,
            0.5419276232446765090703842e-02,
            0.3980457937856074619030326e-02,
            0.2536054696856106109823094e-02,
            0.1090118595275830866109234e-02,
        ]
    )
    EvenW42 = np.array(
        [
            0.3717153701903406760328362e-01,
            0.3712016261260209427372758e-01,
            0.3701748480379452058524442e-01,
            0.3686364550259030771845208e-01,
            0.3665885732875907563657692e-01,
            0.3640340331800212248862624e-01,
            0.3609763653077256670175260e-01,
            0.3574197956431530727788894e-01,
            0.3533692396860127616038866e-01,
            0.3488302956696330845641672e-01,
            0.3438092368237270062133504e-01,
            0.3383130027042598480372494e-01,
            0.3323491896024044407471552e-01,
            0.3259260400458425718361322e-01,
            0.3190524314069272748402282e-01,
            0.3117378636334566129196750e-01,
            0.3039924461190246977311372e-01,
            0.2958268837311084528960516e-01,
            0.2872524620162180221266452e-01,
            0.2782810316025840603576668e-01,
            0.2689249918219763751581640e-01,
            0.2591972735733464772516052e-01,
            0.2491113214520642888439108e-01,
            0.2386810751695823938471552e-01,
            0.2279209502894212933888898e-01,
            0.2168458183064482298924430e-01,
            0.2054709860975627861152400e-01,
            0.1938121747731880864780669e-01,
            0.1818854979605654992760044e-01,
            0.1697074395521161134308213e-01,
            0.1572948309558359820159970e-01,
            0.1446648278916118624227443e-01,
            0.1318348867918234598679997e-01,
            0.1188227408980122349505120e-01,
            0.1056463762300824526484878e-01,
            0.9232400784190247014382770e-02,
            0.7887405752648146382107148e-02,
            0.6531513687713654601121566e-02,
            0.5166605182746808329881136e-02,
            0.3794591650452349696393000e-02,
            0.2417511265443122855238466e-02,
            0.1039133516451971889197062e-02,
        ]
    )
    EvenW43 = np.array(
        [
            0.3631239537581333828231516e-01,
            0.3626450208420238743149194e-01,
            0.3616877866860063758274494e-01,
            0.3602535138093525771008956e-01,
            0.3583440939092405578977942e-01,
            0.3559620453657549559069116e-01,
            0.3531105099203420508058466e-01,
            0.3497932485321009937141316e-01,
            0.3460146364173769225993442e-01,
            0.3417796572791990463423808e-01,
            0.3370938967341755486497158e-01,
            0.3319635349455159712009034e-01,
            0.3263953384718992195609868e-01,
            0.3203966513429401611022852e-01,
            0.3139753853730286555853332e-01,
            0.3071400097263205318303994e-01,
            0.2998995397466493249133840e-01,
            0.2922635250670994458366154e-01,
            0.2842420370149349475731242e-01,
            0.2758456553285124838738412e-01,
            0.2670854542037220957530654e-01,
            0.2579729876883953540777106e-01,
            0.2485202744439983591832606e-01,
            0.2387397818947900497321768e-01,
            0.2286444097854800644577274e-01,
            0.2182474731692762780068420e-01,
            0.2075626848490914279058154e-01,
            0.1966041372956217980740210e-01,
            0.1853862840670985920631482e-01,
            0.1739239207569054238672012e-01,
            0.1622321654972902258808405e-01,
            0.1503264390508137868494523e-01,
            0.1382224445276667086664874e-01,
            0.1259361467806969781040954e-01,
            0.1134837515617770397716730e-01,
            0.1008816846038610565467284e-01,
            0.8814657101954815703782366e-02,
            0.7529521612194562606844596e-02,
            0.6234459139140123463885784e-02,
            0.4931184096960103696423408e-02,
            0.3621439249610901437553882e-02,
            0.2307087488809902925963262e-02,
            0.9916432666203635255681510e-03,
        ]
    )
    EvenW44 = np.array(
        [
            0.3549206430171454529606746e-01,
            0.3544734460447076970614316e-01,
            0.3535796155642384379366902e-01,
            0.3522402777945910853287866e-01,
            0.3504571202900426139658624e-01,
            0.3482323898139935499312912e-01,
            0.3455688895080708413486530e-01,
            0.3424699753602007873736958e-01,
            0.3389395519761025923989258e-01,
            0.3349820676595309252806520e-01,
            0.3306025088074670014528066e-01,
            0.3258063936273210868623942e-01,
            0.3205997651840638806926700e-01,
            0.3149891837860489232004182e-01,
            0.3089817187191219763370292e-01,
            0.3025849393394352533513752e-01,
            0.2958069055361934911335230e-01,
            0.2886561575763542924647688e-01,
            0.2811417053440861349157908e-01,
            0.2732730169885533083562360e-01,
            0.2650600069943473772140906e-01,
            0.2565130236896194788477952e-01,
            0.2476428362076873302532156e-01,
            0.2384606209185966126357838e-01,
            0.2289779473478114232724788e-01,
            0.2192067635998985359563460e-01,
            0.2091593813057662423225406e-01,
            0.1988484601127411324360109e-01,
            0.1882869917375545139470985e-01,
            0.1774882836032407455649534e-01,
            0.1664659420821765604511323e-01,
            0.1552338553693355384016474e-01,
            0.1438061760129994423593466e-01,
            0.1321973031362791170818164e-01,
            0.1204218643958121230973900e-01,
            0.1084946977542927125940107e-01,
            0.9643083322053204400769368e-02,
            0.8424547492702473015098308e-02,
            0.7195398459796372059759572e-02,
            0.5957186996138046583131162e-02,
            0.4711479279598661743021848e-02,
            0.3459867667862796423976646e-02,
            0.2204058563143696628535344e-02,
            0.9473355981619272667700360e-03,
        ]
    )
    EvenW45 = np.array(
        [
            0.3470797248895005792046014e-01,
            0.3466615208568824018827232e-01,
            0.3458256166949689141805380e-01,
            0.3445730196032425617459566e-01,
            0.3429052388637504193169728e-01,
            0.3408242840225399546360508e-01,
            0.3383326624683168725792750e-01,
            0.3354333764112427668293316e-01,
            0.3321299192655131651404080e-01,
            0.3284262714400750457863018e-01,
            0.3243268955425561691178950e-01,
            0.3198367310021857603945600e-01,
            0.3149611881181863607695780e-01,
            0.3097061415408092094593650e-01,
            0.3040779231928695269039426e-01,
            0.2980833146403127548714788e-01,
            0.2917295389210074248655798e-01,
            0.2850242518416141631875546e-01,
            0.2779755327530227515803874e-01,
            0.2705918748154795852161408e-01,
            0.2628821747651458736159580e-01,
            0.2548557221944322848446706e-01,
            0.2465221883590485293596628e-01,
            0.2378916145252872321010090e-01,
            0.2289743998716318463498862e-01,
            0.2197812889593413383869188e-01,
            0.2103233587872256311706242e-01,
            0.2006120054463959596453232e-01,
            0.1906589303913731842532399e-01,
            0.1804761263446023616404962e-01,
            0.1700758628522267570939747e-01,
            0.1594706715100663901320649e-01,
            0.1486733308804332405038481e-01,
            0.1376968511233709343075118e-01,
            0.1265544583716812886887583e-01,
            0.1152595788914805885059348e-01,
            0.1038258230989321461380844e-01,
            0.9226696957741990940319884e-02,
            0.8059694944620015658670990e-02,
            0.6882983208463284314729370e-02,
            0.5697981560747352600849438e-02,
            0.4506123613674977864136850e-02,
            0.3308867243336018195431340e-02,
            0.2107778774526329891473788e-02,
            0.9059323712148330937360098e-03,
        ]
    )
    EvenW46 = np.array(
        [
            0.3395777082810234796700260e-01,
            0.3391860442372254949502722e-01,
            0.3384031678893360189141840e-01,
            0.3372299821957387169380074e-01,
            0.3356678402920367631007550e-01,
            0.3337185439303681030780114e-01,
            0.3313843414012938182262046e-01,
            0.3286679249406566032646806e-01,
            0.3255724276244004524316198e-01,
            0.3221014197549332953574452e-01,
            0.3182589047432008582597260e-01,
            0.3140493144912217791614030e-01,
            0.3094775042804103166804096e-01,
            0.3045487471715832098063528e-01,
            0.2992687279231107330786762e-01,
            0.2936435364342281261274650e-01,
            0.2876796607210717582237958e-01,
            0.2813839794335440451445112e-01,
            0.2747637539216417339517938e-01,
            0.2678266198604032330048838e-01,
            0.2605805784431417922245786e-01,
            0.2530339871531322569754810e-01,
            0.2451955501244097425717108e-01,
            0.2370743081028191239353720e-01,
            0.2286796280189254240434106e-01,
            0.2200211921848585739874382e-01,
            0.2111089871276246180997612e-01,
            0.2019532920718748374956428e-01,
            0.1925646670855947471237209e-01,
            0.1829539409026755729118717e-01,
            0.1731321984368977636114053e-01,
            0.1631107680025595800481463e-01,
            0.1529012082579650150690625e-01,
            0.1425152948895392526580707e-01,
            0.1319650070571113802911160e-01,
            0.1212625136263771052929676e-01,
            0.1104201592263539422398575e-01,
            0.9945045019726082041770092e-02,
            0.8836604056467877374547944e-02,
            0.7717971837373568504533128e-02,
            0.6590439334214895223179124e-02,
            0.5455308908000870987158870e-02,
            0.4313895331861700472339122e-02,
            0.3167535943396097874261610e-02,
            0.2017671366262838591883234e-02,
            0.8671851787671421353540866e-03,
        ]
    )
    EvenW47 = np.array(
        [
            0.3323930891781532080070524e-01,
            0.3320257661860686379876634e-01,
            0.3312915261254696321600516e-01,
            0.3301911803949165507667076e-01,
            0.3287259449712959072614770e-01,
            0.3268974390660630715252838e-01,
            0.3247076833358767948450850e-01,
            0.3221590976496030711281812e-01,
            0.3192544984141561392584074e-01,
            0.3159970954621320046477392e-01,
            0.3123904885046741788219108e-01,
            0.3084386631534918741110674e-01,
            0.3041459865164271220328128e-01,
            0.2995172023714386920008800e-01,
            0.2945574259243367639719146e-01,
            0.2892721381560625584227516e-01,
            0.2836671797657610681272962e-01,
            0.2777487447163422062065088e-01,
            0.2715233733896656472388262e-01,
            0.2649979453589169919669406e-01,
            0.2581796717861672816440260e-01,
            0.2510760874535240512858038e-01,
            0.2436950424366898830634656e-01,
            0.2360446934301438228050796e-01,
            0.2281334947335523641001192e-01,
            0.2199701889094007717339700e-01,
            0.2115637971222138981504522e-01,
            0.2029236091701113217988866e-01,
            0.1940591732198200488605189e-01,
            0.1849802852566591095380957e-01,
            0.1756969782614325199872555e-01,
            0.1662195111266549663832874e-01,
            0.1565583573251555786002188e-01,
            0.1467241933449946420426407e-01,
            0.1367278869060687850644038e-01,
            0.1265804849763899444482439e-01,
            0.1162932016112241459607371e-01,
            0.1058774056495412223672440e-01,
            0.9534460832865158250063918e-02,
            0.8470645094534635999910406e-02,
            0.7397469288142356200862272e-02,
            0.6316120091036448223107804e-02,
            0.5227794289507767545307002e-02,
            0.4133699875407776483295790e-02,
            0.3035065891038628027389626e-02,
            0.1933219888725418943121000e-02,
            0.8308716126821624946495838e-03,
        ]
    )
    EvenW48 = np.array(
        [
            0.3255061449236316624196142e-01,
            0.3251611871386883598720548e-01,
            0.3244716371406426936401278e-01,
            0.3234382256857592842877486e-01,
            0.3220620479403025066866710e-01,
            0.3203445623199266321813896e-01,
            0.3182875889441100653475374e-01,
            0.3158933077072716855802074e-01,
            0.3131642559686135581278434e-01,
            0.3101033258631383742324982e-01,
            0.3067137612366914901422878e-01,
            0.3029991542082759379408878e-01,
            0.2989634413632838598438796e-01,
            0.2946108995816790597043632e-01,
            0.2899461415055523654267862e-01,
            0.2849741106508538564559948e-01,
            0.2797000761684833443981840e-01,
            0.2741296272602924282342110e-01,
            0.2682686672559176219805676e-01,
            0.2621234073567241391345816e-01,
            0.2557003600534936149879724e-01,
            0.2490063322248361028838244e-01,
            0.2420484179236469128226730e-01,
            0.2348339908592621984223612e-01,
            0.2273706965832937400134754e-01,
            0.2196664443874434919475618e-01,
            0.2117293989219129898767356e-01,
            0.2035679715433332459524556e-01,
            0.1951908114014502241008485e-01,
            0.1866067962741146738515655e-01,
            0.1778250231604526083761406e-01,
            0.1688547986424517245047785e-01,
            0.1597056290256229138061685e-01,
            0.1503872102699493800587588e-01,
            0.1409094177231486091586166e-01,
            0.1312822956696157263706415e-01,
            0.1215160467108831963518178e-01,
            0.1116210209983849859121361e-01,
            0.1016077053500841575758671e-01,
            0.9148671230783386632584044e-02,
            0.8126876925698759217383246e-02,
            0.7096470791153865269143206e-02,
            0.6058545504235961683315686e-02,
            0.5014202742927517692471308e-02,
            0.3964554338444686673733524e-02,
            0.2910731817934946408411678e-02,
            0.1853960788946921732331620e-02,
            0.7967920655520124294367096e-03,
        ]
    )
    EvenW49 = np.array(
        [
            0.3188987535287646727794502e-01,
            0.3185743815812401071309920e-01,
            0.3179259676252863019831786e-01,
            0.3169541712034925160907410e-01,
            0.3156599807910805290145092e-01,
            0.3140447127904656151748860e-01,
            0.3121100101922626441684056e-01,
            0.3098578409040993463104290e-01,
            0.3072904957489366992001356e-01,
            0.3044105861349325839490764e-01,
            0.3012210413992189884853100e-01,
            0.2977251058282947626617570e-01,
            0.2939263353580649216776328e-01,
            0.2898285939568834204744914e-01,
            0.2854360496952788570349054e-01,
            0.2807531705063613875324586e-01,
            0.2757847196412239390009986e-01,
            0.2705357508239612827767608e-01,
            0.2650116031112363935248738e-01,
            0.2592178954616244891846836e-01,
            0.2531605210202609734314644e-01,
            0.2468456411246099618197954e-01,
            0.2402796790374549880324124e-01,
            0.2334693134134927471268304e-01,
            0.2264214715061843311126274e-01,
            0.2191433221217865041901888e-01,
            0.2116422683277485691127980e-01,
            0.2039259399229191457948346e-01,
            0.1960021856772633077323700e-01,
            0.1878790653490468656148738e-01,
            0.1795648414877062812244296e-01,
            0.1710679710308990026235402e-01,
            0.1623970967045369565272614e-01,
            0.1535610382349775576849818e-01,
            0.1445687833830440197756895e-01,
            0.1354294788102946514364726e-01,
            0.1261524207892195285778215e-01,
            0.1167470457713812428742924e-01,
            0.1072229208322431712024324e-01,
            0.9758973402174096835348026e-02,
            0.8785728467392263202699392e-02,
            0.7803547379100754890979542e-02,
            0.6813429479165215998771186e-02,
            0.5816382546439639112764538e-02,
            0.4813422398586770918478190e-02,
            0.3805574085352359565512666e-02,
            0.2793881135722130870629084e-02,
            0.1779477041014528741695358e-02,
            0.7647669822743134580383448e-03,
        ]
    )
    EvenW50 = np.array(
        [
            0.3125542345386335694764248e-01,
            0.3122488425484935773237650e-01,
            0.3116383569620990678381832e-01,
            0.3107233742756651658781016e-01,
            0.3095047885049098823406346e-01,
            0.3079837903115259042771392e-01,
            0.3061618658398044849645950e-01,
            0.3040407952645482001650792e-01,
            0.3016226510516914491906862e-01,
            0.2989097959333283091683684e-01,
            0.2959048805991264251175454e-01,
            0.2926108411063827662011896e-01,
            0.2890308960112520313487610e-01,
            0.2851685432239509799093676e-01,
            0.2810275565910117331764820e-01,
            0.2766119822079238829420408e-01,
            0.2719261344657688013649158e-01,
            0.2669745918357096266038448e-01,
            0.2617621923954567634230892e-01,
            0.2562940291020811607564182e-01,
            0.2505754448157958970376402e-01,
            0.2446120270795705271997480e-01,
            0.2384096026596820596256040e-01,
            0.2319742318525412162248878e-01,
            0.2253122025633627270179672e-01,
            0.2184300241624738631395360e-01,
            0.2113344211252764154267220e-01,
            0.2040323264620943276683910e-01,
            0.1965308749443530586538157e-01,
            0.1888373961337490455294131e-01,
            0.1809594072212811666439111e-01,
            0.1729046056832358243934388e-01,
            0.1646808617614521264310506e-01,
            0.1562962107754600272393719e-01,
            0.1477588452744130176887969e-01,
            0.1390771070371877268795387e-01,
            0.1302594789297154228555807e-01,
            0.1213145766297949740774437e-01,
            0.1122511402318597711722209e-01,
            0.1030780257486896958578198e-01,
            0.9380419653694457951417628e-02,
            0.8443871469668971402620252e-02,
            0.7499073255464711578829804e-02,
            0.6546948450845322764152444e-02,
            0.5588428003865515157213478e-02,
            0.4624450063422119351093868e-02,
            0.3655961201326375182342828e-02,
            0.2683925371553482419437272e-02,
            0.1709392653518105239533969e-02,
            0.7346344905056717304142370e-03,
        ]
    )

    OddW1 = np.array([0.5555555555555555555555555e00])
    OddW2 = np.array([0.4786286704993664680412916e00, 0.2369268850561890875142644e00])
    OddW3 = np.array([0.3818300505051189449503698e00, 0.2797053914892766679014680e00, 0.1294849661688696932706118e00])
    OddW4 = np.array(
        [
            0.3123470770400028400686304e00,
            0.2606106964029354623187428e00,
            0.1806481606948574040584721e00,
            0.8127438836157441197189206e-01,
        ]
    )
    OddW5 = np.array(
        [
            0.2628045445102466621806890e00,
            0.2331937645919904799185238e00,
            0.1862902109277342514260979e00,
            0.1255803694649046246346947e00,
            0.5566856711617366648275374e-01,
        ]
    )
    OddW6 = np.array(
        [
            0.2262831802628972384120902e00,
            0.2078160475368885023125234e00,
            0.1781459807619457382800468e00,
            0.1388735102197872384636019e00,
            0.9212149983772844791442126e-01,
            0.4048400476531587952001996e-01,
        ]
    )
    OddW7 = np.array(
        [
            0.1984314853271115764561182e00,
            0.1861610000155622110268006e00,
            0.1662692058169939335532006e00,
            0.1395706779261543144478051e00,
            0.1071592204671719350118693e00,
            0.7036604748810812470926662e-01,
            0.3075324199611726835462762e-01,
        ]
    )
    OddW8 = np.array(
        [
            0.1765627053669926463252710e00,
            0.1680041021564500445099705e00,
            0.1540457610768102880814317e00,
            0.1351363684685254732863199e00,
            0.1118838471934039710947887e00,
            0.8503614831717918088353538e-01,
            0.5545952937398720112944102e-01,
            0.2414830286854793196010920e-01,
        ]
    )
    OddW9 = np.array(
        [
            0.1589688433939543476499565e00,
            0.1527660420658596667788553e00,
            0.1426067021736066117757460e00,
            0.1287539625393362276755159e00,
            0.1115666455473339947160242e00,
            0.9149002162244999946446222e-01,
            0.6904454273764122658070790e-01,
            0.4481422676569960033283728e-01,
            0.1946178822972647703631351e-01,
        ]
    )
    OddW10 = np.array(
        [
            0.1445244039899700590638271e00,
            0.1398873947910731547221335e00,
            0.1322689386333374617810526e00,
            0.1218314160537285341953671e00,
            0.1087972991671483776634747e00,
            0.9344442345603386155329010e-01,
            0.7610011362837930201705132e-01,
            0.5713442542685720828363528e-01,
            0.3695378977085249379995034e-01,
            0.1601722825777433332422273e-01,
        ]
    )
    OddW11 = np.array(
        [
            0.1324620394046966173716425e00,
            0.1289057221880821499785954e00,
            0.1230490843067295304675784e00,
            0.1149966402224113649416434e00,
            0.1048920914645414100740861e00,
            0.9291576606003514747701876e-01,
            0.7928141177671895492289248e-01,
            0.6423242140852585212716980e-01,
            0.4803767173108466857164124e-01,
            0.3098800585697944431069484e-01,
            0.1341185948714177208130864e-01,
        ]
    )
    OddW12 = np.array(
        [
            0.1222424429903100416889594e00,
            0.1194557635357847722281782e00,
            0.1148582591457116483393255e00,
            0.1085196244742636531160939e00,
            0.1005359490670506442022068e00,
            0.9102826198296364981149704e-01,
            0.8014070033500101801323524e-01,
            0.6803833381235691720718712e-01,
            0.5490469597583519192593686e-01,
            0.4093915670130631265562402e-01,
            0.2635498661503213726190216e-01,
            0.1139379850102628794789998e-01,
        ]
    )
    OddW13 = np.array(
        [
            0.1134763461089651486203700e00,
            0.1112524883568451926721632e00,
            0.1075782857885331872121629e00,
            0.1025016378177457986712478e00,
            0.9608872737002850756565252e-01,
            0.8842315854375695019432262e-01,
            0.7960486777305777126307488e-01,
            0.6974882376624559298432254e-01,
            0.5898353685983359911030058e-01,
            0.4744941252061506270409646e-01,
            0.3529705375741971102257772e-01,
            0.2268623159618062319603554e-01,
            0.9798996051294360261149438e-02,
        ]
    )
    OddW14 = np.array(
        [
            0.1058761550973209414065914e00,
            0.1040733100777293739133284e00,
            0.1010912737599149661218204e00,
            0.9696383409440860630190016e-01,
            0.9173775713925876334796636e-01,
            0.8547225736617252754534480e-01,
            0.7823832713576378382814484e-01,
            0.7011793325505127856958160e-01,
            0.6120309065707913854210970e-01,
            0.5159482690249792391259412e-01,
            0.4140206251868283610482948e-01,
            0.3074049220209362264440778e-01,
            0.1973208505612270598385931e-01,
            0.8516903878746409654261436e-02,
        ]
    )
    OddW15 = np.array(
        [
            0.9922501122667230787487546e-01,
            0.9774333538632872509347402e-01,
            0.9529024291231951280720412e-01,
            0.9189011389364147821536290e-01,
            0.8757674060847787612619794e-01,
            0.8239299176158926390382334e-01,
            0.7639038659877661642635764e-01,
            0.6962858323541036616775632e-01,
            0.6217478656102842691034334e-01,
            0.5410308242491685371166596e-01,
            0.4549370752720110290231576e-01,
            0.3643227391238546402439264e-01,
            0.2700901918497942180060860e-01,
            0.1731862079031058246315918e-01,
            0.7470831579248775858700554e-02,
        ]
    )
    OddW16 = np.array(
        [
            0.9335642606559611616099912e-01,
            0.9212398664331684621324104e-01,
            0.9008195866063857723974370e-01,
            0.8724828761884433760728158e-01,
            0.8364787606703870761392808e-01,
            0.7931236479488673836390848e-01,
            0.7427985484395414934247216e-01,
            0.6859457281865671280595482e-01,
            0.6230648253031748003162750e-01,
            0.5547084663166356128494468e-01,
            0.4814774281871169567014706e-01,
            0.4040154133166959156340938e-01,
            0.3230035863232895328156104e-01,
            0.2391554810174948035053310e-01,
            0.1532170151293467612794584e-01,
            0.6606227847587378058647800e-02,
        ]
    )
    OddW17 = np.array(
        [
            0.8814053043027546297073886e-01,
            0.8710444699718353424332214e-01,
            0.8538665339209912522594402e-01,
            0.8300059372885658837992644e-01,
            0.7996494224232426293266204e-01,
            0.7630345715544205353865872e-01,
            0.7204479477256006466546180e-01,
            0.6722228526908690396430546e-01,
            0.6187367196608018888701398e-01,
            0.5604081621237012857832772e-01,
            0.4976937040135352980519956e-01,
            0.4310842232617021878230592e-01,
            0.3611011586346338053271748e-01,
            0.2882926010889425404871630e-01,
            0.2132297991148358088343844e-01,
            0.1365082834836149226640441e-01,
            0.5883433420443084975750336e-02,
        ]
    )
    OddW18 = np.array(
        [
            0.8347457362586278725225302e-01,
            0.8259527223643725089123018e-01,
            0.8113662450846503050987774e-01,
            0.7910886183752938076721222e-01,
            0.7652620757052923788588804e-01,
            0.7340677724848817272462668e-01,
            0.6977245155570034488508154e-01,
            0.6564872287275124948402376e-01,
            0.6106451652322598613098804e-01,
            0.5605198799827491780853916e-01,
            0.5064629765482460160387558e-01,
            0.4488536466243716665741054e-01,
            0.3880960250193454448896226e-01,
            0.3246163984752148106723444e-01,
            0.2588603699055893352275954e-01,
            0.1912904448908396604350259e-01,
            0.1223878010030755652630649e-01,
            0.5273057279497939351724544e-02,
        ]
    )
    OddW19 = np.array(
        [
            0.7927622256836847101015574e-01,
            0.7852361328737117672506330e-01,
            0.7727455254468201672851160e-01,
            0.7553693732283605770478448e-01,
            0.7332175341426861738115402e-01,
            0.7064300597060876077011486e-01,
            0.6751763096623126536302120e-01,
            0.6396538813868238898670650e-01,
            0.6000873608859614957494160e-01,
            0.5567269034091629990739094e-01,
            0.5098466529212940521402098e-01,
            0.4597430110891663188417682e-01,
            0.4067327684793384393905618e-01,
            0.3511511149813133076106530e-01,
            0.2933495598390337859215654e-01,
            0.2336938483217816459471240e-01,
            0.1725622909372491904080491e-01,
            0.1103478893916459424267603e-01,
            0.4752944691635101370775866e-02,
        ]
    )
    OddW20 = np.array(
        [
            0.7547874709271582402724706e-01,
            0.7482962317622155189130518e-01,
            0.7375188202722346993928094e-01,
            0.7225169686102307339634646e-01,
            0.7033766062081749748165896e-01,
            0.6802073676087676673553342e-01,
            0.6531419645352741043616384e-01,
            0.6223354258096631647157330e-01,
            0.5879642094987194499118590e-01,
            0.5502251924257874188014710e-01,
            0.5093345429461749478117008e-01,
            0.4655264836901434206075674e-01,
            0.4190519519590968942934048e-01,
            0.3701771670350798843526154e-01,
            0.3191821173169928178706676e-01,
            0.2663589920711044546754900e-01,
            0.2120106336877955307569710e-01,
            0.1564493840781858853082666e-01,
            0.9999938773905945338496546e-02,
            0.4306140358164887684003630e-02,
        ]
    )
    OddW21 = np.array(
        [
            0.7202750197142197434530754e-01,
            0.7146373425251414129758106e-01,
            0.7052738776508502812628636e-01,
            0.6922334419365668428229950e-01,
            0.6755840222936516919240796e-01,
            0.6554124212632279749123378e-01,
            0.6318238044939611232562970e-01,
            0.6049411524999129451967862e-01,
            0.5749046195691051942760910e-01,
            0.5418708031888178686337342e-01,
            0.5060119278439015652385048e-01,
            0.4675149475434658001064704e-01,
            0.4265805719798208376380686e-01,
            0.3834222219413265757212856e-01,
            0.3382649208686029234496834e-01,
            0.2913441326149849491594084e-01,
            0.2429045661383881590201850e-01,
            0.1931990142368390039612543e-01,
            0.1424875643157648610854214e-01,
            0.9103996637401403318866628e-02,
            0.3919490253844127282968528e-02,
        ]
    )
    OddW22 = np.array(
        [
            0.6887731697766132288200278e-01,
            0.6838457737866967453169206e-01,
            0.6756595416360753627091012e-01,
            0.6642534844984252808291474e-01,
            0.6496819575072343085382664e-01,
            0.6320144007381993774996374e-01,
            0.6113350083106652250188634e-01,
            0.5877423271884173857436156e-01,
            0.5613487875978647664392382e-01,
            0.5322801673126895194590376e-01,
            0.5006749923795202979913194e-01,
            0.4666838771837336526776814e-01,
            0.4304688070916497115169120e-01,
            0.3922023672930244756418756e-01,
            0.3520669220160901624770010e-01,
            0.3102537493451546716250854e-01,
            0.2669621396757766480567536e-01,
            0.2223984755057873239395080e-01,
            0.1767753525793759061709347e-01,
            0.1303110499158278432063191e-01,
            0.8323189296218241645734836e-02,
            0.3582663155283558931145652e-02,
        ]
    )
    OddW23 = np.array(
        [
            0.6599053358881047453357062e-01,
            0.6555737776654974025114294e-01,
            0.6483755623894572670260402e-01,
            0.6383421660571703063129384e-01,
            0.6255174622092166264056434e-01,
            0.6099575300873964533071060e-01,
            0.5917304094233887597615438e-01,
            0.5709158029323154022201646e-01,
            0.5476047278153022595712512e-01,
            0.5218991178005714487221170e-01,
            0.4939113774736116960457022e-01,
            0.4637638908650591120440168e-01,
            0.4315884864847953826830162e-01,
            0.3975258612253100378090162e-01,
            0.3617249658417495161345948e-01,
            0.3243423551518475676761786e-01,
            0.2855415070064338650473990e-01,
            0.2454921165965881853783378e-01,
            0.2043693814766842764203432e-01,
            0.1623533314643305967072624e-01,
            0.1196284846431232096394232e-01,
            0.7638616295848833614105174e-02,
            0.3287453842528014883248206e-02,
        ]
    )
    OddW24 = np.array(
        [
            0.6333550929649174859083696e-01,
            0.6295270746519569947439960e-01,
            0.6231641732005726740107682e-01,
            0.6142920097919293629682652e-01,
            0.6029463095315201730310616e-01,
            0.5891727576002726602452756e-01,
            0.5730268153018747548516450e-01,
            0.5545734967480358869043158e-01,
            0.5338871070825896852794302e-01,
            0.5110509433014459067462262e-01,
            0.4861569588782824027765094e-01,
            0.4593053935559585354249958e-01,
            0.4306043698125959798834538e-01,
            0.4001694576637302136860494e-01,
            0.3681232096300068981946734e-01,
            0.3345946679162217434248744e-01,
            0.2997188462058382535069014e-01,
            0.2636361892706601696094518e-01,
            0.2264920158744667649877160e-01,
            0.1884359585308945844445106e-01,
            0.1496214493562465102958377e-01,
            0.1102055103159358049750846e-01,
            0.7035099590086451473452956e-02,
            0.3027278988922905077484090e-02,
        ]
    )
    OddW25 = np.array(
        [
            0.6088546484485634388119860e-01,
            0.6054550693473779513812526e-01,
            0.5998031577750325209006396e-01,
            0.5919199392296154378353896e-01,
            0.5818347398259214059843780e-01,
            0.5695850772025866210007778e-01,
            0.5552165209573869301673704e-01,
            0.5387825231304556143409938e-01,
            0.5203442193669708756413650e-01,
            0.4999702015005740977954886e-01,
            0.4777362624062310199999514e-01,
            0.4537251140765006874816670e-01,
            0.4280260799788008665360980e-01,
            0.4007347628549645318680892e-01,
            0.3719526892326029284290846e-01,
            0.3417869320418833623620910e-01,
            0.3103497129016000845442504e-01,
            0.2777579859416247719599602e-01,
            0.2441330057378143427314164e-01,
            0.2095998840170321057979252e-01,
            0.1742871472340105225950284e-01,
            0.1383263400647782229668883e-01,
            0.1018519129782172993923731e-01,
            0.6500337783252600292109494e-02,
            0.2796807171089895575547228e-02,
        ]
    )
    OddW26 = np.array(
        [
            0.5861758623272026331807196e-01,
            0.5831431136225600755627570e-01,
            0.5781001499171319631968304e-01,
            0.5710643553626719177338328e-01,
            0.5620599838173970980865512e-01,
            0.5511180752393359900234954e-01,
            0.5382763486873102904208140e-01,
            0.5235790722987271819970160e-01,
            0.5070769106929271529648556e-01,
            0.4888267503269914042044844e-01,
            0.4688915034075031402187278e-01,
            0.4473398910367281021276570e-01,
            0.4242462063452001359228150e-01,
            0.3996900584354038212709364e-01,
            0.3737560980348291567417214e-01,
            0.3465337258353423795838740e-01,
            0.3181167845901932306323576e-01,
            0.2886032361782373626279970e-01,
            0.2580948251075751771396152e-01,
            0.2266967305707020839878928e-01,
            0.1945172110763689538804750e-01,
            0.1616672525668746392806095e-01,
            0.1282602614424037917915135e-01,
            0.9441202284940344386662890e-02,
            0.6024276226948673281242120e-02,
            0.2591683720567031811603734e-02,
        ]
    )
    OddW27 = np.array(
        [
            0.5651231824977200140065834e-01,
            0.5624063407108436802827906e-01,
            0.5578879419528408710293598e-01,
            0.5515824600250868759665114e-01,
            0.5435100932991110207032224e-01,
            0.5336967000160547272357054e-01,
            0.5221737154563208456439348e-01,
            0.5089780512449397922477522e-01,
            0.4941519771155173948075862e-01,
            0.4777429855120069555003682e-01,
            0.4598036394628383810390480e-01,
            0.4403914042160658989516800e-01,
            0.4195684631771876239520718e-01,
            0.3974015187433717960946388e-01,
            0.3739615786796554528291572e-01,
            0.3493237287358988740726862e-01,
            0.3235668922618583168470572e-01,
            0.2967735776516104122129630e-01,
            0.2690296145639627066711996e-01,
            0.2404238800972562200779126e-01,
            0.2110480166801645412020978e-01,
            0.1809961452072906240796732e-01,
            0.1503645833351178821315019e-01,
            0.1192516071984861217075236e-01,
            0.8775746107058528177390204e-02,
            0.5598632266560767354082364e-02,
            0.2408323619979788819164582e-02,
        ]
    )
    OddW28 = np.array(
        [
            0.5455280360476188648013898e-01,
            0.5430847145249864313874678e-01,
            0.5390206148329857464280950e-01,
            0.5333478658481915842657698e-01,
            0.5260833972917743244023134e-01,
            0.5172488892051782472062386e-01,
            0.5068707072492740865664050e-01,
            0.4949798240201967899383808e-01,
            0.4816117266168775126885110e-01,
            0.4668063107364150378384082e-01,
            0.4506077616138115779721374e-01,
            0.4330644221621519659643210e-01,
            0.4142286487080111036319668e-01,
            0.3941566547548011408995280e-01,
            0.3729083432441731735473546e-01,
            0.3505471278231261750575064e-01,
            0.3271397436637156854248994e-01,
            0.3027560484269399945849064e-01,
            0.2774688140218019232125814e-01,
            0.2513535099091812264727322e-01,
            0.2244880789077643807968978e-01,
            0.1969527069948852038242318e-01,
            0.1688295902344154903500062e-01,
            0.1402027079075355617024753e-01,
            0.1111576373233599014567619e-01,
            0.8178160067821232626211086e-02,
            0.5216533474718779390504886e-02,
            0.2243753872250662909727492e-02,
        ]
    )
    OddW29 = np.array(
        [
            0.5272443385912793196130422e-01,
            0.5250390264782873905094128e-01,
            0.5213703364837539138398724e-01,
            0.5162484939089148214644000e-01,
            0.5096877742539391685024800e-01,
            0.5017064634299690281072034e-01,
            0.4923268067936198577969374e-01,
            0.4815749471460644038814684e-01,
            0.4694808518696201919315986e-01,
            0.4560782294050976983186828e-01,
            0.4414044353029738069079808e-01,
            0.4255003681106763866730838e-01,
            0.4084103553868670766020196e-01,
            0.3901820301616000950303072e-01,
            0.3708661981887092269183778e-01,
            0.3505166963640010878371850e-01,
            0.3291902427104527775751116e-01,
            0.3069462783611168323975056e-01,
            0.2838468020053479790515332e-01,
            0.2599561973129850018665014e-01,
            0.2353410539371336342527500e-01,
            0.2100699828843718735046168e-01,
            0.1842134275361002936061624e-01,
            0.1578434731308146614732024e-01,
            0.1310336630634519101831859e-01,
            0.1038588550099586219379846e-01,
            0.7639529453487575142699186e-02,
            0.4872239168265284768580414e-02,
            0.2095492284541223402697724e-02,
        ]
    )
    OddW30 = np.array(
        [
            0.5101448703869726354373512e-01,
            0.5081476366881834320770052e-01,
            0.5048247038679740464814450e-01,
            0.5001847410817825342505160e-01,
            0.4942398534673558993996884e-01,
            0.4870055505641152608753004e-01,
            0.4785007058509560716183348e-01,
            0.4687475075080906597642932e-01,
            0.4577714005314595937133982e-01,
            0.4456010203508348827154136e-01,
            0.4322681181249609790104358e-01,
            0.4178074779088849206667564e-01,
            0.4022568259099824736764020e-01,
            0.3856567320700817274615216e-01,
            0.3680505042315481738432126e-01,
            0.3494840751653335109085198e-01,
            0.3300058827590741063272390e-01,
            0.3096667436839739482469792e-01,
            0.2885197208818340150434184e-01,
            0.2666199852415088966281066e-01,
            0.2440246718754420291534050e-01,
            0.2207927314831904400247522e-01,
            0.1969847774610118133051782e-01,
            0.1726629298761374359443389e-01,
            0.1478906588493791454617878e-01,
            0.1227326350781210462927897e-01,
            0.9725461830356133736135366e-02,
            0.7152354991749089585834616e-02,
            0.4560924006012417184541648e-02,
            0.1961453361670282671779431e-02,
        ]
    )
    OddW31 = np.array(
        [
            0.4941183303991817896703964e-01,
            0.4923038042374756078504314e-01,
            0.4892845282051198994470936e-01,
            0.4850678909788384786409014e-01,
            0.4796642113799513141105276e-01,
            0.4730867131226891908060508e-01,
            0.4653514924538369651039536e-01,
            0.4564774787629260868588592e-01,
            0.4464863882594139537033256e-01,
            0.4354026708302759079896428e-01,
            0.4232534502081582298250554e-01,
            0.4100684575966639863511004e-01,
            0.3958799589154409398480778e-01,
            0.3807226758434955676363856e-01,
            0.3646337008545728963045232e-01,
            0.3476524064535587769718026e-01,
            0.3298203488377934176568344e-01,
            0.3111811662221981750821608e-01,
            0.2917804720828052694555162e-01,
            0.2716657435909793322519012e-01,
            0.2508862055334498661862972e-01,
            0.2294927100488993314894282e-01,
            0.2075376125803909077534152e-01,
            0.1850746446016127040926083e-01,
            0.1621587841033833888228333e-01,
            0.1388461261611561082486681e-01,
            0.1151937607688004175075116e-01,
            0.9125968676326656354058462e-02,
            0.6710291765960136251908410e-02,
            0.4278508346863761866081200e-02,
            0.1839874595577084117085868e-02,
        ]
    )
    OddW32 = np.array(
        [
            0.4790669250049586203134730e-01,
            0.4774134868124062155903898e-01,
            0.4746619823288550315264446e-01,
            0.4708187401045452224600686e-01,
            0.4658925997223349830225508e-01,
            0.4598948914665169696389334e-01,
            0.4528394102630023065712822e-01,
            0.4447423839508297442732352e-01,
            0.4356224359580048653228480e-01,
            0.4255005424675580271921714e-01,
            0.4143999841724029302268646e-01,
            0.4023462927300553381544642e-01,
            0.3893671920405119761667398e-01,
            0.3754925344825770980977246e-01,
            0.3607542322556527393216642e-01,
            0.3451861839854905862522142e-01,
            0.3288241967636857498404946e-01,
            0.3117059038018914246443218e-01,
            0.2938706778931066806264472e-01,
            0.2753595408845034394249940e-01,
            0.2562150693803775821408458e-01,
            0.2364812969128723669878144e-01,
            0.2162036128493406284165378e-01,
            0.1954286583675006282683714e-01,
            0.1742042199767024849536596e-01,
            0.1525791214644831034926464e-01,
            0.1306031163999484633616732e-01,
            0.1083267878959796862151440e-01,
            0.8580148266881459893636434e-02,
            0.6307942578971754550189764e-02,
            0.4021524172003736347075858e-02,
            0.1729258251300250898337759e-02,
        ]
    )
    OddW33 = np.array(
        [
            0.4649043816026462820831466e-01,
            0.4633935168241562110844706e-01,
            0.4608790448976157619721740e-01,
            0.4573664116106369093689412e-01,
            0.4528632245466953156805004e-01,
            0.4473792366088982547214182e-01,
            0.4409263248975101830783160e-01,
            0.4335184649869951735915584e-01,
            0.4251717006583049147154770e-01,
            0.4159041091519924309854838e-01,
            0.4057357620174452522725164e-01,
            0.3946886816430888264288692e-01,
            0.3827867935617948064763712e-01,
            0.3700558746349258202313488e-01,
            0.3565234972274500666133270e-01,
            0.3422189694953664673983902e-01,
            0.3271732719153120542712204e-01,
            0.3114189901947282393742616e-01,
            0.2949902447094566969584718e-01,
            0.2779226166243676998720012e-01,
            0.2602530708621323880370460e-01,
            0.2420198760967316472069180e-01,
            0.2232625219645207692279754e-01,
            0.2040216337134354044925720e-01,
            0.1843388845680457387216616e-01,
            0.1642569062253087920472674e-01,
            0.1438191982720055093097663e-01,
            0.1230700384928815052195302e-01,
            0.1020544003410244098666155e-01,
            0.8081790299023136215346300e-02,
            0.5940693177582235216514606e-02,
            0.3787008301825508445960626e-02,
            0.1628325035240012866460003e-02,
        ]
    )
    OddW34 = np.array(
        [
            0.4515543023614546051651704e-01,
            0.4501700814039980219871620e-01,
            0.4478661887831255754213528e-01,
            0.4446473312204713809623108e-01,
            0.4405200846590928438098588e-01,
            0.4354928808292674103357578e-01,
            0.4295759900230521387841984e-01,
            0.4227815001128051285158270e-01,
            0.4151232918565450208287406e-01,
            0.4066170105406160053752604e-01,
            0.3972800340176164120645862e-01,
            0.3871314372049251393273936e-01,
            0.3761919531164090650815840e-01,
            0.3644839305070051405664348e-01,
            0.3520312882168348614775456e-01,
            0.3388594663083228949780964e-01,
            0.3249953740964611124473418e-01,
            0.3104673351789053903268552e-01,
            0.2953050295790671177981110e-01,
            0.2795394331218770599086132e-01,
            0.2632027541686948379176090e-01,
            0.2463283678454245536433616e-01,
            0.2289507479074078565552120e-01,
            0.2111053963987189462789068e-01,
            0.1928287712884940278924393e-01,
            0.1741582123196982913207401e-01,
            0.1551318654340616473976910e-01,
            0.1357886064907567099981112e-01,
            0.1161679661067196554873961e-01,
            0.9631006150415575588660562e-02,
            0.7625555931201510611459992e-02,
            0.5604579927870594828535346e-02,
            0.3572416739397372609702552e-02,
            0.1535976952792084075135094e-02,
        ]
    )
    OddW35 = np.array(
        [
            0.4389487921178858632125256e-01,
            0.4376774491340214497230982e-01,
            0.4355612710410853337113396e-01,
            0.4326043426324126659885626e-01,
            0.4288123715758043502060704e-01,
            0.4241926773962459303533940e-01,
            0.4187541773473300618954268e-01,
            0.4125073691986602424910896e-01,
            0.4054643109724689643492514e-01,
            0.3976385976685758167433708e-01,
            0.3890453350226294749240264e-01,
            0.3797011103483115621441804e-01,
            0.3696239605198203185608278e-01,
            0.3588333371564891077796844e-01,
            0.3473500690768218837536532e-01,
            0.3351963220945403083440624e-01,
            0.3223955562344352694190700e-01,
            0.3089724804509072169860608e-01,
            0.2949530049370881246493644e-01,
            0.2803641911174149061798030e-01,
            0.2652341994215790800810512e-01,
            0.2495922349431387305527612e-01,
            0.2334684910922325263171504e-01,
            0.2168940913598536796183230e-01,
            0.1999010293235011128748561e-01,
            0.1825221070467867050232934e-01,
            0.1647908720746239655059230e-01,
            0.1467415533461152920040808e-01,
            0.1284089966808780607041846e-01,
            0.1098286015429855170627475e-01,
            0.9103626461992005851317578e-02,
            0.7206835281831493387342912e-02,
            0.5296182844025892632677844e-02,
            0.3375555496730675865126842e-02,
            0.1451267330029397268489446e-02,
        ]
    )
    OddW36 = np.array(
        [
            0.4270273086485722207660098e-01,
            0.4258568982601838702576300e-01,
            0.4239085899223159440537396e-01,
            0.4211859425425563626894556e-01,
            0.4176939294869285375410172e-01,
            0.4134389294952549452688336e-01,
            0.4084287150293886154936056e-01,
            0.4026724380756003336494178e-01,
            0.3961806134270614331650800e-01,
            0.3889650994769673952047552e-01,
            0.3810390765573980059550798e-01,
            0.3724170228634977315689404e-01,
            0.3631146880069778469034650e-01,
            0.3531490642472828750906318e-01,
            0.3425383554530221541412972e-01,
            0.3313019438504384067706900e-01,
            0.3194603546197670648650132e-01,
            0.3070352184043350493812614e-01,
            0.2940492318011656010545704e-01,
            0.2805261159057206032380240e-01,
            0.2664905729872748295223048e-01,
            0.2519682413753831281333190e-01,
            0.2369856486421897462660896e-01,
            0.2215701631704007205676952e-01,
            0.2057499442036116916601972e-01,
            0.1895538904867002168973610e-01,
            0.1730115876248908300560664e-01,
            0.1561532543359142299553300e-01,
            0.1390096878831465086752053e-01,
            0.1216122092928111272776412e-01,
            0.1039926099500053220130511e-01,
            0.8618310479532247613912182e-02,
            0.6821631349174792362208078e-02,
            0.5012538571606190263812266e-02,
            0.3194524377289034522078870e-02,
            0.1373376462759619223985654e-02,
        ]
    )
    OddW37 = np.array(
        [
            0.4157356944178127878299940e-01,
            0.4146558103261909213524834e-01,
            0.4128580808246718908346088e-01,
            0.4103456181139210667622250e-01,
            0.4071227717293733029875788e-01,
            0.4031951210114157755817430e-01,
            0.3985694654465635257596536e-01,
            0.3932538128963516252076754e-01,
            0.3872573657343257584146640e-01,
            0.3805905049151360313563098e-01,
            0.3732647720033209016730652e-01,
            0.3652928491929033900685118e-01,
            0.3566885373524045308911856e-01,
            0.3474667321333040653509838e-01,
            0.3376433981833409264695562e-01,
            0.3272355415093422052152286e-01,
            0.3162611800374964805603220e-01,
            0.3047393124221453920313760e-01,
            0.2926898851572598680503318e-01,
            0.2801337580478054082525924e-01,
            0.2670926681012085177235442e-01,
            0.2535891919021637909420806e-01,
            0.2396467065371695917476570e-01,
            0.2252893491386577645054636e-01,
            0.2105419751228284223644546e-01,
            0.1954301152012788937957076e-01,
            0.1799799312564505063794604e-01,
            0.1642181711902464004359937e-01,
            0.1481721228981446852013731e-01,
            0.1318695676282480211961300e-01,
            0.1153387332830449596681366e-01,
            0.9860824916114018392051822e-02,
            0.8170710707327826403717118e-02,
            0.6466464907037538401963982e-02,
            0.4751069185015273965898868e-02,
            0.3027671014606041291230134e-02,
            0.1301591717375855993899257e-02,
        ]
    )
    OddW38 = np.array(
        [
            0.4050253572678803195524960e-01,
            0.4040269003221775617032620e-01,
            0.4023646282485108419526524e-01,
            0.4000412721559123741035150e-01,
            0.3970606493128931068103760e-01,
            0.3934276568757015193713232e-01,
            0.3891482638423378562103292e-01,
            0.3842295012455452367368120e-01,
            0.3786794506008932026166678e-01,
            0.3725072306289371887876038e-01,
            0.3657229822732745453345840e-01,
            0.3583378520391196260264276e-01,
            0.3503639736797827845487748e-01,
            0.3418144482611567926531782e-01,
            0.3327033226369854530283962e-01,
            0.3230455663703097559357210e-01,
            0.3128570471390543339395640e-01,
            0.3021545046662299869139892e-01,
            0.2909555232176876134870268e-01,
            0.2792785027127696854150716e-01,
            0.2671426284955789083200264e-01,
            0.2545678398169440375263742e-01,
            0.2415747970795584494059388e-01,
            0.2281848479012952051290956e-01,
            0.2144199920545613550512462e-01,
            0.2003028453431617639624646e-01,
            0.1858566024834148550917969e-01,
            0.1711049990653110417623953e-01,
            0.1560722726874913129508073e-01,
            0.1407831234002700405016720e-01,
            0.1252626736922736518735940e-01,
            0.1095364285391135423859170e-01,
            0.9363023692386430769260798e-02,
            0.7757025950083070731841176e-02,
            0.6138296159756341839268696e-02,
            0.4509523600205835333238688e-02,
            0.2873553083652691657275240e-02,
            0.1235291177139409614163874e-02,
        ]
    )
    OddW39 = np.array(
        [
            0.3948525740129116475372166e-01,
            0.3939275600474300393426418e-01,
            0.3923874749659464355491890e-01,
            0.3902347234287979602650502e-01,
            0.3874726667023996706818530e-01,
            0.3841056174110417740541666e-01,
            0.3801388328032604954551756e-01,
            0.3755785065432977047790708e-01,
            0.3704317590404678415983790e-01,
            0.3647066263315342752925638e-01,
            0.3584120475334575228920704e-01,
            0.3515578508861113112825058e-01,
            0.3441547384067660088259166e-01,
            0.3362142691803093004992252e-01,
            0.3277488413113081785342150e-01,
            0.3187716725661117036051890e-01,
            0.3092967797352483528829388e-01,
            0.2993389567483836289564858e-01,
            0.2889137515760726678163634e-01,
            0.2780374419544705894443552e-01,
            0.2667270099710555653788310e-01,
            0.2550001155512877394733978e-01,
            0.2428750688879949263942200e-01,
            0.2303708018571902627697914e-01,
            0.2175068384660807976864198e-01,
            0.2043032643814085987844290e-01,
            0.1907806955893748858478357e-01,
            0.1769602462431041786466318e-01,
            0.1628634957619168209183741e-01,
            0.1485124552635006931857919e-01,
            0.1339295334482567619730830e-01,
            0.1191375021511699869960077e-01,
            0.1041594620451338257918368e-01,
            0.8901880982652486253740074e-02,
            0.7373921131330176830391914e-02,
            0.5834459868763465589211910e-02,
            0.4285929113126531218219446e-02,
            0.2730907065754855918535274e-02,
            0.1173930129956613021207112e-02,
        ]
    )
    OddW40 = np.array(
        [
            0.3851778959688469523783810e-01,
            0.3843192958037517210025656e-01,
            0.3828897129558352443032002e-01,
            0.3808912713547560183102332e-01,
            0.3783269400830055924757518e-01,
            0.3752005289647583785923924e-01,
            0.3715166829056371214474266e-01,
            0.3672808749918043951690600e-01,
            0.3624993983586341279832570e-01,
            0.3571793568410456853072614e-01,
            0.3513286544193937941597898e-01,
            0.3449559834765979589474544e-01,
            0.3380708118839624555119598e-01,
            0.3306833689348800442087536e-01,
            0.3228046301473268887240310e-01,
            0.3144463009577406641803652e-01,
            0.3056207993305266189565968e-01,
            0.2963412373090559765847516e-01,
            0.2866214015356067622579182e-01,
            0.2764757327692492691108618e-01,
            0.2659193044321992109092004e-01,
            0.2549678002166567706947970e-01,
            0.2436374907856309733249090e-01,
            0.2319452096027391988145570e-01,
            0.2199083279275163277050144e-01,
            0.2075447290144560853952252e-01,
            0.1948727815560191821592671e-01,
            0.1819113124125576115176324e-01,
            0.1686795786763513947433495e-01,
            0.1551972391246436293824549e-01,
            0.1414843251323606554825229e-01,
            0.1275612111513442100025550e-01,
            0.1134485849541625576200880e-01,
            0.9916741809595875499750926e-02,
            0.8473893785345565449616918e-02,
            0.7018460484931625511609624e-02,
            0.5552611370256278902273182e-02,
            0.4078551113421395586018386e-02,
            0.2598622299928953013499446e-02,
            0.1117029847124606606122469e-02,
        ]
    )
    OddW41 = np.array(
        [
            0.3759656394395517759196934e-01,
            0.3751672450373727271505762e-01,
            0.3738378433575740441091762e-01,
            0.3719793160197673054400130e-01,
            0.3695942935618497107975802e-01,
            0.3666861517167809004390068e-01,
            0.3632590066346228889989584e-01,
            0.3593177090566064734733082e-01,
            0.3548678374494710264584324e-01,
            0.3499156901097965473152462e-01,
            0.3444682762495051683252180e-01,
            0.3385333060751519869931002e-01,
            0.3321191798750501518117324e-01,
            0.3252349761296806599129116e-01,
            0.3178904386622215064354856e-01,
            0.3100959628473919484306724e-01,
            0.3018625808981441705410184e-01,
            0.2932019462510452791804122e-01,
            0.2841263170724764156375054e-01,
            0.2746485389090326123892810e-01,
            0.2647820265067376248510830e-01,
            0.2545407448248949675081806e-01,
            0.2439391892715855749743432e-01,
            0.2329923651890054937016126e-01,
            0.2217157666180362262199056e-01,
            0.2101253543726991787400918e-01,
            0.1982375334565493904931242e-01,
            0.1860691298547847284166721e-01,
            0.1736373667382462235016547e-01,
            0.1609598401193537091543832e-01,
            0.1480544940071787768084914e-01,
            0.1349395951237523498069998e-01,
            0.1216337072779861206303406e-01,
            0.1081556655803715872036043e-01,
            0.9452455092479699888244178e-02,
            0.8075966593123452283593892e-02,
            0.6688051635243685741358420e-02,
            0.5290681445859865555240374e-02,
            0.3885859435353202192003776e-02,
            0.2475719322545939743331242e-02,
            0.1064168219666567756385077e-02,
        ]
    )
    OddW42 = np.array(
        [
            0.3671834473341961622215226e-01,
            0.3664397593378570248640692e-01,
            0.3652013948874488485747660e-01,
            0.3634700257169520376675674e-01,
            0.3612479890936246037475190e-01,
            0.3585382846628081255691520e-01,
            0.3553445703985569908199156e-01,
            0.3516711576655578824981280e-01,
            0.3475230053990063752924744e-01,
            0.3429057134102984670822224e-01,
            0.3378255148275753033131186e-01,
            0.3322892676813276976252854e-01,
            0.3263044456464217818903764e-01,
            0.3198791279530467445976990e-01,
            0.3130219884802087044839684e-01,
            0.3057422840464999572392432e-01,
            0.2980498419139588737561256e-01,
            0.2899550465219015208986610e-01,
            0.2814688254686507584638292e-01,
            0.2726026347601116478577010e-01,
            0.2633684433451435982173160e-01,
            0.2537787169586608847736972e-01,
            0.2438464012943568314241580e-01,
            0.2335849045298989189769872e-01,
            0.2230080792283937418945736e-01,
            0.2121302036408937967241628e-01,
            0.2009659624357542174179408e-01,
            0.1895304268818284044680496e-01,
            0.1778390345139817090774314e-01,
            0.1659075683115467007520452e-01,
            0.1537521354238962687440865e-01,
            0.1413891454840083293055609e-01,
            0.1288352885649808429050626e-01,
            0.1161075128670389800962475e-01,
            0.1032230023052424589381722e-01,
            0.9019915439993631278967098e-02,
            0.7705355960382757079897960e-02,
            0.6380398587897515098686098e-02,
            0.5046838426924442725450432e-02,
            0.3706500125759316706868292e-02,
            0.2361331704285020896763904e-02,
            0.1014971908967743695374167e-02,
        ]
    )
    OddW43 = np.array(
        [
            0.3588019106018701587773518e-01,
            0.3581080434383374175662560e-01,
            0.3569525919440943377647946e-01,
            0.3553370454416059391133478e-01,
            0.3532634862941021369843054e-01,
            0.3507345872215153655662536e-01,
            0.3477536078554782924871120e-01,
            0.3443243905378224376593820e-01,
            0.3404513553679937345518354e-01,
            0.3361394945057693558422230e-01,
            0.3313943657366202353628890e-01,
            0.3262220853080144392580048e-01,
            0.3206293200458966777765818e-01,
            0.3146232787615076393796228e-01,
            0.3082117029596223415371898e-01,
            0.3014028568601882474395096e-01,
            0.2942055167462304824922484e-01,
            0.2866289596517621838858744e-01,
            0.2786829514042920598963448e-01,
            0.2703777340373580728397710e-01,
            0.2617240125893355894972542e-01,
            0.2527329413055707316411874e-01,
            0.2434161092616763233921348e-01,
            0.2337855254266017225782364e-01,
            0.2238536031848547821419758e-01,
            0.2136331443380253159361604e-01,
            0.2031373226065556952656956e-01,
            0.1923796666535655878505047e-01,
            0.1813740426535425205021816e-01,
            0.1701346364300153443364516e-01,
            0.1586759351882631900292224e-01,
            0.1470127088723984222989451e-01,
            0.1351599911824565808188095e-01,
            0.1231330603004803654228712e-01,
            0.1109474194056071927972064e-01,
            0.9861877713701826716584494e-02,
            0.8616302838488951832949878e-02,
            0.7359623648818063660769462e-02,
            0.6093462047634872130101964e-02,
            0.4819456238501885899307624e-02,
            0.3539271655388628540179688e-02,
            0.2254690753752853092482060e-02,
            0.9691097381770753376096654e-03,
        ]
    )
    OddW44 = np.array(
        [
            0.3507942401790202531716760e-01,
            0.3501458416619644336915306e-01,
            0.3490660650856070989101148e-01,
            0.3475562407298142092081152e-01,
            0.3456182286913780813643384e-01,
            0.3432544165923908781796544e-01,
            0.3404677166387108716735582e-01,
            0.3372615620321457070630952e-01,
            0.3336399027407732093971928e-01,
            0.3296072006326111707429234e-01,
            0.3251684239786320696758578e-01,
            0.3203290413318958550703170e-01,
            0.3150950147903428365879858e-01,
            0.3094727926515484478947892e-01,
            0.3034693014684912934340756e-01,
            0.2970919375161245962730194e-01,
            0.2903485576792681183001942e-01,
            0.2832474697730520722803496e-01,
            0.2757974223078458253347716e-01,
            0.2680075937112917771256550e-01,
            0.2598875810207383625148160e-01,
            0.2514473880600256862281534e-01,
            0.2426974131152233927366188e-01,
            0.2336484361245544582716880e-01,
            0.2243116053983636712835892e-01,
            0.2146984238856114084341254e-01,
            0.2048207350040027021224486e-01,
            0.1946907080515187313867415e-01,
            0.1843208232178411567584622e-01,
            0.1737238562150240166964102e-01,
            0.1629128625479238457754130e-01,
            0.1519011614466612339747308e-01,
            0.1407023194864448281388687e-01,
            0.1293301339260267729158710e-01,
            0.1177986158087489217661933e-01,
            0.1061219728997218803268093e-01,
            0.9431459260797890539711922e-02,
            0.8239102525389078730572362e-02,
            0.7036596870989114137389446e-02,
            0.5825425788770107459644064e-02,
            0.4607087343463241433054622e-02,
            0.3383104792407455132632698e-02,
            0.2155112582219113764637582e-02,
            0.9262871051934728155239026e-03,
        ]
    )
    OddW45 = np.array(
        [
            0.3431359817623139857242020e-01,
            0.3425291647165106006719224e-01,
            0.3415185977541012618567448e-01,
            0.3401054720622907866548866e-01,
            0.3382914533369793579365620e-01,
            0.3360786798193575310982430e-01,
            0.3334697597754983863697838e-01,
            0.3304677684219179120016898e-01,
            0.3270762443007278294842040e-01,
            0.3232991851086539448409380e-01,
            0.3191410429848369728859888e-01,
            0.3146067192629708854519032e-01,
            0.3097015586939654421561894e-01,
            0.3044313431459439490344712e-01,
            0.2988022847890037493277136e-01,
            0.2928210187727747971826382e-01,
            0.2864945954054102439649608e-01,
            0.2798304718432316638118606e-01,
            0.2728365033008298027898986e-01,
            0.2655209337919890810307922e-01,
            0.2578923864123601618879028e-01,
            0.2499598531753495743256148e-01,
            0.2417326844132287942221788e-01,
            0.2332205777559880283599600e-01,
            0.2244335667009737337332098e-01,
            0.2153820087868566629622426e-01,
            0.2060765733859846074045938e-01,
            0.1965282291296914660474199e-01,
            0.1867482309816812542178599e-01,
            0.1767481069752190506037194e-01,
            0.1665396446306124017225753e-01,
            0.1561348770705005975095101e-01,
            0.1455460688520869608484063e-01,
            0.1347857015383097919431856e-01,
            0.1238664590355674305453526e-01,
            0.1128012127376968298340906e-01,
            0.1016030065441547672889225e-01,
            0.9028504189234487748913298e-02,
            0.7886066314628901599629988e-02,
            0.6734334432268884665261132e-02,
            0.5574668047479788997832340e-02,
            0.4408439747302676819065170e-02,
            0.3237045507972104977098260e-02,
            0.2061987122032229660677942e-02,
            0.8862412406694141765769646e-03,
        ]
    )
    OddW46 = np.array(
        [
            0.3358047670273290820423322e-01,
            0.3352360509236689973246714e-01,
            0.3342889041048296629425518e-01,
            0.3329643957561578934524218e-01,
            0.3312640210470322597293962e-01,
            0.3291896994430459113247722e-01,
            0.3267437725392241575486392e-01,
            0.3239290014167229270630344e-01,
            0.3207485635259921958171598e-01,
            0.3172060490999230883258760e-01,
            0.3133054571010280192591498e-01,
            0.3090511907072293590876800e-01,
            0.3044480523413530949647580e-01,
            0.2995012382499392416587776e-01,
            0.2942163326374897748551588e-01,
            0.2885993013627770636290672e-01,
            0.2826564852043306435742870e-01,
            0.2763945927027071971311622e-01,
            0.2698206925876273304878794e-01,
            0.2629422057985327475229788e-01,
            0.2557668971075783892217594e-01,
            0.2483028663545258189183534e-01,
            0.2405585393034465615306556e-01,
            0.2325426581315775168991978e-01,
            0.2242642715610957188910656e-01,
            0.2157327246449981801505782e-01,
            0.2069576482186873448858912e-01,
            0.1979489480292792866805571e-01,
            0.1887167935550803461442971e-01,
            0.1792716065281371317885285e-01,
            0.1696240491732901090122756e-01,
            0.1597850121778211678831695e-01,
            0.1497656024067188095391932e-01,
            0.1395771303800797072406999e-01,
            0.1292310975318535045602668e-01,
            0.1187391832744712509861298e-01,
            0.1081132319054248938202577e-01,
            0.9736523941887687826947068e-02,
            0.8650734035428648314139846e-02,
            0.7555179500769820751618632e-02,
            0.6451097794311275889059324e-02,
            0.5339737098169214613757504e-02,
            0.4222357382406607998634106e-02,
            0.3100240403099316775464478e-02,
            0.1974768768686808388940061e-02,
            0.8487371680679110048896640e-03,
        ]
    )
    OddW47 = np.array(
        [
            0.3287800959763194823557646e-01,
            0.3282463569369918669308888e-01,
            0.3273574336068393226919658e-01,
            0.3261142878598215425670652e-01,
            0.3245182648620325926685946e-01,
            0.3225710916161441434734840e-01,
            0.3202748750926769529295728e-01,
            0.3176320999501228029097900e-01,
            0.3146456258463840201321734e-01,
            0.3113186843444399825682258e-01,
            0.3076548754155891475295788e-01,
            0.3036581635440506677724356e-01,
            0.2993328734371411225240016e-01,
            0.2946836853456688237515152e-01,
            0.2897156299996101153484194e-01,
            0.2844340831645486261311894e-01,
            0.2788447598247691424309350e-01,
            0.2729537079993022266578380e-01,
            0.2667673021976135431896846e-01,
            0.2602922365220227153290076e-01,
            0.2535355174243201293660006e-01,
            0.2465044561244261997612948e-01,
            0.2392066606993061007707546e-01,
            0.2316500278507139174920030e-01,
            0.2238427343606939184041926e-01,
            0.2157932282441140120676856e-01,
            0.2075102196078490181790884e-01,
            0.1990026712265721124487174e-01,
            0.1902797888454570639306994e-01,
            0.1813510112204514410759734e-01,
            0.1722259999071698441334003e-01,
            0.1629146288099104326591566e-01,
            0.1534269735028835663459242e-01,
            0.1437733003365908208357459e-01,
            0.1339640553436828544136536e-01,
            0.1240098529611606104018197e-01,
            0.1139214645908584403924275e-01,
            0.1037098070311609684083942e-01,
            0.9338593083876397086740596e-02,
            0.8296100874530990238145090e-02,
            0.7244632443933199672626606e-02,
            0.6185326261033323769312750e-02,
            0.5119330329927718280032034e-02,
            0.4047803316371759906879922e-02,
            0.2971924240818190718436604e-02,
            0.1892968377922935762776147e-02,
            0.8135642494541165010544716e-03,
        ]
    )
    OddW48 = np.array(
        [
            0.3220431459661350533475748e-01,
            0.3215415737958550153577998e-01,
            0.3207061987527279934927952e-01,
            0.3195378880670864194528382e-01,
            0.3180378546007149044495368e-01,
            0.3162076555877401604294910e-01,
            0.3140491910180172362457798e-01,
            0.3115647016646904145775102e-01,
            0.3087567667579765382432642e-01,
            0.3056283013075858386135104e-01,
            0.3021825530765601453452082e-01,
            0.2984230992096702903457814e-01,
            0.2943538425198732086424294e-01,
            0.2899790074366843187205222e-01,
            0.2853031356206718751823808e-01,
            0.2803310812486267752680532e-01,
            0.2750680059743034256009616e-01,
            0.2695193735699644067363378e-01,
            0.2636909442542934975707846e-01,
            0.2575887687125678489535242e-01,
            0.2512191818153004673565192e-01,
            0.2445887960418784729059960e-01,
            0.2377044946160306882104198e-01,
            0.2305734243602599579639616e-01,
            0.2232029882766713237862322e-01,
            0.2156008378619171827843500e-01,
            0.2077748651642656849799008e-01,
            0.1997331945910804688818908e-01,
            0.1914841744752812933525703e-01,
            0.1830363684096414082229124e-01,
            0.1743985463580780463940516e-01,
            0.1655796755534245662902801e-01,
            0.1565889111915692052020687e-01,
            0.1474355869323695017635984e-01,
            0.1381292052185304327114855e-01,
            0.1286794274249338667571135e-01,
            0.1190960638533075683273654e-01,
            0.1093890635919594895396767e-01,
            0.9956850427084044948237490e-02,
            0.8964458176697999432566250e-02,
            0.7962759997865495595598110e-02,
            0.6952796096469405526464256e-02,
            0.5935615630788222954183688e-02,
            0.4912276262166028130833504e-02,
            0.3883845329489294421733034e-02,
            0.2851409243213055771419126e-02,
            0.1816146398210039609502983e-02,
            0.7805332219425612457264822e-03,
        ]
    )
    OddW49 = np.array(
        [
            0.3155766036791122885809208e-01,
            0.3151046648162834771323796e-01,
            0.3143186227722154616152128e-01,
            0.3132192610907518012817474e-01,
            0.3118076756395815837033438e-01,
            0.3100852735178559535833486e-01,
            0.3080537716535627949917920e-01,
            0.3057151950920577999218210e-01,
            0.3030718749774580397961262e-01,
            0.3001264462289103447190280e-01,
            0.2968818449140509844801766e-01,
            0.2933413053222750347643324e-01,
            0.2895083567407331040373860e-01,
            0.2853868199362694972663692e-01,
            0.2809808033468091126593440e-01,
            0.2762946989859901232207604e-01,
            0.2713331780651255092639320e-01,
            0.2661011863368585130179228e-01,
            0.2606039391651548254092866e-01,
            0.2548469163265475465058230e-01,
            0.2488358565478194644598738e-01,
            0.2425767517855707823164026e-01,
            0.2360758412533789404661778e-01,
            0.2293396052025105528408320e-01,
            0.2223747584623937158435550e-01,
            0.2151882437473022381824646e-01,
            0.2077872247359421120742490e-01,
            0.2001790789308656620794778e-01,
            0.1923713903048718479867380e-01,
            0.1843719417417849927098560e-01,
            0.1761887072792438050675710e-01,
            0.1678298441613870708950299e-01,
            0.1593036847096084971103802e-01,
            0.1506187280199023331295260e-01,
            0.1417836314957944606614279e-01,
            0.1328072022265728347995425e-01,
            0.1236983882217516210343368e-01,
            0.1144662695149825376113323e-01,
            0.1051200491552474540574917e-01,
            0.9566904411326136356898158e-02,
            0.8612267615478888991732218e-02,
            0.7649046279335257935390770e-02,
            0.6678200860575098165183170e-02,
            0.5700699773395926875152328e-02,
            0.4717519037520830079689318e-02,
            0.3729643487243034749198276e-02,
            0.2738075873626878091327392e-02,
            0.1743906958219244938639563e-02,
            0.7494736467374053633626714e-03,
        ]
    )

    if l < 1 or 100 < l:
        print("")
        print("LEGENDRE_WEIGHT - Fatal error!")
        print("  1 <= L <= 100 is required.")
        # exit ( 'LEGENDRE_WEIGHT - Fatal error!' )
        # return -1

    lhalf = (l + 1) // 2

    if (l % 2) == 1:
        if lhalf < k:
            kcopy = k - lhalf
        elif lhalf == k:
            kcopy = lhalf
        else:
            kcopy = lhalf - k
    else:
        if lhalf < k:
            kcopy = k - lhalf
        else:
            kcopy = lhalf + 1 - k

    if kcopy < 1 or lhalf < kcopy:
        print("")
        print("LEGENDRE_WEIGHT - Fatal error!")
        print("  1 <= K <= (L+1)/2 is required.")
        # exit ( 'LEGENDRE_WEIGHT - Fatal error!' )
        # return -1
    #
    #  If L is odd, and K = ( L - 1 ) / 2, then it's easy.
    #
    if (l % 2) == 1 and kcopy == lhalf:
        weight = 2.0e00 / cl[l] ** 2
    elif l == 2:
        weight = EvenW1[kcopy - 1]
    elif l == 3:
        weight = OddW1[kcopy - 1]
    elif l == 4:
        weight = EvenW2[kcopy - 1]
    elif l == 5:
        weight = OddW2[kcopy - 1]
    elif l == 6:
        weight = EvenW3[kcopy - 1]
    elif l == 7:
        weight = OddW3[kcopy - 1]
    elif l == 8:
        weight = EvenW4[kcopy - 1]
    elif l == 9:
        weight = OddW4[kcopy - 1]
    elif l == 10:
        weight = EvenW5[kcopy - 1]
    elif l == 11:
        weight = OddW5[kcopy - 1]
    elif l == 12:
        weight = EvenW6[kcopy - 1]
    elif l == 13:
        weight = OddW6[kcopy - 1]
    elif l == 14:
        weight = EvenW7[kcopy - 1]
    elif l == 15:
        weight = OddW7[kcopy - 1]
    elif l == 16:
        weight = EvenW8[kcopy - 1]
    elif l == 17:
        weight = OddW8[kcopy - 1]
    elif l == 18:
        weight = EvenW9[kcopy - 1]
    elif l == 19:
        weight = OddW9[kcopy - 1]
    elif l == 20:
        weight = EvenW10[kcopy - 1]
    elif l == 21:
        weight = OddW10[kcopy - 1]
    elif l == 22:
        weight = EvenW11[kcopy - 1]
    elif l == 23:
        weight = OddW11[kcopy - 1]
    elif l == 24:
        weight = EvenW12[kcopy - 1]
    elif l == 25:
        weight = OddW12[kcopy - 1]
    elif l == 26:
        weight = EvenW13[kcopy - 1]
    elif l == 27:
        weight = OddW13[kcopy - 1]
    elif l == 28:
        weight = EvenW14[kcopy - 1]
    elif l == 29:
        weight = OddW14[kcopy - 1]
    elif l == 30:
        weight = EvenW15[kcopy - 1]
    elif l == 31:
        weight = OddW15[kcopy - 1]
    elif l == 32:
        weight = EvenW16[kcopy - 1]
    elif l == 33:
        weight = OddW16[kcopy - 1]
    elif l == 34:
        weight = EvenW17[kcopy - 1]
    elif l == 35:
        weight = OddW17[kcopy - 1]
    elif l == 36:
        weight = EvenW18[kcopy - 1]
    elif l == 37:
        weight = OddW18[kcopy - 1]
    elif l == 38:
        weight = EvenW19[kcopy - 1]
    elif l == 39:
        weight = OddW19[kcopy - 1]
    elif l == 40:
        weight = EvenW20[kcopy - 1]
    elif l == 41:
        weight = OddW20[kcopy - 1]
    elif l == 42:
        weight = EvenW21[kcopy - 1]
    elif l == 43:
        weight = OddW21[kcopy - 1]
    elif l == 44:
        weight = EvenW22[kcopy - 1]
    elif l == 45:
        weight = OddW22[kcopy - 1]
    elif l == 46:
        weight = EvenW23[kcopy - 1]
    elif l == 47:
        weight = OddW23[kcopy - 1]
    elif l == 48:
        weight = EvenW24[kcopy - 1]
    elif l == 49:
        weight = OddW24[kcopy - 1]
    elif l == 50:
        weight = EvenW25[kcopy - 1]
    elif l == 51:
        weight = OddW25[kcopy - 1]
    elif l == 52:
        weight = EvenW26[kcopy - 1]
    elif l == 53:
        weight = OddW26[kcopy - 1]
    elif l == 54:
        weight = EvenW27[kcopy - 1]
    elif l == 55:
        weight = OddW27[kcopy - 1]
    elif l == 56:
        weight = EvenW28[kcopy - 1]
    elif l == 57:
        weight = OddW28[kcopy - 1]
    elif l == 58:
        weight = EvenW29[kcopy - 1]
    elif l == 59:
        weight = OddW29[kcopy - 1]
    elif l == 60:
        weight = EvenW30[kcopy - 1]
    elif l == 61:
        weight = OddW30[kcopy - 1]
    elif l == 62:
        weight = EvenW31[kcopy - 1]
    elif l == 63:
        weight = OddW31[kcopy - 1]
    elif l == 64:
        weight = EvenW32[kcopy - 1]
    elif l == 65:
        weight = OddW32[kcopy - 1]
    elif l == 66:
        weight = EvenW33[kcopy - 1]
    elif l == 67:
        weight = OddW33[kcopy - 1]
    elif l == 68:
        weight = EvenW34[kcopy - 1]
    elif l == 69:
        weight = OddW34[kcopy - 1]
    elif l == 70:
        weight = EvenW35[kcopy - 1]
    elif l == 71:
        weight = OddW35[kcopy - 1]
    elif l == 72:
        weight = EvenW36[kcopy - 1]
    elif l == 73:
        weight = OddW36[kcopy - 1]
    elif l == 74:
        weight = EvenW37[kcopy - 1]
    elif l == 75:
        weight = OddW37[kcopy - 1]
    elif l == 76:
        weight = EvenW38[kcopy - 1]
    elif l == 77:
        weight = OddW38[kcopy - 1]
    elif l == 78:
        weight = EvenW39[kcopy - 1]
    elif l == 79:
        weight = OddW39[kcopy - 1]
    elif l == 80:
        weight = EvenW40[kcopy - 1]
    elif l == 81:
        weight = OddW40[kcopy - 1]
    elif l == 82:
        weight = EvenW41[kcopy - 1]
    elif l == 83:
        weight = OddW41[kcopy - 1]
    elif l == 84:
        weight = EvenW42[kcopy - 1]
    elif l == 85:
        weight = OddW42[kcopy - 1]
    elif l == 86:
        weight = EvenW43[kcopy - 1]
    elif l == 87:
        weight = OddW43[kcopy - 1]
    elif l == 88:
        weight = EvenW44[kcopy - 1]
    elif l == 89:
        weight = OddW44[kcopy - 1]
    elif l == 90:
        weight = EvenW45[kcopy - 1]
    elif l == 91:
        weight = OddW45[kcopy - 1]
    elif l == 92:
        weight = EvenW46[kcopy - 1]
    elif l == 93:
        weight = OddW46[kcopy - 1]
    elif l == 94:
        weight = EvenW47[kcopy - 1]
    elif l == 95:
        weight = OddW47[kcopy - 1]
    elif l == 96:
        weight = EvenW48[kcopy - 1]
    elif l == 97:
        weight = OddW48[kcopy - 1]
    elif l == 98:
        weight = EvenW49[kcopy - 1]
    elif l == 99:
        weight = OddW49[kcopy - 1]
    elif l == 100:
        weight = EvenW50[kcopy - 1]

    return weight


def legendre_weight_test():
    # *****************************************************************************80
    #
    ## LEGENDRE_WEIGHT_TEST tests LEGENDRE_WEIGHT.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 January 2016
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Reference:
    #
    #    Ignace Bogaert,
    #    Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
    #    SIAM Journal on Scientific Computing,
    #    Volume 36, Number 3, 2014, pages A1008-1026.
    #
    import platform

    print("")
    print("LEGENDRE_WEIGHT_TEST:")
    print("  Python version: %s" % (platform.python_version()))
    print("  LEGENDRE_WEIGHT returns the K-th weight for")
    print("  a Gauss Legendre rule of order L.")

    for l in range(1, 11):
        print("")
        print("  Gauss Legendre rule of order %d" % (l))
        print("")
        print("   K      Weight")
        print("")
        for k in range(1, l + 1):
            weight = legendre_weight(l, k)
            print("  %2d  %14.6g" % (k, weight))
    #
    #  Terminate.
    #
    print("")
    print("LEGENDRE_WEIGHT_TEST:")
    print("  Normal end of execution.")
    return


def timestamp():
    # *****************************************************************************80
    #
    ## TIMESTAMP prints the date as a timestamp.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 April 2013
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    None
    #
    import time

    t = time.time()
    print(time.ctime(t))

    return None


def timestamp_test():
    # *****************************************************************************80
    #
    ## TIMESTAMP_TEST tests TIMESTAMP.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    03 December 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    None
    #
    import platform

    print("")
    print("TIMESTAMP_TEST:")
    print("  Python version: %s" % (platform.python_version()))
    print("  TIMESTAMP prints a timestamp of the current date and time.")
    print("")

    timestamp()
    #
    #  Terminate.
    #
    print("")
    print("TIMESTAMP_TEST:")
    print("  Normal end of execution.")
    return


if __name__ == "__main__":
    timestamp()
    fastgl_test()
    timestamp()
