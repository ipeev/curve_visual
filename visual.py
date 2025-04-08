import matplotlib

from pylab import *
import math
import matplotlib.pyplot as plt

M_PI = math.pi


def set_axes_equal(ax):
    from matplotlib import cm

    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def get_data(fn):
    fd = open(fn)
    X, Y = [], []
    for d in fd:
        d = d.strip()
        if not d or "$" in d:
            continue
        x, y = list(map(float, d.split(",")[:2]))
        X.append(x)
        Y.append(y)
    return X, Y


finp = r"c:\temp\2dinp.dat"
fout = r"c:\temp\2dout.dat"
fpar = r"c:\temp\2dpar.dat"

plane = "XY"


def test1():
    X, Y = get_data(finp)
    plot(X, Y, ".")

    # X,Y = get_data(r"c:\temp\2dinpc.dat")
    # plot(X,Y)

    X, Y = get_data(fout)
    plot(X, Y, ".")

    show()


def loadPar(fpar):
    par = open(fpar).readlines()
    P = {}
    for p in par:
        p = p.strip()
        if not p or p.startswith("#"):
            continue
        p = p.split("=")
        P[p[0]] = p[1]
    return P


class ExtList(list):
    pass


def get_name(d):
    t = d.split("(")[1].split("=")[1].split(",")[0].strip()
    return t


def getElements(fn):
    fd = open(fn)
    E = []
    for d in fd:
        d = d.strip()
        if not d:
            continue
        if d.startswith("$ELE"):
            e = ExtList()
            name = get_name(d)
            e.name = name
            continue
        if d.startswith("$END"):
            ee = ExtList(e)
            ee.name = e.name
            E.append(ee)
            continue
        # print d
        nums = list(map(float, d.split(",")))
        e.append(nums)
    return E


def getXY(e):
    X, Y = [], []
    for r in e:
        if plane == "XY":
            X.append(r[0])
            Y.append(r[1])
        elif plane == "YZ":
            X.append(r[1])
            Y.append(r[2])
        elif plane == "ZX":
            X.append(r[2])
            Y.append(r[0])
        else:
            raise "plane: "
    return X, Y


def getXA(e):
    X, Y = [], []
    for r in e:
        if plane == "XY":
            X.append(r[0])
            Y.append(r[7])
        elif plane == "YZ":
            X.append(r[1])
            Y.append(r[7])
        elif plane == "ZX":
            X.append(r[2])
            Y.append(r[7])
        else:
            raise "plane: "
    return X, Y


def getXD(e):
    X, Y = [], []
    for r in e:
        if plane == "XY":
            X.append(r[0])
            Y.append(r[8])
        elif plane == "YZ":
            X.append(r[1])
            Y.append(r[8])
        elif plane == "ZX":
            X.append(r[2])
            Y.append(r[8])
        else:
            raise "plane: "
    return X, Y


def getA(e):
    A = []
    for r in e:
        A.append(r[7])
    return A


def getD(e):
    A = []
    for r in e:
        A.append(r[8])
    return A


def getR(e):
    A = []
    for r in e:
        A.append(r[3])
    return A


def getUV(e):
    X, Y = [], []
    for r in e:
        if plane == "XY":
            X.append(r[4])
            Y.append(r[5])
        elif plane == "YZ":
            X.append(r[5])
            Y.append(r[6])
        elif plane == "ZX":
            X.append(r[6])
            Y.append(r[4])
        else:
            raise "plane: "
    return X, Y


def plotPointsLine(e, sign=".-", label=None):
    X, Y = getXY(e)
    plot(X, Y, sign, label=label)


def plotPoints(e):
    X, Y = getXY(e)
    for x, y in zip(X, Y):
        # print x,y
        plot([x], [y], "o")


def plotEllipse_xyr(
    x,
    y,
    a,
    b,
    fmt="-",
    label=None,
):
    an = linspace(0, 2 * pi, 400)
    plot(x + a * cos(an), y + b * sin(an), fmt, label=label)
    plot([x], [y], "+")


def plotCircle_xyr(x, y, r, label=None):
    an = linspace(0, 2 * pi, 200)
    plot(x + r * cos(an), y + r * sin(an), label=label)
    plot([x], [y], "v")


def plotCircle(e):
    X, Y = getXY(e)
    x, y = X[0], Y[0]
    r = e[0][7]
    an = linspace(0, 2 * pi, 200)
    plot(x + r * cos(an), y + r * sin(an))
    plot([x], [y], "+")


def plotCircleR(e):
    X, Y = getXY(e)
    x, y = X[0], Y[0]
    r = e[0][3]
    an = linspace(0, 2 * pi, 200)
    plot(x + r * cos(an), y + r * sin(an))
    plot([x], [y], "+")


def plotSingleLine(e):
    X, Y = getXY(e)
    U, V = getUV(e)
    x, y = X[0], Y[0]
    u, v = U[0], V[0]
    t = linspace(-100, 100, 200)
    plot(x + u * t, y + v * t)


def plotSingleLine_around(e, p1, p2):
    print("^^", e, p1, p2)
    X, Y = getXY(e)
    U, V = getUV(e)
    x, y = X[0], Y[0]
    u, v = U[0], V[0]
    X1, Y1 = getXY([p1])
    X2, Y2 = getXY([p2])
    t = linspace(-100, 100, 200)
    plot(x + u * t, y + v * t)


def display18():
    import matplotlib as mpl

    E = getElements(finp)
    # assert (len(E)==1)
    plotPointsLine(E[0])
    # show()
    R = getElements(fout)
    # print R
    plotPointsLine(R[0])
    plotPoints(R[1])
    plotCircle(R[2])
    plotPointsLine(R[5])
    plotPointsLine(R[6])
    plotPointsLine(R[7], "+")
    plotPointsLine(R[8], "+")
    if len(R) > 3:
        plotSingleLine(R[3])
        # plotSingleLine_around(R[3], R[0][0],R[0][-1])
        pass
    if len(R) > 4:
        plotPointsLine(R[4], "+-")
    if len(R) > 10:
        # plotPointsLine(R[10], "+-")
        pass
    # plotPoints(R[9])

    gca().set_aspect("equal", adjustable="box")
    mpl.pyplot.gcf().tight_layout()
    show()


def display6():
    E = getElements(finp)
    assert len(E) == 2
    plotPointsLine(E[0], "g.-")
    plotPointsLine(E[1], "y.-")
    R = getElements(fout)
    # print R
    plotPointsLine(R[0], "r.-")
    print(R[1])
    gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    show()


def display7():
    E = getElements(finp)
    assert len(E) == 1
    plotPointsLine(E[0], "-*")

    R = getElements(fout)
    # print R
    plotPointsLine(R[0], "+-")
    par = loadPar(fpar)
    if par["plane"] == "XY":
        x = par["xline"]
        y = par["yline"]
    elif par["plane"] == "YZ":
        x = par["yline"]
        y = par["zline"]
    elif par["plane"] == "ZX":
        x = par["zline"]
        y = par["xline"]
    else:
        print("WRONG plane %s" % par["plane"])
        exit(1)
    x = float(x)
    y = float(y)
    u = sin(float(par["angle"]))
    v = cos(float(par["angle"]))
    print(x, y, u, v)
    t = linspace(-10, 10, 200)
    plot(x + u * t, y + v * t)
    show()


def display21():
    E = getElements(finp)
    assert 3 <= len(E) <= 4
    plotPointsLine(E[0], "*r")
    plotPointsLine(E[1], "-.b")
    plotPointsLine(E[2], "-.g")

    R = getElements(fout)
    # print R
    plotPointsLine(R[0], "g")
    if len(R) >= 3:
        plotPointsLine(R[1], ".y")
        plotPointsLine(R[2], ".y")
    # plotPointsLine(R[3],"b")
    # K=getElements(r"c:\temp\Ok.dat")
    # plotPointsLine(K[0])
    if len(E) == 4:
        print("plotPointsLine(E[3])")
        plotPointsLine(E[3])
    # print R[1]
    show()


def plotVectors_d_color(e, width=0.0015, scale=1):
    X, Y = getXY(e)
    U, V = getUV(e)
    A = getA(e)
    D = getD(e)
    X1, Y1, U1, V1 = [], [], [], []
    X2, Y2, U2, V2 = [], [], [], []

    for a in A:
        if a:
            for k in range(len(U)):
                if D[k] != 0:
                    X2.append(X[k])
                    Y2.append(Y[k])
                    U2.append(U[k] * D[k])
                    V2.append(V[k] * D[k])
                else:
                    X1.append(X[k])
                    Y1.append(Y[k])
                    U1.append(U[k] * A[k])
                    V1.append(V[k] * A[k])
    print((len(X1), len(Y1), len(U1), len(V1)))
    print((len(X2), len(Y2), len(U2), len(V2)))
    plt.quiver(
        X1,
        Y1,
        U1,
        V1,
        width=width,
        angles="xy",
        scale_units="xy",
        scale=scale,
        color="b",
    )
    plt.quiver(
        X2,
        Y2,
        U2,
        V2,
        width=width,
        angles="xy",
        scale_units="xy",
        scale=scale,
        color="r",
    )


def plotVectors_r_color(e, width=0.0015, scale=1):
    X, Y = getXY(e)
    U, V = getUV(e)
    A = getA(e)
    R = getR(e)
    X1, Y1, U1, V1 = [], [], [], []
    X2, Y2, U2, V2 = [], [], [], []

    for a in A:
        if a:
            for k in range(len(U)):
                X2.append(X[k])
                Y2.append(Y[k])
                U2.append(U[k] * R[k])
                V2.append(V[k] * R[k])

    print((len(X1), len(Y1), len(U1), len(V1)))
    print((len(X2), len(Y2), len(U2), len(V2)))
    plt.quiver(
        X1,
        Y1,
        U1,
        V1,
        width=width,
        angles="xy",
        scale_units="xy",
        scale=scale,
        color="b",
    )
    plt.quiver(
        X2,
        Y2,
        U2,
        V2,
        width=width,
        angles="xy",
        scale_units="xy",
        scale=scale,
        color="r",
    )


def plotVectors(e, width=0.0015, scale=1):
    X, Y = getXY(e)
    U, V = getUV(e)
    A = getA(e)
    for a in A:
        if a:
            for k in range(len(U)):
                U[k] *= A[k]
                V[k] *= A[k]
            break
    plt.quiver(
        X, Y, U, V, width=width, angles="xy", scale_units="xy", scale=scale, color="b"
    )


def display4():
    E = getElements(finp)
    assert 2 == len(E)
    plotPointsLine(E[0], "g,-")
    plotPointsLine(E[1], "b,-")

    R = getElements(fout)
    # print R
    plotPointsLine(R[0], "*r")
    plotVectors(R[0])
    # K=getElements(r"c:\temp\Ok.dat")
    # plotPointsLine(K[0])
    if len(R) == 2:
        plotPointsLine(R[1], "*r-")
    # print R[1]
    # plt.arrow(0,0,10,10,shape='right',lw=3,fill=True)
    show()


def move_xy(T, dx, dy):
    R = [[a[0] + dx, a[1] + dy] + a[2:] for a in T]
    return R


def display19():
    E = getElements(finp)

    plotPointsLine(E[0])

    R = getElements(fout)
    # print R

    plotPointsLine(move_xy(R[0], 1, -1))
    plotPointsLine(R[0])

    show()


def display22():
    E = getElements(finp)

    plotPointsLine(E[0], "g")
    plotPointsLine(E[1], "g")

    R = getElements(fout)
    plotPointsLine(R[0], "r")

    show()


def display23():
    # E=getElements(finp)

    # plotPointsLine(E[0])

    R = getElements(fout)
    plotPointsLine(R[0])

    X, Y, D1, D2 = [], [], [], []
    for p in R[0]:
        X.append(p[0])
        D1.append(p[7])
    for p in R[1]:
        D2.append(p[7])

    plot(X, D1)
    plot(X, D2, ".-")
    # plotPointsLine(move_xy(R[0],1,-1))

    show()


def display1():
    E = getElements(finp)

    # plotPointsLine(E[0],"b*-")
    fig = plt.figure()
    gca().set_aspect(
        "equal",
    )  # adjustable='box')

    R = getElements(fout)
    # plotPointsLine(E[0],"r.-")
    plotPointsLine(R[0], "r.-")
    plotVectors(R[0])

    # ax = fig.add_subplot(111, projection='3d')
    # set_axes_equal(ax)
    show()


def display13():
    E = getElements(finp)

    # plotPointsLine(E[0],"b*-")
    fig = plt.figure()
    gca().set_aspect(
        "equal",
    )  # adjustable='box')

    R = getElements(fout)
    X, A = [], []
    for i, a in enumerate(R[0]):
        A.append(a[7] * 100000)
        X.append(i)
    # print(A)
    # plotPointsLine(E[0],"r.-")
    plot(X, A, "r.")

    # ax = fig.add_subplot(111, projection='3d')
    # set_axes_equal(ax)
    show()


def display3():
    E = getElements(finp)

    plotPointsLine(E[0], "g*-")

    R = getElements(fout)
    plotPointsLine(R[0], "r.-")
    plotVectors(R[0], width=0.0015, scale=10.05)

    show()


def display24():
    E = getElements(finp)

    plotPointsLine(E[0])

    R = getElements(fout)
    plotPointsLine(R[0], "r")

    show()


def display25():
    E = getElements(finp)

    plotPointsLine(E[0])

    R = getElements(fout)
    plotPointsLine(R[0])
    gca().set_aspect("equal", adjustable="box")
    show()


def angle_between_points(c, p1, p2):
    cx, cy = c
    x1, y1 = p1
    x2, y2 = p2
    a = sqrt((cy - y1) ** 2 + (cx - x1) ** 2)
    b = sqrt((cy - y2) ** 2 + (cx - x2) ** 2)
    c = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    cos_C = (a * a + b * b - c * c) / (2 * a * b)
    # print cos_C, a,b,c, a*a + b*b - c*c
    a_rad = math.acos(cos_C)
    a_gra = (a_rad / M_PI) * 180
    return a_gra


def calc_net_dim(R, X, Y, N):
    xmin, ymin = min(X) - R, min(Y) - R
    xmax, ymax = max(X) + R, max(Y) + R
    xs, ys = (xmax - xmin) / N, (ymax - ymin) / N
    return [xmin, xmax, xs], [ymin, ymax, ys]


def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def curve_dist(x, y, X, Y):
    dm = None
    for k in range(len(X)):
        d = dist(x, y, X[k], Y[k])
        if d < dm or dm == None:
            dm = d
    return dm


def curve_in():
    E = getElements(finp)
    X, Y = getXY(E[0])
    plotPointsLine(E[0])
    R = getElements(fout)

    X, Y = getXY(R[10])
    plotPointsLine(R[10])
    show()


def display26():
    E = getElements(finp)

    # plotPointsLine(E[0])

    R = getElements(fout)
    plotPointsLine(R[0])

    show()


def display12():
    E = getElements(finp)

    plotPointsLine(E[0], ".r")

    R = getElements(fout)
    plotPointsLine(R[0], "g")
    plotVectors(R[0], width=0.005, scale=0.02)
    plotCircle(R[1])
    gca().set_aspect("equal", adjustable="box")
    show()


def display15():
    E = getElements(finp)

    plotPointsLine(E[0], "r+-")

    R = getElements(fout)
    plotPointsLine(R[1], "g*")

    print(R[1])
    show()


def display28():
    E = getElements(finp)

    plotPointsLine(E[0], "r")

    R = getElements(fout)
    plotPointsLine(R[0], "g*")

    show()


def display29():
    E = getElements(finp)

    plotPointsLine(E[0], ".r")

    # R=getElements(fout)
    # plotPointsLine(R[0],"g*")
    show()


def display30():
    global plane
    plane = "XY"
    E = getElements(finp)

    plotPointsLine(E[0], "r")

    R = getElements(fout)
    plotPointsLine(R[0], "g*")
    show()


def display41():
    E = getElements(finp)

    plotPointsLine(E[0], "b.")
    plotPointsLine(E[1], "r.-")

    R = getElements(fout)
    plotPointsLine(R[0], "g*-")
    plotPointsLine(R[1], "c.")

    show()


def display42():
    global plane
    plane = "XY"
    E = getElements(finp)

    # plotPointsLine(E[0],"r-*")
    plotPointsLine(E[1], "g*-")

    plotVectors(E[1], width=0.0005, scale=1)

    R = getElements(fout)
    plotPointsLine(R[0], "y*-")
    plotVectors(R[0], width=0.0005, scale=1)
    # plotPointsLine(R[1],"b.")
    show()


def get_points_file_xyz(fn):
    fd = open(fn)
    E = []
    for d in fd:
        d = d.strip()
        if not d:
            continue
        if d.startswith("L"):
            continue

        # print(d)
        nums = list(map(float, d.split()[:3]))
        E.append(nums)
    return E


def getXYZ(e):
    X, Y, Z = [], [], []
    for r in e:
        X.append(r[0])
        Y.append(r[1])
        Z.append(r[2])
    return X, Y, Z


def plot3D(
    ax,
    E,
    sign=".",
):
    X, Y, Z = getXYZ(E)
    p = ax.plot(X, Y, Z, sign)
    return p


def display16():
    E = getElements(finp)

    plotPointsLine(E[0], "r*")

    R = getElements(fout)
    for r in R:
        if r.name.startswith("LINE$FIT"):
            plotSingleLine(r)
        elif r.name.startswith("CIRCLE$FIT"):
            plotCircle(r)
        else:
            plotPointsLine(r, "g-")

    show()


def display31():
    E = getElements(finp)
    assert len(E) == 2
    plotPointsLine(E[0], label="Nominal")
    plotPointsLine(E[1], label="Measured")
    R = getElements(fout)
    # print R
    plotPointsLine(R[0], label="STRETCH")
    # plotCircle(R[1])
    # plotPointsLine(R[1])
    # plotCircle(R[3])
    # plotPointsLine(R[4])
    # plotPoints(R[1])
    # plotCircle(R[2])
    # if len(R) > 3:
    #    plotSingleLine(R[3])
    # if len(R) > 4:
    #    plotPointsLine(R[4],"+-")
    legend()
    show()


def display10():
    E = getElements(finp)
    # assert (len(E)==1)
    plotPointsLine(E[0], label="Input")
    par = loadPar(fpar)
    if par["plane"] == "XY":
        x = par["xstart"]
        y = par["ystart"]
    elif par["plane"] == "YZ":
        x = par["ystart"]
        y = par["zstart"]
    elif par["plane"] == "ZX":
        x = par["zstart"]
        y = par["xstart"]
    else:
        print("WRONG plane %s" % par["plane"])
        exit(1)
    x, y = float(x), float(y)
    r = float(par["diameter"]) / 2.0
    plotCircle_xyr(x, y, r)
    R = getElements(fout)
    # print R
    X, Y = getXY(R[0])
    xr, yr = X[2], Y[2]
    plotCircle_xyr(xr, yr, r)
    plotPointsLine(R[0], label="RESULT")
    if len(R) > 1:
        plotPointsLine(R[1], label="DEBUG")
    # plotPointsLine(E[1], label="Measured")
    # R=getElements(fout)
    # print R
    # plotPointsLine(R[0], label="STRETCH")
    # plotCircle(R[1])
    # plotPointsLine(R[1])
    # plotCircle(R[3])
    # plotPointsLine(R[4])
    # plotPoints(R[1])
    # plotCircle(R[2])
    # if len(R) > 3:
    #    plotSingleLine(R[3])
    # if len(R) > 4:
    #    plotPointsLine(R[4],"+-")
    legend()
    show()


def display():
    global plane
    par = loadPar(fpar)
    print(par)
    if "plane" in par:
        plane = par["plane"]
    else:
        plane = "XY"
    # getattr(sys.modules["__main__"], "display" + par["function"] )()
    fnum = par["function"] if "function" in par else par["Function"]
    globals()["display" + fnum]()


if __name__ == "__main__":
    import matplotlib

    print(matplotlib.__version__)
    display()
