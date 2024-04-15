import numpy as np
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from flow_sim import compute_saturation_data
from flow_sim import numerical_solution_test
from flow_sim import compute_solution
    

def main():
    print('inside the main funciton')
    # numerical_solution_test()
    t = 0.1
    x = 0.6
    press, pgrad, s_wat, s_oil, u_wat, u_oil = compute_solution(t, x, nx=100)
    print('press = ' + str(press))
    print('pgrad = ' + str(pgrad))
    print('s_wat = ' + str(s_wat))
    print('s_oil = ' + str(s_oil))
    print('u_wat = ' + str(u_wat))
    print('u_oil = ' + str(u_oil))

    numx = 20
    x = np.linspace(0.0, 1.0, numx + 1)
    pv = 0.0 * x
    pg = 0.0 * x
    sw = 0.0 * x
    so = 0.0 * x
    uw = 0.0 * x
    uo = 0.0 * x
    for idx in range(numx + 1):
        press, pgrad, s_wat, s_oil, u_wat, u_oil = compute_solution(t, x[idx], nx=100)
        pv[idx] = press
        pg[idx] = pgrad
        sw[idx] = s_wat
        so[idx] = s_oil
        uw[idx] = u_wat
        uo[idx] = u_oil

    plt.figure()
    plt.plot(x, pv, c='r')
    plt.plot(x, pg, c='b')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(x, sw, c='r')
    plt.plot(x, so, c='b')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(x, uw, c='r')
    plt.plot(x, uo, c='b')
    plt.grid()
    plt.show()




    return 0


main()


