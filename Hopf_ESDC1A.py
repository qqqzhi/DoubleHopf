import numdifftools as nd
import numpy as np
from numpy import sin
from numpy import cos

"""
Specify the generator parameters
"""
ratio = 6.9

Tdp = 5.33
Tqp = 0.593
H = 6.1 * ratio
KD = 0 * ratio
Xd = 1.942 / ratio
Xq = 1.921 / ratio
Xdp = 0.374 / ratio
Xqp = Xdp
Xl = 0.214 / ratio
Rs = 0

KA1 = 900
TA1 = 0.0045
KE1 = 1.0
TE1 = 0.78
Aex1 = 0.00325
Bex1 = 0.795
TF1 = 0.69
KF1 = 0.01

KA2 = 50
TA2 = 0.006
KE2 = 1
TE2 = 0.78
Aex2 = 0.00325
Bex2 = 0.795
TF2 = 1.19
KF2 = 0.001

Z13 = 3.8E-4 + 1j * 1.216E-2  #
Z23 = 3.8E-4 + 1j * 1.184E-2  #
Z34 = 3.57E-3 + 1j * 3.362E-2  #

Z12 = Z13 + Z23 + (Z13 * Z23) / Z34;
Z24 = Z23 + Z34 + (Z23 * Z34) / Z13;
Z14 = Z13 + Z34 + (Z13 * Z34) / Z23;
Y12 = 1 / Z12
Y24 = 1 / Z24
Y14 = 1 / Z14
Y1 = Y2 = 1 / (1j * Xdp);
C = np.linalg.solve(np.array([[Y12 + Y14 + Y1, -Y12], [-Y12, Y12 + Y24 + Y2]]),
                    np.array([[Y1, 0, Y14], [0, Y2, Y24]]))
c11r, c11i, c12r, c12i, c13r, c13i = C[0][0].real, C[0][0].imag, C[0][1].real, C[0][1].imag, C[0][2].real, C[0][2].imag
c21r, c21i, c22r, c22i, c23r, c23i = C[1][0].real, C[1][0].imag, C[1][1].real, C[1][1].imag, C[1][2].real, C[1][2].imag


def f_Vdq(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    delta1 = x[2]
    w1 = x[3]
    E1 = (Edp1 + 1j * Eqp1) * np.exp(1j * (delta1 - np.pi / 2))

    Eqp2 = x[7]
    Edp2 = x[8]
    delta2 = x[9]
    w2 = x[10]
    E2 = (Edp2 + 1j * Eqp2) * np.exp(1j * (delta2 - np.pi / 2))

    Vdq = np.zeros((2, 1), dtype=complex)
    D = np.array([[np.exp(1j * (np.pi / 2 - delta1)), 0], [0, np.exp(1j * (np.pi / 2 - delta2))]], dtype=complex)
    y = np.array([[E1], [E2], [1]], dtype=complex)
    Vdq = np.matmul(np.matmul(D, C), y)
    return Vdq


def f_Id1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Vd1 = np.real(f_Vdq(x)[0])
    Vq1 = np.imag(f_Vdq(x)[0])
    return (Eqp1 - Vq1) / Xdp


def f_Id2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vd2 = np.real(f_Vdq(x)[1])
    Vq2 = np.imag(f_Vdq(x)[1])
    return (Eqp2 - Vq2) / Xdp


def f_Iq1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Vd1 = np.real(f_Vdq(x)[0])
    Vq1 = np.imag(f_Vdq(x)[0])
    return -(Edp1 - Vd1) / Xqp


def f_Iq2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vd2 = np.real(f_Vdq(x)[1])
    Vq2 = np.imag(f_Vdq(x)[1])
    return -(Edp2 - Vd2) / Xqp


def f_Eqp1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Vf1 = x[4]
    Ve1 = x[6]
    Id1 = f_Id1(x)
    return 1 / Tdp * (-Eqp1 - (Xd - Xdp) * Id1 + Ve1)


def f_Eqp2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vf2 = x[11]
    Ve2 = x[13]
    Id2 = f_Id2(x)
    return 1 / Tdp * (-Eqp2 - (Xd - Xdp) * Id2 + Ve2)


def f_Edp1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Iq1 = f_Iq1(x)
    return 1 / Tqp * (-Edp1 + (Xq - Xqp) * Iq1)


def f_Edp2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Iq2 = f_Iq2(x)
    return 1 / Tqp * (-Edp2 + (Xq - Xqp) * Iq2)


def f_delta1(x):
    return x[3] * 120 * np.pi


def f_delta2(x):
    return x[10] * 120 * np.pi


# Pe = Eq*Iq + Ed*Id = Eq*(Vd-Ed)/Xqp + Ed*(Eq-Vq)/Xdp = (Eq*Vd - Ed*Vq)/Xdp
def f_w1(x):
    #Pm = x[14]
    #Pm = 5.1338388341401240
    Pm = 5.4 # For time domain simulation
    Eqp1 = x[0]
    Edp1 = x[1]
    Vd1 = np.real(f_Vdq(x)[0])
    Vq1 = np.imag(f_Vdq(x)[0])
    Pe = (Eqp1 * Vd1 - Edp1 * Vq1) / Xdp
    return 1 / (2 * H) * (Pm - Pe)


def f_w2(x):
    #Pm = x[15]
    #Pm = 4.3440102560399287
    Pm = 4.3 # For time domain simulation
    Eqp2 = x[7]
    Edp2 = x[8]
    Vd2 = np.real(f_Vdq(x)[1])
    Vq2 = np.imag(f_Vdq(x)[1])
    Pe = (Eqp2 * Vd2 - Edp2 * Vq2) / Xdp
    return 1 / (2 * H) * (Pm - Pe)


def f_VF1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Vf1 = x[4]
    Ve1 = x[6]
    Id1 = f_Id1(x)
    return (Ve1 - Vf1) / TF1


def f_VF2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vf2 = x[11]
    Ve2 = x[13]
    Id2 = f_Id2(x)
    return (Ve2 - Vf2) / TF2


def f_VA1(x):
    #Vref = x[16]
    Vref = 1.04
    Eqp1 = x[0]
    Edp1 = x[1]
    Vf1 = x[4]
    Va1 = x[5]
    Ve1 = x[6]
    yf = KF1 / TF1 * (Ve1 - Vf1)
    Vsum = Vref - np.absolute(f_Vdq(x)[0]) - yf
    return (Vsum - Va1) / TA1


def f_VA2(x):
    #Vref = x[17]
    Vref = 1.03
    Eqp2 = x[7]
    Edp2 = x[8]
    Vf2 = x[11]
    Va2 = x[12]
    Ve2 = x[13]
    yf = KF2 / TF2 * (Ve2 - Vf2)
    Vsum = Vref - np.absolute(f_Vdq(x)[1]) - yf
    return (Vsum - Va2) / TA2


def f_VE1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Vf1 = x[4]
    Va1 = x[5]
    Ve1 = x[6]
    Vfe = KE1 * Ve1 + Aex1 * np.exp(Bex1 * Ve1)
    return (KA1 * Va1 - Vfe) / TE1


def f_VE2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vf2 = x[11]
    Va2 = x[12]
    Ve2 = x[13]
    Vfe = KE2 * Ve2 + Aex2 * np.exp(Bex2 * Ve2)
    return (KA2 * Va2 - Vfe) / TE2


# n is the length of column vector u

def T2_mat(n):
    T2 = np.eye(n ** 2, dtype=int)
    rmidx = np.triu_indices(n, 1)[1] * n + np.triu_indices(n, 1)[0]
    T2 = np.delete(T2, rmidx, 0)
    return T2


def S2_mat(n):
    S2 = np.eye(n ** 2, dtype=int)
    rmidx = np.triu_indices(n, 1)[1] * n + np.triu_indices(n, 1)[0]
    addidx = np.triu_indices(n, 1)[0] * n + np.triu_indices(n, 1)[1]
    S2[rmidx, addidx] = 1
    S2 = np.delete(S2, rmidx, 1)
    return S2


def T3_mat(n):
    Bx3 = [(i, j, k) for i in range(n) for j in range(i, n) for k in range(j, n)]  # extracted from x \otimes Bx^2
    x_Bx2 = [(i, j, k) for i in range(n) for j in range(n) for k in range(j, n)]  # x \otimes Bx^2
    Bx3_idx = [x_Bx2.index(i) for i in Bx3]
    rmidx = list(set(range(len(x_Bx2))) - set(Bx3_idx))
    rmele = [x_Bx2[i] for i in rmidx]
    rmele = [tuple(sorted(i)) for i in rmele]
    rmidx_inBx3 = [Bx3.index(i) for i in rmele]
    T3 = np.eye(n * n * (n + 1) // 2, dtype=int)
    T3 = T3[Bx3_idx]
    return T3


def S3_mat(n):
    Bx3 = [(i, j, k) for i in range(n) for j in range(i, n) for k in range(j, n)]  # extracted from x \otimes Bx^2
    x_Bx2 = [(i, j, k) for i in range(n) for j in range(n) for k in range(j, n)]  # x \otimes Bx^2
    Bx3_idx = [x_Bx2.index(i) for i in Bx3]
    rmidx = list(set(range(len(x_Bx2))) - set(Bx3_idx))
    rmele = [x_Bx2[i] for i in rmidx]
    rmele = [tuple(sorted(i)) for i in rmele]
    rmidx_inBx3 = [Bx3.index(i) for i in rmele]
    S3 = np.eye(n * n * (n + 1) // 2, dtype=int)
    S3 = S3[:, Bx3_idx]
    S3[rmidx, rmidx_inBx3] = 1
    return S3

def Trissian(f_test, x0):
    """
    This function calculates the 3rd order derivative of a function f: R^n -> R
    input: 
        f_test is the function
        x0 where the 3rd order want to be calcuated
    return: 3-D matrix
    """
    Trissian = np.zeros((x0.shape[0],x0.shape[0],x0.shape[0]))
    for i in range(x0.shape[0]):
        h = 1e-4
        xp1 = np.array(x0, copy=True) 
        xp1[i] += h
        #print(xp1)
        xp2 = np.array(x0, copy=True) 
        xp2[i] += 2*h
        #print(xp2)
        xm1 = np.array(x0, copy=True) 
        xm1[i] -= h
        #print(xm1)
        xm2 = np.array(x0, copy=True) 
        xm2[i] -= 2*h
        #print(xm2)
        Trissian[i] = (-nd.Hessian(f_test)(xp2) + 8*nd.Hessian(f_test)(xp1)- 8*nd.Hessian(f_test)(xm1) + nd.Hessian(f_test)(xm2))/(12*h)
    return Trissian


def sys_fun(x):
    fun = [f_Eqp1, f_Edp1, f_delta1, f_w1, f_VF1, f_VA1, f_VE1, f_Eqp2, f_Edp2, f_delta2, f_w2, f_VF2, f_VA2, f_VE2]
#     J = np.array([nd.Jacobian(f)(x).ravel() for f in fun])
#     J = J[:,:14]
#     lam, v = np.linalg.eig(J)
#     res = np.append(np.array([f(x).ravel() for f in fun]).ravel(), [lam[5].real,lam[6].real])
#     return res
    return np.array([f(x).ravel() for f in fun]).ravel()

all_fun = [f_Eqp1,f_Edp1,f_delta1,f_w1,f_VF1,f_VA1,f_VE1,f_Eqp2,f_Edp2,f_delta2,f_w2,f_VF2,f_VA2,f_VE2]


x = np.array([1.0586322295940347e+00, 5.3628104492194140e-01,\
              1.0725778892717863e+00, 1.9922585212178192e-31,\
              2.1610676325913563e+00, 2.4213133003525961e-03,\
              2.1610676325913563e+00, 7.0816205732775250e-01,\
              6.8184037951945065e-01, 1.3762356481251654e+00,\
              1.2001195685758683e-31, 1.4381753395147898e+00,\
              2.8967428606158897e-02, 1.4381753395147898e+00])