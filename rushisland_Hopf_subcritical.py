import numpy as np
from numpy import sin
from numpy import cos
import numdifftools as nd
"""
Specify the generator parameters
"""
x = np.array([9.8857022079307244e-01,  6.6765166154557409e-01,
        1.4327991070863149e+00, -6.2012602999812389e-27,
        5.5853319944416731e+00,  5.5853319944416731e+00,
        2.6698652981865929e+00,  1.0107960956515956e+00,
        6.1341539935903511e-01,  1.3167338461767972e+00,
        2.9090465583649489e-27,  5.0705619655303469e+00,
        5.0705619655303469e+00,  2.4249037561439817e+00,])

KA1 = 500
TA1 = 0.045
KE1 = 1.0
TE1 = 0.78
KF1 = 0.0165
TF1 = 0.69
Aex1 = 0.00325
Bex1 = 0.795
KC1 = 0.156
KD1 = 1.1792

KA2 = 300
TA2 = 2.0
KE2 = 1.0
TE2 = 0.78
KF2 = 0.0042
TF2 = 1.0
Aex2 = 0.00325
Bex2 = 0.795
KC2 = 0.156
KD2 = 1.1792

ratio = 6.9

Tdp = 5.33
Tqp = 0.593
H = 6.1*ratio
# KD = 0*ratio
Xd = 1.942/ratio
Xq = 1.921/ratio
Xdp = Xqp = 0.374/ratio
Xl = 0.214/ratio
# Rs = 0

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



def f_Efd(Ve, XadIfd, Kc):
    return Ve - 0.577 * Kc * XadIfd
    #print("Ve=",Ve)
#     if Ve <= 0:
#         Efd = 0
#     else:
#         IN = Kc * XadIfd / Ve
#         #print("IN=",IN)
#         if IN <= 0:
#             Efd = Ve
#         elif IN <= 0.433:
#             Efd = Ve - 0.577 * Kc * XadIfd
#         elif IN < 0.75:
#             Efd = np.sqrt(0.75 * Ve ** 2 - (Kc * XadIfd) ** 2)
#         elif IN <= 1:
#             Efd = 1.732 * Ve - 1.732 * Kc * XadIfd
#         else:
#             Efd = 0
#     return Efd


def f_Id1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Vd1 = np.real(f_Vdq(x)[0])
    Vq1 = np.imag(f_Vdq(x)[0])
    return (Eqp1 - Vq1)/Xdp


def f_Id2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vd2 = np.real(f_Vdq(x)[1])
    Vq2 = np.imag(f_Vdq(x)[1])
    return (Eqp2 - Vq2)/Xdp
    

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
    XadIfd = Eqp1 + (Xd - Xdp) * Id1
    Efd = f_Efd(Ve1, XadIfd, KC1)
    return 1 / Tdp * (-Eqp1 - (Xd - Xdp) * Id1 + Efd)


def f_Eqp2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vf2 = x[11]
    Ve2 = x[13]
    Id2 = f_Id2(x)
    XadIfd = Eqp2 + (Xd - Xdp) * Id2
    Efd = f_Efd(Ve2, XadIfd, KC2)
    return 1 / Tdp * (-Eqp2 - (Xd - Xdp) * Id2 + Efd)


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




def f_VF1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Vf1 = x[4]
    Ve1 = x[6]
    Id1 = f_Id1(x)
    XadIfd = Eqp1 + (Xd - Xdp) * Id1
    Vfe = KD1 * XadIfd + KE1 * Ve1 + Aex1 * np.exp(Bex1 * Ve1)
    return (Vfe - Vf1) / TF1


def f_VF2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vf2 = x[11]
    Ve2 = x[13]
    Id2 = f_Id2(x)
    XadIfd = Eqp2 + (Xd - Xdp) * Id2
    Vfe = KD2 * XadIfd + KE2 * Ve2 + Aex2 * np.exp(Bex2 * Ve2)
    return (Vfe - Vf2) / TF2


def f_VE1(x):
    Eqp1 = x[0]
    Edp1 = x[1]
    Vf1 = x[4]
    Va1 = x[5]
#     Va1 = min(Va1, 25)
#     Va1 = max(Va1, -25)
    Ve1 = x[6]
    Id1 = f_Id1(x)
    XadIfd = Eqp1 + (Xd - Xdp) * Id1
    Vfe = KD1 * XadIfd + KE1 * Ve1 + Aex1 * np.exp(Bex1 * Ve1)
#     if Ve1 < 0:
#         return np.array([0.0])
#     else:
#         return (Va1 - Vfe) / TE1
    return (Va1 - Vfe) / TE1

def f_VE2(x):
    Eqp2 = x[7]
    Edp2 = x[8]
    Vf2 = x[11]
    Va2 = x[12]
#     Va2 = min(Va2, 20)
#     Va2 = max(Va2, -10)
    Ve2 = x[13]
    Id2 = f_Id2(x)
    XadIfd = Eqp2 + (Xd - Xdp) * Id2
    Vfe = KD2 * XadIfd + KE2 * Ve2 + Aex2 * np.exp(Bex2 * Ve2)
#     if Ve2 < 0:
#         return np.array([0.0])
#     else:
#         return (Va2 - Vfe) / TE2
    return (Va2 - Vfe) / TE2



def f_w1(x):
#     Pm1 = x[14]
    Eqp1 = x[0]
    Edp1 = x[1]
    Vd1 = np.real(f_Vdq(x)[0])
    Vq1 = np.imag(f_Vdq(x)[0])
    Pe = (Eqp1*Vd1 - Edp1*Vq1)/Xdp
    return 1 / (2 * H) * (7.2357560754961314 - Pe) #7.2357560754961314


def f_w2(x):
#     Pm2 = x[15]
    Eqp2 = x[7]
    Edp2 = x[8]
    Vd2 = np.real(f_Vdq(x)[1])
    Vq2 = np.imag(f_Vdq(x)[1])
    Pe = (Eqp2*Vd2 - Edp2*Vq2)/Xdp
    return 1 / (2 * H) * (6.0421477732317959 - Pe) #6.0421477732317959 


def f_VA1(x):
#     Vref1 = x[14]  
    Eqp1 = x[0]
    Edp1 = x[1]
    Vf1 = x[4]
    Va1 = x[5]
    Ve1 = x[6]
    Id1 = f_Id1(x)
    XadIfd = Eqp1 + (Xd - Xdp) * Id1
    Vfe = KD1 * XadIfd + KE1 * Ve1 + Aex1 * np.exp(Bex1 * Ve1)
    yf = KF1 / TF1 * (Vfe - Vf1)
    Vsum = 1.0586038003890010 - np.absolute(f_Vdq(x)[0]) - yf
    return (KA1 * Vsum - Va1) / TA1


def f_VA2(x):
#     Vref2 = x[15]
    Eqp2 = x[7]
    Edp2 = x[8]
    Vf2 = x[11]
    Va2 = x[12]
    Ve2 = x[13]
    Id2 = f_Id2(x)
    XadIfd = Eqp2 + (Xd - Xdp) * Id2
    Vfe = KD2 * XadIfd + KE2 * Ve2 + Aex2 * np.exp(Bex2 * Ve2)
    yf = KF2 / TF2 * (Vfe - Vf2)
    Vsum = 1.0659163034132286 - np.absolute(f_Vdq(x)[1]) - yf
    return (KA2 * Vsum - Va2) / TA2


def sys_fun(x):
    fun = [f_Eqp1, f_Edp1, f_delta1, f_w1, f_VF1, f_VA1, f_VE1, f_Eqp2, f_Edp2, f_delta2, f_w2, f_VF2, f_VA2, f_VE2]
    
#     J = np.array([nd.Jacobian(f)(x).ravel() for f in fun])
#     J = J[:,:14]
#     lam, v = np.linalg.eig(J)
#     print(lam)
#     res = np.append(np.array([f(x).ravel() for f in fun]).ravel(), [lam[8].real,lam[9].real])
#     return res

    return np.array([f(x).ravel() for f in fun]).ravel()

all_fun = [f_Eqp1,f_Edp1,f_delta1,f_w1,f_VF1,f_VA1,f_VE1,f_Eqp2,f_Edp2,f_delta2,f_w2,f_VF2,f_VA2,f_VE2]

     


######################################################################################################
# ====================================================================================================
# =========================================== DERIVATIVE =============================================
# ====================================================================================================
######################################################################################################



def d_Vd1(x, order):
    Eq1_idx, Ed1_idx, delta1_idx = 0, 1, 2
    Eq2_idx, Ed2_idx, delta2_idx = 7, 8, 9
    Eq1, Ed1, de1 = x[Eq1_idx], x[Ed1_idx], x[delta1_idx]
    Eq2, Ed2, de2 = x[Eq2_idx], x[Ed2_idx], x[delta2_idx]
    if order == 1:
        dVd1 = np.zeros(x.shape[0])
        dVd1[Eq1_idx] = -c11i
        dVd1[Ed1_idx] = c11r
        dVd1[delta1_idx] = c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) + c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) + c13r * cos(de1) + c13i * sin(de1)
        dVd1[Eq2_idx] = c12r * sin(de1 - de2) - c12i * cos(de1 - de2)
        dVd1[Ed2_idx] = c12r * cos(de1 - de2) + c12i * sin(de1 - de2)
        dVd1[delta2_idx] = - c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))
        return dVd1
    if order == 2:
        dVd = np.zeros((x.shape[0], x.shape[0]))
        dVd[delta1_idx][delta1_idx] = c12i * (Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) - c12r * (
                    Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) + c13i * cos(de1) - c13r * sin(de1)
        dVd[Eq2_idx][delta1_idx] = c12r * cos(de1 - de2) + c12i * sin(de1 - de2)
        dVd[Ed2_idx][delta1_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)
        dVd[delta2_idx][delta1_idx] = c12r * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12i * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))

        dVd[delta1_idx][Eq2_idx] = c12r * cos(de1 - de2) + c12i * sin(de1 - de2)
        dVd[delta2_idx][Eq2_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)

        dVd[delta1_idx][Ed2_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)
        dVd[delta2_idx][Ed2_idx] = c12r * sin(de1 - de2) - c12i * cos(de1 - de2)

        dVd[delta1_idx][delta2_idx] = c12r * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12i * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))
        dVd[Eq2_idx][delta2_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)
        dVd[Ed2_idx][delta2_idx] = c12r * sin(de1 - de2) - c12i * cos(de1 - de2)
        dVd[delta2_idx][delta2_idx] = c12i * (Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) - c12r * (
                    Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2))
        return dVd
    if order == 3:
        dVd = np.zeros((x.shape[0], x.shape[0], x.shape[0]))
        dVd[delta1_idx][delta1_idx][delta1_idx] = - c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) - c13r * cos(de1) - c13i * sin(de1)
        dVd[Eq2_idx][delta1_idx][delta1_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)
        dVd[Ed2_idx][delta1_idx][delta1_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)
        dVd[delta2_idx][delta1_idx][delta1_idx] = c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) + c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))
        dVd[delta1_idx][Eq2_idx][delta1_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)
        dVd[delta2_idx][Eq2_idx][delta1_idx] = c12r * sin(de1 - de2) - c12i * cos(de1 - de2)
        dVd[delta1_idx][Ed2_idx][delta1_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)
        dVd[delta2_idx][Ed2_idx][delta1_idx] = c12r * cos(de1 - de2) + c12i * sin(de1 - de2)
        dVd[delta1_idx][delta2_idx][delta1_idx] = dVd[delta2_idx][delta1_idx][delta1_idx]
        dVd[Eq2_idx][delta2_idx][delta1_idx] = dVd[delta2_idx][Eq2_idx][delta1_idx]
        dVd[Ed2_idx][delta2_idx][delta1_idx] = dVd[delta2_idx][Ed2_idx][delta1_idx]
        dVd[delta2_idx][delta2_idx][delta1_idx] = - c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))

        dVd[delta1_idx][delta1_idx][Eq2_idx] = dVd[delta1_idx][Eq2_idx][delta1_idx]
        dVd[delta2_idx][delta1_idx][Eq2_idx] = dVd[delta2_idx][Eq2_idx][delta1_idx]
        dVd[delta1_idx][delta2_idx][Eq2_idx] = dVd[delta2_idx][Eq2_idx][delta1_idx]
        dVd[delta2_idx][delta2_idx][Eq2_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)

        dVd[delta1_idx][delta1_idx][Ed2_idx] = dVd[Ed2_idx][delta1_idx][delta1_idx]
        dVd[delta2_idx][delta1_idx][Ed2_idx] = dVd[delta2_idx][Ed2_idx][delta1_idx]
        dVd[delta1_idx][delta2_idx][Ed2_idx] = dVd[delta2_idx][Ed2_idx][delta1_idx]
        dVd[delta2_idx][delta2_idx][Ed2_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)

        dVd[delta1_idx][delta1_idx][delta2_idx] = dVd[delta1_idx][delta2_idx][delta1_idx]
        dVd[Eq2_idx][delta1_idx][delta2_idx] = dVd[Eq2_idx][delta2_idx][delta1_idx]
        dVd[Ed2_idx][delta1_idx][delta2_idx] = dVd[Ed2_idx][delta2_idx][delta1_idx]
        dVd[delta2_idx][delta1_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][delta1_idx]
        dVd[delta1_idx][Eq2_idx][delta2_idx] = dVd[Eq2_idx][delta2_idx][delta1_idx]
        dVd[delta2_idx][Eq2_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][Eq2_idx]
        dVd[delta1_idx][Ed2_idx][delta2_idx] = dVd[Ed2_idx][delta2_idx][delta1_idx]
        dVd[delta2_idx][Ed2_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][Ed2_idx]
        dVd[delta1_idx][delta2_idx][delta2_idx] = dVd[delta2_idx][delta1_idx][delta2_idx]
        dVd[Eq2_idx][delta2_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][Eq2_idx]
        dVd[Ed2_idx][delta2_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][Ed2_idx]
        dVd[delta2_idx][delta2_idx][delta2_idx] = c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) + c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))
        return dVd


def d_Vq1(x, order):
    Eq1_idx, Ed1_idx, delta1_idx = 0, 1, 2
    Eq2_idx, Ed2_idx, delta2_idx = 7, 8, 9
    Eq1, Ed1, de1 = x[Eq1_idx], x[Ed1_idx], x[delta1_idx]
    Eq2, Ed2, de2 = x[Eq2_idx], x[Ed2_idx], x[delta2_idx]
    if order == 1:
        dVq1 = np.zeros(x.shape[0])
        dVq1[Eq1_idx] = c11r
        dVq1[Ed1_idx] = c11i
        dVq1[delta1_idx] = c12i * (Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) - c12r * (
                    Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) + c13i * cos(de1) - c13r * sin(de1)
        dVq1[Eq2_idx] = c12r * cos(de1 - de2) + c12i * sin(de1 - de2)
        dVq1[Ed2_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)
        dVq1[delta2_idx] = c12r * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12i * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))
        return dVq1
    if order == 2:
        dVq = np.zeros((x.shape[0], x.shape[0]))
        dVq[delta1_idx][delta1_idx] = - c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) - c13r * cos(de1) - c13i * sin(de1)
        dVq[Eq2_idx][delta1_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)
        dVq[Ed2_idx][delta1_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)
        dVq[delta2_idx][delta1_idx] = c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) + c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))

        dVq[delta1_idx][Eq2_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)
        dVq[delta2_idx][Eq2_idx] = c12r * sin(de1 - de2) - c12i * cos(de1 - de2)

        dVq[delta1_idx][Ed2_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)
        dVq[delta2_idx][Ed2_idx] = c12r * cos(de1 - de2) + c12i * sin(de1 - de2)

        dVq[delta1_idx][delta2_idx] = c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) + c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))
        dVq[Eq2_idx][delta2_idx] = c12r * sin(de1 - de2) - c12i * cos(de1 - de2)
        dVq[Ed2_idx][delta2_idx] = c12r * cos(de1 - de2) + c12i * sin(de1 - de2)
        dVq[delta2_idx][delta2_idx] = - c12i * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12r * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))
        return dVq
    if order == 3:
        dVq = np.zeros((x.shape[0], x.shape[0], x.shape[0]))
        dVq[delta1_idx][delta1_idx][delta1_idx] = c12r * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12i * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) - c13i * cos(de1) + c13r * sin(de1)
        dVq[Eq2_idx][delta1_idx][delta1_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)
        dVq[Ed2_idx][delta1_idx][delta1_idx] = c12r * sin(de1 - de2) - c12i * cos(de1 - de2)
        dVq[delta2_idx][delta1_idx][delta1_idx] = c12i * (Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) - c12r * (
                    Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2))
        dVq[delta1_idx][Eq2_idx][delta1_idx] = dVq[Eq2_idx][delta1_idx][delta1_idx]
        dVq[delta2_idx][Eq2_idx][delta1_idx] = c12r * cos(de1 - de2) + c12i * sin(de1 - de2)
        dVq[delta1_idx][Ed2_idx][delta1_idx] = dVq[Ed2_idx][delta1_idx][delta1_idx]
        dVq[delta2_idx][Ed2_idx][delta1_idx] = c12i * cos(de1 - de2) - c12r * sin(de1 - de2)
        dVq[delta1_idx][delta2_idx][delta1_idx] = dVq[delta2_idx][delta1_idx][delta1_idx]
        dVq[Eq2_idx][delta2_idx][delta1_idx] = dVq[delta2_idx][Eq2_idx][delta1_idx]
        dVq[Ed2_idx][delta2_idx][delta1_idx] = dVq[delta2_idx][Ed2_idx][delta1_idx]
        dVq[delta2_idx][delta2_idx][delta1_idx] = c12r * (Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2)) - c12i * (
                    Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2))

        dVq[delta1_idx][delta1_idx][Eq2_idx] = dVq[Eq2_idx][delta1_idx][delta1_idx]
        dVq[delta2_idx][delta1_idx][Eq2_idx] = dVq[delta2_idx][Eq2_idx][delta1_idx]
        dVq[delta1_idx][delta2_idx][Eq2_idx] = dVq[delta2_idx][Eq2_idx][delta1_idx]
        dVq[delta2_idx][delta2_idx][Eq2_idx] = - c12r * cos(de1 - de2) - c12i * sin(de1 - de2)

        dVq[delta1_idx][delta1_idx][Ed2_idx] = dVq[Ed2_idx][delta1_idx][delta1_idx]
        dVq[delta2_idx][delta1_idx][Ed2_idx] = dVq[Ed2_idx][delta2_idx][delta1_idx]
        dVq[delta1_idx][delta2_idx][Ed2_idx] = dVq[Ed2_idx][delta2_idx][delta1_idx]
        dVq[delta2_idx][delta2_idx][Ed2_idx] = c12r * sin(de1 - de2) - c12i * cos(de1 - de2)

        dVq[delta1_idx][delta1_idx][delta2_idx] = dVq[delta2_idx][delta1_idx][delta1_idx]
        dVq[Eq2_idx][delta1_idx][delta2_idx] = dVq[delta2_idx][Eq2_idx][delta1_idx]
        dVq[Ed2_idx][delta1_idx][delta2_idx] = dVq[delta2_idx][Ed2_idx][delta1_idx]
        dVq[delta2_idx][delta1_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][delta1_idx]
        dVq[delta1_idx][Eq2_idx][delta2_idx] = dVq[delta2_idx][delta1_idx][Eq2_idx]
        dVq[delta2_idx][Eq2_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][Eq2_idx]
        dVq[delta1_idx][Ed2_idx][delta2_idx] = dVq[delta2_idx][delta1_idx][Ed2_idx]
        dVq[delta2_idx][Ed2_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][Ed2_idx]
        dVq[delta1_idx][delta2_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][delta1_idx]
        dVq[Eq2_idx][delta2_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][Eq2_idx]
        dVq[Ed2_idx][delta2_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][Ed2_idx]
        dVq[delta2_idx][delta2_idx][delta2_idx] = c12i * (Eq2 * cos(de1 - de2) - Ed2 * sin(de1 - de2)) - c12r * (
                    Ed2 * cos(de1 - de2) + Eq2 * sin(de1 - de2))
        return dVq


def d_Vd2(x, order):
    Eq1_idx, Ed1_idx, delta1_idx = 0, 1, 2
    Eq2_idx, Ed2_idx, delta2_idx = 7, 8, 9
    Eq1, Ed1, de1 = x[Eq1_idx], x[Ed1_idx], x[delta1_idx]
    Eq2, Ed2, de2 = x[Eq2_idx], x[Ed2_idx], x[delta2_idx]
    if order == 1:
        dVd2 = np.zeros(x.shape[0])
        dVd2[Eq1_idx] = - c21i * cos(de1 - de2) - c21r * sin(de1 - de2)
        dVd2[Ed1_idx] = c21r * cos(de1 - de2) - c21i * sin(de1 - de2)
        dVd2[delta1_idx] = - c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))
        dVd2[Eq2_idx] = -c22i
        dVd2[Ed2_idx] = c22r
        dVd2[delta2_idx] = c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) + c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) + c23r * cos(de2) + c23i * sin(de2)
        return dVd2
    if order == 2:
        dVd = np.zeros((x.shape[0], x.shape[0]))
        dVd[delta1_idx][Eq1_idx] = c21i * sin(de1 - de2) - c21r * cos(de1 - de2)
        dVd[delta2_idx][Eq1_idx] = c21r * cos(de1 - de2) - c21i * sin(de1 - de2)

        dVd[delta1_idx][Ed1_idx] = - c21i * cos(de1 - de2) - c21r * sin(de1 - de2)
        dVd[delta2_idx][Ed1_idx] = c21i * cos(de1 - de2) + c21r * sin(de1 - de2)

        dVd[Eq1_idx][delta1_idx] = c21i * sin(de1 - de2) - c21r * cos(de1 - de2)
        dVd[Ed1_idx][delta1_idx] = - c21i * cos(de1 - de2) - c21r * sin(de1 - de2)
        dVd[delta1_idx][delta1_idx] = c21i * (Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) - c21r * (
                    Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2))
        dVd[delta2_idx][delta1_idx] = c21r * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21i * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))

        dVd[Eq1_idx][delta2_idx] = c21r * cos(de1 - de2) - c21i * sin(de1 - de2)
        dVd[Ed1_idx][delta2_idx] = c21i * cos(de1 - de2) + c21r * sin(de1 - de2)
        dVd[delta1_idx][delta2_idx] = c21r * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21i * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))
        dVd[delta2_idx][delta2_idx] = c21i * (Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) - c21r * (
                    Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) + c23i * cos(de2) - c23r * sin(de2)
        return dVd
    if order == 3:
        dVd = np.zeros((x.shape[0], x.shape[0], x.shape[0]))
        dVd[delta1_idx][delta1_idx][Eq1_idx] = c21i * cos(de1 - de2) + c21r * sin(de1 - de2)
        dVd[delta2_idx][delta1_idx][Eq1_idx] = - c21i * cos(de1 - de2) - c21r * sin(de1 - de2)
        dVd[delta1_idx][delta2_idx][Eq1_idx] = dVd[delta2_idx][delta1_idx][Eq1_idx]
        dVd[delta2_idx][delta2_idx][Eq1_idx] = c21i * cos(de1 - de2) + c21r * sin(de1 - de2)

        dVd[delta1_idx][delta1_idx][Ed1_idx] = c21i * sin(de1 - de2) - c21r * cos(de1 - de2)
        dVd[delta2_idx][delta1_idx][Ed1_idx] = c21r * cos(de1 - de2) - c21i * sin(de1 - de2)
        dVd[delta1_idx][delta2_idx][Ed1_idx] = dVd[delta2_idx][delta1_idx][Ed1_idx]
        dVd[delta2_idx][delta2_idx][Ed1_idx] = c21i * sin(de1 - de2) - c21r * cos(de1 - de2)

        dVd[delta1_idx][Eq1_idx][delta1_idx] = dVd[delta1_idx][delta1_idx][Eq1_idx]
        dVd[delta2_idx][Eq1_idx][delta1_idx] = dVd[delta2_idx][delta1_idx][Eq1_idx]
        dVd[delta1_idx][Ed1_idx][delta1_idx] = dVd[delta1_idx][delta1_idx][Ed1_idx]
        dVd[delta2_idx][Ed1_idx][delta1_idx] = dVd[delta2_idx][delta1_idx][Ed1_idx]
        dVd[Eq1_idx][delta1_idx][delta1_idx] = dVd[delta1_idx][delta1_idx][Eq1_idx]
        dVd[Ed1_idx][delta1_idx][delta1_idx] = dVd[delta1_idx][delta1_idx][Ed1_idx]
        dVd[delta1_idx][delta1_idx][delta1_idx] = c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) + c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))
        dVd[delta2_idx][delta1_idx][delta1_idx] = - c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))
        dVd[Eq1_idx][delta2_idx][delta1_idx] = dVd[delta2_idx][delta1_idx][Eq1_idx]
        dVd[Ed1_idx][delta2_idx][delta1_idx] = dVd[delta2_idx][delta1_idx][Ed1_idx]
        dVd[delta1_idx][delta2_idx][delta1_idx] = dVd[delta2_idx][delta1_idx][delta1_idx]
        dVd[delta2_idx][delta2_idx][delta1_idx] = c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) + c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))

        dVd[delta1_idx][Eq1_idx][delta2_idx] = dVd[delta2_idx][delta1_idx][Eq1_idx]
        dVd[delta2_idx][Eq1_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][Eq1_idx]
        dVd[delta1_idx][Ed1_idx][delta2_idx] = dVd[delta2_idx][delta1_idx][Ed1_idx]
        dVd[delta2_idx][Ed1_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][Ed1_idx]
        dVd[Eq1_idx][delta1_idx][delta2_idx] = dVd[delta2_idx][delta1_idx][Eq1_idx]
        dVd[Ed1_idx][delta1_idx][delta2_idx] = dVd[delta2_idx][delta1_idx][Ed1_idx]
        dVd[delta1_idx][delta1_idx][delta2_idx] = dVd[delta2_idx][delta1_idx][delta1_idx]
        dVd[delta2_idx][delta1_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][delta1_idx]
        dVd[Eq1_idx][delta2_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][Eq1_idx]
        dVd[Ed1_idx][delta2_idx][delta2_idx] = dVd[delta2_idx][delta2_idx][Ed1_idx]
        dVd[delta1_idx][delta2_idx][delta2_idx] = dVd[delta2_idx][delta1_idx][delta2_idx]
        dVd[delta2_idx][delta2_idx][delta2_idx] = - c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) - c23r * cos(de2) - c23i * sin(de2)
        return dVd


def d_Vq2(x, order):
    Eq1_idx, Ed1_idx, delta1_idx = 0, 1, 2
    Eq2_idx, Ed2_idx, delta2_idx = 7, 8, 9
    Eq1, Ed1, de1 = x[Eq1_idx], x[Ed1_idx], x[delta1_idx]
    Eq2, Ed2, de2 = x[Eq2_idx], x[Ed2_idx], x[delta2_idx]
    if order == 1:
        dVq2 = np.zeros(x.shape[0])
        dVq2[Eq1_idx] = c21r * cos(de1 - de2) - c21i * sin(de1 - de2)
        dVq2[Ed1_idx] = c21i * cos(de1 - de2) + c21r * sin(de1 - de2)
        dVq2[delta1_idx] = c21r * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21i * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))
        dVq2[Eq2_idx] = c22r
        dVq2[Ed2_idx] = c22i
        dVq2[delta2_idx] = c21i * (Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) - c21r * (
                    Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) + c23i * cos(de2) - c23r * sin(de2)
        return dVq2
    if order == 2:
        dVq = np.zeros((x.shape[0], x.shape[0]))
        dVq[delta1_idx][Eq1_idx] = - c21i * cos(de1 - de2) - c21r * sin(de1 - de2)
        dVq[delta2_idx][Eq1_idx] = c21i * cos(de1 - de2) + c21r * sin(de1 - de2)

        dVq[delta1_idx][Ed1_idx] = c21r * cos(de1 - de2) - c21i * sin(de1 - de2)
        dVq[delta2_idx][Ed1_idx] = c21i * sin(de1 - de2) - c21r * cos(de1 - de2)

        dVq[Eq1_idx][delta1_idx] = - c21i * cos(de1 - de2) - c21r * sin(de1 - de2)
        dVq[Ed1_idx][delta1_idx] = c21r * cos(de1 - de2) - c21i * sin(de1 - de2)
        dVq[delta1_idx][delta1_idx] = - c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))
        dVq[delta2_idx][delta1_idx] = c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) + c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))

        dVq[Eq1_idx][delta2_idx] = c21i * cos(de1 - de2) + c21r * sin(de1 - de2)
        dVq[Ed1_idx][delta2_idx] = c21i * sin(de1 - de2) - c21r * cos(de1 - de2)
        dVq[delta1_idx][delta2_idx] = c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) + c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))
        dVq[delta2_idx][delta2_idx] = - c21i * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21r * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) - c23r * cos(de2) - c23i * sin(de2)
        return dVq
    if order == 3:
        dVq = np.zeros((x.shape[0], x.shape[0], x.shape[0]))
        dVq[delta1_idx][delta1_idx][Eq1_idx] = c21i * sin(de1 - de2) - c21r * cos(de1 - de2)
        dVq[delta2_idx][delta1_idx][Eq1_idx] = c21r * cos(de1 - de2) - c21i * sin(de1 - de2)
        dVq[delta1_idx][delta2_idx][Eq1_idx] = dVq[delta2_idx][delta1_idx][Eq1_idx]
        dVq[delta2_idx][delta2_idx][Eq1_idx] = c21i * sin(de1 - de2) - c21r * cos(de1 - de2)

        dVq[delta1_idx][delta1_idx][Ed1_idx] = - c21i * cos(de1 - de2) - c21r * sin(de1 - de2)
        dVq[delta2_idx][delta1_idx][Ed1_idx] = c21i * cos(de1 - de2) + c21r * sin(de1 - de2)
        dVq[delta1_idx][delta2_idx][Ed1_idx] = dVq[delta2_idx][delta1_idx][Ed1_idx]
        dVq[delta2_idx][delta2_idx][Ed1_idx] = - c21i * cos(de1 - de2) - c21r * sin(de1 - de2)

        dVq[delta1_idx][Eq1_idx][delta1_idx] = dVq[delta1_idx][delta1_idx][Eq1_idx]
        dVq[delta2_idx][Eq1_idx][delta1_idx] = dVq[delta2_idx][delta1_idx][Eq1_idx]
        dVq[delta1_idx][Ed1_idx][delta1_idx] = dVq[delta1_idx][delta1_idx][Ed1_idx]
        dVq[delta2_idx][Ed1_idx][delta1_idx] = dVq[delta2_idx][delta1_idx][Ed1_idx]
        dVq[Eq1_idx][delta1_idx][delta1_idx] = dVq[delta1_idx][delta1_idx][Eq1_idx]
        dVq[Ed1_idx][delta1_idx][delta1_idx] = dVq[delta1_idx][delta1_idx][Ed1_idx]
        dVq[delta1_idx][delta1_idx][delta1_idx] = c21i * (Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) - c21r * (
                    Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2))
        dVq[delta2_idx][delta1_idx][delta1_idx] = c21r * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21i * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2))
        dVq[Eq1_idx][delta2_idx][delta1_idx] = dVq[delta2_idx][delta1_idx][Eq1_idx]
        dVq[Ed1_idx][delta2_idx][delta1_idx] = dVq[delta2_idx][delta1_idx][Ed1_idx]
        dVq[delta1_idx][delta2_idx][delta1_idx] = dVq[delta2_idx][delta1_idx][delta1_idx]
        dVq[delta2_idx][delta2_idx][delta1_idx] = c21i * (Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) - c21r * (
                    Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2))

        dVq[delta1_idx][Eq1_idx][delta2_idx] = dVq[delta2_idx][delta1_idx][Eq1_idx]
        dVq[delta2_idx][Eq1_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][Eq1_idx]
        dVq[delta1_idx][Ed1_idx][delta2_idx] = dVq[delta2_idx][delta1_idx][Ed1_idx]
        dVq[delta2_idx][Ed1_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][Ed1_idx]
        dVq[Eq1_idx][delta1_idx][delta2_idx] = dVq[delta2_idx][delta1_idx][Eq1_idx]
        dVq[Ed1_idx][delta1_idx][delta2_idx] = dVq[delta2_idx][delta1_idx][Ed1_idx]
        dVq[delta1_idx][delta1_idx][delta2_idx] = dVq[delta2_idx][delta1_idx][delta1_idx]
        dVq[delta2_idx][delta1_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][delta1_idx]
        dVq[Eq1_idx][delta2_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][Eq1_idx]
        dVq[Ed1_idx][delta2_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][Ed1_idx]
        dVq[delta1_idx][delta2_idx][delta2_idx] = dVq[delta2_idx][delta2_idx][delta1_idx]
        dVq[delta2_idx][delta2_idx][delta2_idx] = c21r * (Ed1 * cos(de1 - de2) - Eq1 * sin(de1 - de2)) - c21i * (
                    Eq1 * cos(de1 - de2) + Ed1 * sin(de1 - de2)) - c23i * cos(de2) + c23r * sin(de2)
        return dVq


def d_Id1(x, order):
    if order == 1:
        dId1 = -d_Vq1(x, order) / Xdp
        dId1[0] += 1.0 / Xdp
        return dId1
    if order == 2:
        return -d_Vq1(x, order) / Xdp
    if order == 3:
        return -d_Vq1(x, order) / Xdp


def d_Id2(x, order):
    if order == 1:
        dId2 = -d_Vq2(x, order) / Xdp
        dId2[7] += 1.0 / Xdp
        return dId2
    if order == 2:
        return -d_Vq2(x, order) / Xdp
    if order == 3:
        return -d_Vq2(x, order) / Xdp


def d_Iq1(x, order):
    if order == 1:
        dIq1 = d_Vd1(x, order) / Xqp
        dIq1[1] -= 1.0 / Xqp
        return dIq1
    if order == 2:
        return d_Vd1(x, order) / Xqp
    if order == 3:
        return d_Vd1(x, order) / Xqp


def d_Iq2(x, order):
    if order == 1:
        dIq2 = d_Vd2(x, order) / Xqp
        dIq2[8] -= 1.0 / Xqp
        return dIq2
    if order == 2:
        return d_Vd2(x, order) / Xqp
    if order == 3:
        return d_Vd2(x, order) / Xqp


def d_Ifd1(x, order):
    if order == 1:
        dIfd = (Xd - Xdp) * d_Id1(x, order)
        dIfd[0] += 1
        return dIfd
    if order == 2:
        return (Xd - Xdp) * d_Id1(x, order)
    if order == 3:
        return (Xd - Xdp) * d_Id1(x, order)


def d_Ifd2(x, order):
    if order == 1:
        dIfd = (Xd - Xdp) * d_Id2(x, order)
        dIfd[7] += 1
        return dIfd
    if order == 2:
        return (Xd - Xdp) * d_Id2(x, order)
    if order == 3:
        return (Xd - Xdp) * d_Id2(x, order)


def d_Efd1(x, order):
    if order == 1:
        dEfd = -0.577 * KC1 * d_Ifd1(x, order)
        Ve_idx = 6
        dEfd[Ve_idx] += 1
        return dEfd
    if order == 2:
        return -0.577 * KC1 * d_Ifd1(x, order)
    if order == 3:
        return -0.577 * KC1 * d_Ifd1(x, order)


def d_Efd2(x, order):
    if order == 1:
        dEfd = -0.577 * KC2 * d_Ifd2(x, order)
        Ve_idx = 13
        dEfd[Ve_idx] += 1
        return dEfd
    if order == 2:
        return -0.577 * KC2 * d_Ifd2(x, order)
    if order == 3:
        return -0.577 * KC2 * d_Ifd2(x, order)


def d_Eq1(x, order):
    if order == 1:
        dEq = -(Xd - Xdp) / Tdp * d_Id1(x, order)
        dEq += d_Efd1(x, order) / Tdp
        dEq[0] -= 1.0 / Tdp
        return dEq
    if order == 2:
        return -(Xd - Xdp) / Tdp * d_Id1(x, order) + d_Efd1(x, order) / Tdp
    if order == 3:
        return -(Xd - Xdp) / Tdp * d_Id1(x, order) + d_Efd1(x, order) / Tdp


def d_Eq2(x, order):
    if order == 1:
        dEq = -(Xd - Xdp) / Tdp * d_Id2(x, order)
        dEq += d_Efd2(x, order) / Tdp
        dEq[7] -= 1.0 / Tdp
        return dEq
    if order == 2:
        return -(Xd - Xdp) / Tdp * d_Id2(x, order) + d_Efd2(x, order) / Tdp
    if order == 3:
        return -(Xd - Xdp) / Tdp * d_Id2(x, order) + d_Efd2(x, order) / Tdp


def d_Ed1(x, order):
    if order == 1:
        dEd = (Xq - Xqp) / Tqp * d_Iq1(x, order)
        dEd[1] -= 1.0 / Tqp
        return dEd
    if order == 2:
        return (Xq - Xqp) / Tqp * d_Iq1(x, order)
    if order == 3:
        return (Xq - Xqp) / Tqp * d_Iq1(x, order)


def d_Ed2(x, order):
    if order == 1:
        dEd = (Xq - Xqp) / Tqp * d_Iq2(x, order)
        dEd[8] -= 1.0 / Tqp
        return dEd
    if order == 2:
        return (Xq - Xqp) / Tqp * d_Iq2(x, order)
    if order == 3:
        return (Xq - Xqp) / Tqp * d_Iq2(x, order)


def d_delta1(x, order):
    if order == 1:
        dde = np.zeros(x.shape[0])
        dde[3] = 120 * np.pi
        return dde
    if order == 2:
        return np.zeros((x.shape[0], x.shape[0]))
    if order == 3:
        return np.zeros((x.shape[0], x.shape[0], x.shape[0]))


def d_delta2(x, order):
    if order == 1:
        dde = np.zeros(x.shape[0])
        dde[10] = 120 * np.pi
        return dde
    if order == 2:
        return np.zeros((x.shape[0], x.shape[0]))
    if order == 3:
        return np.zeros((x.shape[0], x.shape[0], x.shape[0]))


# Pe = Eq*Iq + Ed*Id = Eq*(Vd-Ed)/Xqp + Ed*(Eq-Vq)/Xdp = (Eq*Vd - Ed*Vq)/Xdp
def d_w1(x, order):
    Eq, Ed = x[0], x[1]
    if order == 1:
        Vd1 = np.real(f_Vdq(x)[0])
        Vq1 = np.imag(f_Vdq(x)[0])
        dw = Eq * d_Vd1(x, order) - Ed * d_Vq1(x, order)
        dw[0] += Vd1
        dw[1] += -Vq1
        return -dw / Xdp / (2 * H)
    if order == 2:
        dw = Eq * d_Vd1(x, order) - Ed * d_Vq1(x, order)
        dw[0, :] += d_Vd1(x, order - 1)
        dw[:, 0] += d_Vd1(x, order - 1)
        dw[1, :] += -d_Vq1(x, order - 1)
        dw[:, 1] += -d_Vq1(x, order - 1)
        return -dw / Xdp / (2 * H)
    if order == 3:
        dw = Eq * d_Vd1(x, order) - Ed * d_Vq1(x, order)
        dw[0, :, :] += d_Vd1(x, order - 1)
        dw[:, 0, :] += d_Vd1(x, order - 1)
        dw[:, :, 0] += d_Vd1(x, order - 1)
        dw[1, :, :] += -d_Vq1(x, order - 1)
        dw[:, 1, :] += -d_Vq1(x, order - 1)
        dw[:, :, 1] += -d_Vq1(x, order - 1)
        return -dw / Xdp / (2 * H)


def d_w2(x, order):
    Eq, Ed = x[7], x[8]
    if order == 1:
        Vd2 = np.real(f_Vdq(x)[1])
        Vq2 = np.imag(f_Vdq(x)[1])
        dw = Eq * d_Vd2(x, order) - Ed * d_Vq2(x, order)
        dw[7] += Vd2
        dw[8] += -Vq2
        return -dw / Xdp / (2 * H)
    if order == 2:
        dw = Eq * d_Vd2(x, order) - Ed * d_Vq2(x, order)
        dw[7, :] += d_Vd2(x, order - 1)
        dw[:, 7] += d_Vd2(x, order - 1)
        dw[8, :] += -d_Vq2(x, order - 1)
        dw[:, 8] += -d_Vq2(x, order - 1)
        return -dw / Xdp / (2 * H)
    if order == 3:
        dw = Eq * d_Vd2(x, order) - Ed * d_Vq2(x, order)
        dw[7, :, :] += d_Vd2(x, order - 1)
        dw[:, 7, :] += d_Vd2(x, order - 1)
        dw[:, :, 7] += d_Vd2(x, order - 1)
        dw[8, :, :] += -d_Vq2(x, order - 1)
        dw[:, 8, :] += -d_Vq2(x, order - 1)
        dw[:, :, 8] += -d_Vq2(x, order - 1)
        return -dw / Xdp / (2 * H)


def d_Vfe1(x, order):
    Ve1 = x[6]
    if order == 1:
        dVfe = KD1 * d_Ifd1(x, order)
        dVfe[6] += KE1 + Aex1 * Bex1 * np.exp(Bex1 * Ve1)
        return dVfe
    if order == 2:
        dVfe = KD1 * d_Ifd1(x, order)
        dVfe[6, 6] += Aex1 * Bex1 * Bex1 * np.exp(Bex1 * Ve1)
        return dVfe
    if order == 3:
        dVfe = KD1 * d_Ifd1(x, order)
        dVfe[6, 6, 6] += Aex1 * Bex1 * Bex1 * Bex1 * np.exp(Bex1 * Ve1)
        return dVfe


def d_Vfe2(x, order):
    Ve2 = x[13]
    if order == 1:
        dVfe = KD2 * d_Ifd2(x, order)
        dVfe[13] += KE2 + Aex2 * Bex2 * np.exp(Bex2 * Ve2)
        return dVfe
    if order == 2:
        dVfe = KD2 * d_Ifd2(x, order)
        dVfe[13, 13] += Aex2 * Bex2 * Bex2 * np.exp(Bex2 * Ve2)
        return dVfe
    if order == 3:
        dVfe = KD2 * d_Ifd2(x, order)
        dVfe[13, 13, 13] += Aex2 * Bex2 * Bex2 * Bex2 * np.exp(Bex2 * Ve2)
        return dVfe


def d_Vf1(x, order):
    if order == 1:
        dVf = d_Vfe1(x, order) / TF1
        dVf[4] -= 1 / TF1
        return dVf
    if order == 2:
        return d_Vfe1(x, order) / TF1
    if order == 3:
        return d_Vfe1(x, order) / TF1


def d_Vf2(x, order):
    if order == 1:
        dVf = d_Vfe2(x, order) / TF2
        dVf[11] -= 1 / TF2
        return dVf
    if order == 2:
        return d_Vfe2(x, order) / TF2
    if order == 3:
        return d_Vfe2(x, order) / TF2


# Vm = sqrt(Vd^2 + Vq^2)
def d_Vm1(x, order):
    Vd = f_Vdq(x)[0].real
    Vq = f_Vdq(x)[0].imag
    Vm = abs(f_Vdq(x)[0])
    if order == 1:
        return Vd / Vm * d_Vd1(x, 1) + Vq / Vm * d_Vq1(x, 1)
    if order == 2:
        dVm = Vd / Vm * d_Vd1(x, 2) + Vq / Vm * d_Vq1(x, 2)
        dVm += np.outer(Vq * d_Vd1(x, 1) - Vd * d_Vq1(x, 1), Vq * d_Vd1(x, 1) - Vd * d_Vq1(x, 1)) / Vm ** 3
        return dVm
    if order == 3:
#         dVm = Vd / Vm * d_Vd1(x, 3) + Vq / Vm * d_Vq1(x, 3)
#         dVdVm = d_Vd1(x, 1) / Vm - Vd / Vm ** 2 * d_Vm1(x, 1)  # (Vq**2 * d_Vd1(x,1) - Vd*Vq * d_Vq1(x,1))/Vm**3
#         dVqVm = d_Vq1(x, 1) / Vm - Vq / Vm ** 2 * d_Vm1(x, 1)  # (Vd**2 * d_Vq1(x,1) - Vd*Vq * d_Vd1(x,1))/Vm**3
#         dVm += np.einsum('ij,k->kij', d_Vd1(x, 2), dVdVm) + np.einsum('ij,k->kij', d_Vq1(x, 2), dVqVm)
#         V = Vq * d_Vd1(x, 1) - Vd * d_Vq1(x, 1)
#         VV = np.outer(V, V)
#         dVm += np.einsum('ij,k->kij', VV, -3.0 / Vm ** 4 * d_Vm1(x, 1))
#         dV = Vq * d_Vd1(x, 2) + np.outer(d_Vq1(x,1), d_Vd1(x,1)) - Vd * d_Vq1(x, 2) - np.outer(d_Vd1(x,1), d_Vq1(x,1)) 
#         dVm += (np.einsum('ij,k->jki', dV, V) + np.einsum('ij,k->jik', dV, V)) / Vm ** 3
#         return dVm
        dVm = Vd / Vm * d_Vd1(x, 3) + Vq / Vm * d_Vq1(x, 3)
        dVm += np.einsum('ij,k->kij', d_Vd1(x, 2), (Vm*d_Vd1(x,1)-Vd*d_Vm1(x,1))/Vm**2) + \
               np.einsum('ij,k->kij', d_Vq1(x, 2), (Vm*d_Vq1(x,1)-Vq*d_Vm1(x,1))/Vm**2)
        dVm += np.einsum('ij,k->kij', np.outer(d_Vd1(x,1), d_Vd1(x,1)), -1/Vm**2*d_Vm1(x,1))
        dVm += (np.einsum('ij,k->jki', d_Vd1(x,2), d_Vd1(x,1)) + np.einsum('ij,k->jik',  d_Vd1(x,2), d_Vd1(x,1))) / Vm
        dVm += np.einsum('ij,k->kij', np.outer(d_Vq1(x,1), d_Vq1(x,1)), -1/Vm**2*d_Vm1(x,1))
        dVm += (np.einsum('ij,k->jki', d_Vq1(x,2), d_Vq1(x,1)) + np.einsum('ij,k->jik',  d_Vq1(x,2), d_Vq1(x,1))) / Vm
        dVm -= np.einsum('ij,k->kij', np.outer(d_Vm1(x,1), d_Vd1(x,1)), (Vm**2*d_Vd1(x,1) - 2*Vd*Vm*d_Vm1(x,1))/Vm**4)
        dVm -= (np.einsum('ij,k->jki', d_Vd1(x,2), d_Vm1(x,1)) + np.einsum('ij,k->jik',  d_Vm1(x,2), d_Vd1(x,1))) * Vd/Vm**2
        dVm -= np.einsum('ij,k->kij', np.outer(d_Vm1(x,1), d_Vq1(x,1)), (Vm**2*d_Vq1(x,1) - 2*Vq*Vm*d_Vm1(x,1))/Vm**4)
        dVm -= (np.einsum('ij,k->jki', d_Vq1(x,2), d_Vm1(x,1)) + np.einsum('ij,k->jik',  d_Vm1(x,2), d_Vq1(x,1))) * Vq/Vm**2
        return dVm
        

def d_Vm2(x, order):
    Vd = f_Vdq(x)[1].real
    Vq = f_Vdq(x)[1].imag
    Vm = abs(f_Vdq(x)[1])
    if order == 1:
        return Vd / Vm * d_Vd2(x, 1) + Vq / Vm * d_Vq2(x, 1)
    if order == 2:
        dVm = Vd / Vm * d_Vd2(x, 2) + Vq / Vm * d_Vq2(x, 2)
        dVm += np.outer(Vq * d_Vd2(x, 1) - Vd * d_Vq2(x, 1), Vq * d_Vd2(x, 1) - Vd * d_Vq2(x, 1)) / Vm ** 3
        return dVm
    if order == 3:
#         dVm = Vd / Vm * d_Vd2(x, 3) + Vq / Vm * d_Vq2(x, 3)
#         dVdVm = (Vq ** 2 * d_Vd2(x, 1) - Vd * Vq * d_Vq2(x, 1)) / Vm ** 3
#         dVqVm = (Vd ** 2 * d_Vq2(x, 1) - Vd * Vq * d_Vd2(x, 1)) / Vm ** 3
#         dVm += np.einsum('ij,k->kij', d_Vd2(x, 2), dVdVm) + np.einsum('ij,k->kij', d_Vq2(x, 2), dVqVm)
#         V = Vq * d_Vd2(x, 1) - Vd * d_Vq2(x, 1)
#         VV = np.outer(V, V) / Vm ** 3
#         dVm += np.einsum('ij,k->kij', VV, -3 / Vm ** 4 * (Vd / Vm * d_Vd2(x, 1) + Vq / Vm * d_Vq2(x, 1)))
#         dV = Vq * d_Vd2(x, 2) - Vd * d_Vq2(x, 2)
#         dVm += (np.einsum('ij,k->ikj', dV, V) + np.einsum('ij,k->jik', dV, V)) / Vm ** 3
        dVm = Vd / Vm * d_Vd2(x, 3) + Vq / Vm * d_Vq2(x, 3)
        dVm += np.einsum('ij,k->kij', d_Vd2(x, 2), (Vm*d_Vd2(x,1)-Vd*d_Vm2(x,1))/Vm**2) + \
               np.einsum('ij,k->kij', d_Vq2(x, 2), (Vm*d_Vq2(x,1)-Vq*d_Vm2(x,1))/Vm**2)
        dVm += np.einsum('ij,k->kij', np.outer(d_Vd2(x,1), d_Vd2(x,1)), -1/Vm**2*d_Vm2(x,1))
        dVm += (np.einsum('ij,k->jki', d_Vd2(x,2), d_Vd2(x,1)) + np.einsum('ij,k->jik',  d_Vd2(x,2), d_Vd2(x,1))) / Vm
        dVm += np.einsum('ij,k->kij', np.outer(d_Vq2(x,1), d_Vq2(x,1)), -1/Vm**2*d_Vm2(x,1))
        dVm += (np.einsum('ij,k->jki', d_Vq2(x,2), d_Vq2(x,1)) + np.einsum('ij,k->jik',  d_Vq2(x,2), d_Vq2(x,1))) / Vm
        dVm -= np.einsum('ij,k->kij', np.outer(d_Vm2(x,1), d_Vd2(x,1)), (Vm**2*d_Vd2(x,1) - 2*Vd*Vm*d_Vm2(x,1))/Vm**4)
        dVm -= (np.einsum('ij,k->jki', d_Vd2(x,2), d_Vm2(x,1)) + np.einsum('ij,k->jik',  d_Vm2(x,2), d_Vd2(x,1))) * Vd/Vm**2
        dVm -= np.einsum('ij,k->kij', np.outer(d_Vm2(x,1), d_Vq2(x,1)), (Vm**2*d_Vq2(x,1) - 2*Vq*Vm*d_Vm2(x,1))/Vm**4)
        dVm -= (np.einsum('ij,k->jki', d_Vq2(x,2), d_Vm2(x,1)) + np.einsum('ij,k->jik',  d_Vm2(x,2), d_Vq2(x,1))) * Vq/Vm**2
        return dVm


def d_Vsum1(x, order):
    Vd1 = f_Vdq(x)[0].real
    Vq1 = f_Vdq(x)[0].imag
    Vm = abs(f_Vdq(x)[0])
    if order == 1:
        dVsum = -KF1 * d_Vf1(x, order)
        dVsum += -d_Vm1(x, order)
        return dVsum
    if order == 2:
        dVsum = -KF1 * d_Vf1(x, order)
        dVsum += -d_Vm1(x, order)
        return dVsum
    if order == 3:
        dVsum = -KF1 * d_Vf1(x, order)
        dVsum += -d_Vm1(x, order)
        return dVsum


def d_Vsum2(x, order):
    Vd2 = f_Vdq(x)[1].real
    Vq2 = f_Vdq(x)[1].imag
    Vm = abs(f_Vdq(x)[1])
    if order == 1:
        dVsum = -KF2 * d_Vf2(x, order)
        dVsum += -d_Vm2(x, order)
        return dVsum
    if order == 2:
        dVsum = -KF2 * d_Vf2(x, order)
        dVsum += -d_Vm2(x, order)
        return dVsum
    if order == 3:
        dVsum = -KF2 * d_Vf2(x, order)
        dVsum += -d_Vm2(x, order)
        return dVsum


def d_Va1(x, order):
    if order == 1:
        dVa = KA1*d_Vsum1(x, order) / TA1
        dVa[5] -= 1.0 / TA1
        return dVa
    if order == 2:
        return KA1*d_Vsum1(x, order) / TA1
    if order == 3:
        return KA1*d_Vsum1(x, order) / TA1


def d_Va2(x, order):
    if order == 1:
        dVa = KA2*d_Vsum2(x, order) / TA2
        dVa[12] -= 1.0 / TA2
        return dVa
    if order == 2:
        return KA2*d_Vsum2(x, order) / TA2
    if order == 3:
        return KA2*d_Vsum2(x, order) / TA2


def d_Ve1(x, order):
    if order == 1:
        dVe = -d_Vfe1(x, order) / TE1
        dVe[5] += 1/ TE1
        return dVe
    if order == 2:
        return -d_Vfe1(x, order) / TE1
    if order == 3:
        return -d_Vfe1(x, order) / TE1


def d_Ve2(x, order):
    if order == 1:
        dVe = -d_Vfe2(x, order) / TE2
        dVe[12] += 1/ TE2
        return dVe
    if order == 2:
        return -d_Vfe2(x, order) / TE2
    if order == 3:
        return -d_Vfe2(x, order) / TE2


def Jacobian(x):
    return np.array([f(x, 1) for f in
                     [d_Eq1, d_Ed1, d_delta1, d_w1, d_Vf1, d_Va1, d_Ve1, d_Eq2, d_Ed2, d_delta2, d_w2, d_Vf2, d_Va2,
                      d_Ve2]])


def Hessian(x):
    return np.array([f(x, 2) for f in
                     [d_Eq1, d_Ed1, d_delta1, d_w1, d_Vf1, d_Va1, d_Ve1, d_Eq2, d_Ed2, d_delta2, d_w2, d_Vf2, d_Va2,
                      d_Ve2]])


def Trissian(x):
    return np.array([f(x, 3) for f in
                     [d_Eq1, d_Ed1, d_delta1, d_w1, d_Vf1, d_Va1, d_Ve1, d_Eq2, d_Ed2, d_delta2, d_w2, d_Vf2, d_Va2,
                      d_Ve2]])






def T2_mat(n):
    T2 = np.eye(n**2,dtype=int)
    rmidx = np.triu_indices(n,1)[1]*n + np.triu_indices(n,1)[0]
    T2 = np.delete(T2,rmidx,0)
    return T2


def S2_mat(n):
    S2 = np.eye(n**2,dtype=int)
    rmidx = np.triu_indices(n,1)[1]*n + np.triu_indices(n,1)[0]
    addidx = np.triu_indices(n,1)[0]*n + np.triu_indices(n,1)[1]
    S2[rmidx,addidx] = 1
    S2 = np.delete(S2,rmidx,1)
    return S2

def T3_mat(n):
    Bx3 = [(i,j,k) for i in range(n) for j in range(i,n) for k in range(j,n)] # extracted from x \otimes Bx^2
    x_Bx2 = [(i,j,k) for i in range(n) for j in range(n) for k in range(j,n)] #  x \otimes Bx^2
    Bx3_idx = [x_Bx2.index(i) for i in Bx3]
    rmidx = list(set(range(len(x_Bx2)))-set(Bx3_idx))
    rmele = [x_Bx2[i] for i in rmidx]
    rmele = [tuple(sorted(i)) for i in rmele]
    rmidx_inBx3 = [Bx3.index(i) for i in rmele]
    T3 = np.eye(n*n*(n+1)//2,dtype=int)
    T3 = T3[Bx3_idx]
    return T3

def S3_mat(n):
    Bx3 = [(i,j,k) for i in range(n) for j in range(i,n) for k in range(j,n)] # extracted from x \otimes Bx^2
    x_Bx2 = [(i,j,k) for i in range(n) for j in range(n) for k in range(j,n)] #  x \otimes Bx^2
    Bx3_idx = [x_Bx2.index(i) for i in Bx3]
    rmidx = list(set(range(len(x_Bx2)))-set(Bx3_idx))
    rmele = [x_Bx2[i] for i in rmidx]
    rmele = [tuple(sorted(i)) for i in rmele]
    rmidx_inBx3 = [Bx3.index(i) for i in rmele]
    S3 = np.eye(n*n*(n+1)//2,dtype=int)
    S3 = S3[:,Bx3_idx]
    S3[rmidx,rmidx_inBx3] = 1
    return S3


