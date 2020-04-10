import numpy as np
import numdifftools as nd

ws = 120*np.pi;

# x = np.array([0.938022851060938,   0.557324461444381,   0.816580802164681,  0,  1.824592816100050,\
#    1.032798078513208, 0.522133778571191,  0.572163472999078,     0,  2.042741493091689,\
#     0.984840910890953,  0.563020335043396,   0.499008849559036,    0,  1.910751086279320,\
#     0.994656925790442,  0.538950764861535,   0.293938679091155,    0,  1.943874501883284])
  
Y_int = np.array([[0.636958839228921 - 0.054712741522639*1j,  0.213248635243739 - 0.062193237360221*1j,  0.027586125919393 - 0.029170898994735*1j,  0.037641537941523 - 0.042214998411686*1j],[0.213248635243739 - 0.062193237360221*1j,  0.570591416523583 - 0.096312143491014*1j,  0.037663947557888 - 0.042214778071118*1j,  0.051294328543277 - 0.060998452075847*1j],[0.027586125919393 - 0.029170898994735*1j,  0.037663947557888 - 0.042214778071118*1j,  0.625787690168615 - 0.072364326591730*1j,  0.189708749040344 - 0.086419262443990*1j],[0.037641537941523 - 0.042214998411686*1j,  0.051294328543277 - 0.060998452075847*1j,  0.189708749040344 - 0.086419262443990*1j,  0.536337934532991 - 0.129518209121736*1j]])

Vref = np.array([1.202732535122228,   1.236971277010188,   1.242305676253258,   1.225986055764810])

Pm = np.array([6.795670446562388, 7.1232970782803751, 7.189999999998113, 6.999999999988546])

ratio = 9;
Tdp = np.array([8, 8, 8, 8])
Tqp = np.array([0.4, 0.4, 0.4, 0.4])
H = np.array([6.5, 6.5, 6.175, 6.175]) * ratio
KD = np.array([1, 1, 1, 1]) * ratio
Xd = np.array([1.8, 1.8, 1.8, 1.8]) / ratio
Xq = np.array([1.7, 1.7, 1.7, 1.7]) / ratio
Xdp = np.array([0.4, 0.4, 0.4, 0.4]) / ratio
Xqp = np.array([0.4, 0.4, 0.4, 0.4]) / ratio
Xl = np.array([0.2, 0.2, 0.2, 0.2]) / ratio
Rs = np.array([0, 0, 0, 0]) / ratio


Ka = np.array([9, 9, 9, 9])
Ta = np.array([0.02, 0.02, 0.02, 0.02])

Eqp_idx = [0, 5, 10, 15]
Edp_idx = [1, 6, 11, 16]
delta_idx = [2, 7, 12, 17]
w_idx = [3, 8, 13, 18]
Efd_idx = [4, 9, 14, 19]


def kundur_hopf_sup(x):
    F = np.zeros(x.shape)
    
    Eqp = x[Eqp_idx]
    Edp = x[Edp_idx]
    delta = x[delta_idx]
    w = x[w_idx]
    Efd = x[Efd_idx]
    
    Edq = (Edp + 1j * Eqp) * np.exp(1j * (delta - np.pi/2));
    Vdq = np.matmul(np.matmul(np.diag(np.exp(1j*(np.pi/2 - delta))),Y_int),Edq);
    Vd = Vdq.real;
    Vq = Vdq.imag;
    Id = (Rs*(Edp-Vd)+Xqp*(Eqp-Vq))/(Rs*Rs+Xdp*Xqp); 
    Iq = (-Xdp*(Edp-Vd)+Rs*(Eqp-Vq))/(Rs*Rs+Xdp*Xqp);
    Pe = (Eqp*Iq - Xdp*Id*Iq + Edp*Id + Xqp*Id*Iq);
    
    F[Eqp_idx] = (-Eqp - (Xd - Xdp)*Id + Efd) / Tdp;
    F[Edp_idx] = (-Edp + (Xq - Xqp)*Iq) /Tqp;
    F[delta_idx] = w * ws;
    F[w_idx] = (Pm - Pe - KD*w) / (2*H);
    F[Efd_idx] = (Ka*(Vref - abs(Vdq)) - Efd)/Ta;
    return F

def kundur_hopf_sup_xPm(x):
    F = np.zeros(20)
    
    Eqp = x[Eqp_idx]
    Edp = x[Edp_idx]
    delta = x[delta_idx]
    w = x[w_idx]
    Efd = x[Efd_idx]
    
    Edq = (Edp + 1j * Eqp) * np.exp(1j * (delta - np.pi/2));
    Vdq = np.matmul(np.matmul(np.diag(np.exp(1j*(np.pi/2 - delta))),Y_int),Edq);
    Vd = Vdq.real;
    Vq = Vdq.imag;
    Id = (Rs*(Edp-Vd)+Xqp*(Eqp-Vq))/(Rs*Rs+Xdp*Xqp); 
    Iq = (-Xdp*(Edp-Vd)+Rs*(Eqp-Vq))/(Rs*Rs+Xdp*Xqp);
    Pe = (Eqp*Iq - Xdp*Id*Iq + Edp*Id + Xqp*Id*Iq);
    
    F[Eqp_idx] = (-Eqp - (Xd - Xdp)*Id + Efd) / Tdp;
    F[Edp_idx] = (-Edp + (Xq - Xqp)*Iq) /Tqp;
    F[delta_idx] = w * ws;
    #F[w_idx] = (Pm - Pe - KD*w) / (2*H);
    F[Efd_idx] = (Ka*(Vref - abs(Vdq)) - Efd)/Ta;
    F[w_idx[0]] = (Pm[0] - Pe[0] - KD[0]*w[0]) / (2*H[0]);
    F[w_idx[1]] = (x[20] - Pe[1] - KD[1]*w[1]) / (2*H[1]);
    F[w_idx[2]] = (Pm[2]  - Pe[2] - KD[2]*w[2]) / (2*H[2]);
    F[w_idx[3]] = (Pm[3]  - Pe[3] - KD[3]*w[3]) / (2*H[3]);
    
    return F

def kundur_hopf_sup_zeroeig(x):
    J = nd.Jacobian(kundur_hopf_sup_xPm)(x)
    J = J[:,:-1]
    lam, v = np.linalg.eig(J)
    print(lam)
    return np.append(kundur_hopf_sup_xPm(x), [lam[10].real])