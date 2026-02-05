'''
This file will contain the quaternion fucntions used for the rest of the dynamics, 
and how quaternions are used together.

First, a quaternion is a attitude representation, this project will use the Hamiltonian
representation of quaternions, and will follow the math behind this textbook:

Fundamentals of Spacecraft Attitude Determination by F. Landis Markley and John L. Crassidis
'''

import numpy as np

class Quaternion:
    # The quaternion will be defines as q = [q1:3, q4] where q1:3 is the vector component, and q4 is the scalar
    # An error is raised if the quaternion is not normalized before using a fucntion that requires a unit quaternion.
    def __init__(self, q): # Initialize a quaternion, and guarantee it is normalized so there is no numerical instability.
        self.q = np.array(q, dtype=float)

        if self.q.shape != (4,):
            raise ValueError("Quaternion is not a 4x1 vector")
            
    
    def normalize(self): # Normalize the quaternion based on a certain threshold for its magnitude.
        n = np.linalg.norm(self.q)
        if n < 1e-12:
            self.q = np.array([0.0, 0.0, 0.0, 1.0]) # Identity Quaternion
        else:
            self.q /= n

    def multiplication(self, q2: 'Quaternion') -> 'Quaternion':
        x1, y1, z1, w1 = self.q
        x2, y2, z2, w2 = q2.q
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*x2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def conjugate(self): # Conjugate of Quaternion
        x, y, z, w = self.q
        return Quaternion([-x, -y, -z, w])
    
    def inverse(self): # Inverse of Quaternion
        norm_sq = np.dot(self.q, self.q)
        return Quaternion(self.conjugate().q / norm_sq)
    
    def quat_to_DCM(self): # Returns DCM from body to inertial (Hamiltonian Convention)
        if abs(np.linalg.norm(self.q) - 1.0) > 1e-6:
            self.normalize()
        x, y, z, w = self.q

        DCM = np.array([
            [w**2 + x**2 - y**2 - z**2, 2 * (x*y + w*z), 2 * (x*z - w*y)],
            [2 * (x*y - w*z), w**2 - x**2 + y**2 - z**2, 2 * (y*z - w*x)],
            [2 * (x*z + w*y), 2 * (y*z - w*x), (w**2 - x**2 - y**2 + z**2)]
        ])
        return DCM
    def quaternion_error(self, target): #  target = error * current --> error = target * current^-1 using the hamiltonian product
        error = target @ (self.inverse())
        return error
    # Converting between body and inertial frames using the DCM matrix
    def inertial_to_body(self, vec_i):
        C_ib = self.quat_to_DCM()
        return C_ib @ vec_i
    def body_to_inertial(self, vec_b):
        C_ib = self.quat_to_DCM()
        C_bi = C_ib.T
        return C_bi @ vec_b