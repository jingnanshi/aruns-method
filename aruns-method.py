import numpy as np
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement

def load_ply(filename):
    plydata = PlyData.read('bun_zipper_res4.ply')
    N = plydata.elements[0].data.shape[0]
    pc = np.zeros((3, N))
    pc[0, :] = plydata.elements[0].data['x']
    pc[1, :] = plydata.elements[0].data['y']
    pc[2, :] = plydata.elements[0].data['z']
    return pc


def arun(A, B):
    """Solve 3D registration using Arun's method: B = RA + t
    """
    N = A.shape[1]
    assert B.shape[1] == N

    # calculate centroids
    A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
    B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))

    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid

    # rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + np.outer(ai, bi)
    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

    # translation estimation
    t = B_centroid - R @ A_centroid

    return R, t


if __name__ == "__main__":
    print("Example of using Arun's method")
    # load point cloud
    pc = load_ply("bun_zipper_res4.ply")

    # apply random transformation
    R_actual = R.random(random_state=1234).as_matrix()
    t_actual = np.random.rand(3,1) * 10
    pc_transformed = R_actual @ pc + t_actual 

    R_est, t_est = arun(pc, pc_transformed)
    print("===============================")
    print("Actual rotation:")
    print(R_actual)
    print("Actual translation:")
    print(t_actual)
    print("===============================")
    print("Est. rotation:")
    print(R_est)
    print("Est. translation:")
    print(t_est)
