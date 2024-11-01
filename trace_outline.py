"""
find true outline of a projected mesh includes intersections
"""
import numpy as np
from nvtools.geometry.array_operation import norm_vectors

tol = 1e-15


def get_unique_edges(faces):
    """Return unique edges

    Args:
        faces: (m,3)
            array of faces

    Returns:
        unique_edges: (n,2)
            unique edges 
    """
    edges = np.vstack((faces[:, [0, 1]], faces[:, [0, 2]],
                      faces[:, [1, 2]]))  # stack all edges
    edges = np.sort(edges, axis=-1)
    # sort left to right, up to down
    edges = edges[np.lexsort(edges[:, ::-1].T)]
    unique_edges = np.unique(edges, axis=0)  # unique
    return unique_edges


def get_neighbor(id, f):
    """Return neighbors indices

    Args:
        id: list of int
        f: (m,3)
            array of faces
    Returns:
        neighbor_list: list of list
            each list contians neighbor indices of given id
    """
    neighbor_list = []
    for i in id:
        neighbor = set(f[np.where(f == i)[0]].flatten())
        if i in neighbor:
            neighbor.remove(i)
        neighbor = list(neighbor)
        neighbor_list.extend(neighbor)
    return neighbor_list


def get_adj(v, f, level):
    """Find neighbors indices for all verts

    Args:
        v: (n,3) float
            array of verts
        f: (m,3) int
            array of faces
        level: int
            level > 1 indicates find neighbors for found verts

    Returns:
        adj: list of list
            contains find neighbors 
    """
    adj = []
    for i, vt in enumerate(v):
        neighbor_list = [i]
        for j in range(level):
            neighbor_list.extend(get_neighbor(neighbor_list, f))
        neighbor_list.remove(i)
        adj.append(neighbor_list)
    return adj


def get_edges_adj(v, f):
    """Return edges and neighbor for each vert

    Args:
        v: (n,3) float
            array of verts
        f: (m,3) int
            array of faces
    Returns:
        edges: list of array
            edges of each vert
        adj: list of list
            neighbors of each vert

    """
    edges = []
    adj = []
    for i, vt in enumerate(v):
        neighbor = get_neighbor([i], f)
        adj.append(neighbor)
        edges.append(
            np.sort(np.vstack((np.repeat(i, len(neighbor)), neighbor)).T, axis=1))
    return edges, adj


def find_clockwise_nearest(vector_a, vector_b_arr):
    """Return first cloest vector for vector_a in vector_b_arr along clockwise
        via finding the smallest angle between vector_a and vector_b_arr along clockwise

    Args: 
        vector_a: (2,1) float
            arry of vector
        vector_b_arr;(n,2) float
            array of vectors

    Returns:
        clockwise_nearest: int
            index of cloest vector in vector_b_arr

    """
    ang = np.arctan2(vector_a[0]*vector_b_arr[:, 1]-vector_a[1]*vector_b_arr[:, 0],
                     vector_a[0]*vector_b_arr[:, 0]+vector_a[1]*vector_b_arr[:, 1])
    positive_id = np.where(ang > 0)[0]  # get positive angle
    if positive_id.shape[0] > 0:
        # e.g angle [-20,20,30] we wanna get 20 degree, rather than -20 degree,
        # because -20 degree means the vector has neg direction compare to vector_a
        clockwise_nearest = positive_id[np.argmin(ang[positive_id])]
    else:
        negative_id = np.where(ang < 0)[0]
        clockwise_nearest = negative_id[np.argmin(ang[negative_id])]
    return clockwise_nearest


def find_inters(pv, rv, qv, sv):
    """Return intersections among given vectors pv(start point) + rv(direction) and qv(start point) + sv(direction)
        refer: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    Args:
        pv: (2,) float
            2D point
        rv: (2,) float
            2D vector 
        qv: (2,) float
            2D point
        sv: (2,) float
            2D vector 

    Returns:
        intersections: (n,2) float
            intersections among given vectors
        line_has_inters: list of int
            indices of qv+sv vectors that has intersections
    """
    cross = rv[0]*sv[:, 1]-rv[1]*sv[:, 0]
    cross[cross == 0] = 1
    qv_minus_pv = qv - pv
    t = (qv_minus_pv[:, 0]*sv[:, 1]-qv_minus_pv[:, 1]*sv[:, 0]) / cross
    u = (qv_minus_pv[:, 0]*rv[1]-qv_minus_pv[:, 1]*rv[0]) / cross
    line_has_inters = np.where(
        ((t < 1-tol) & (t > tol)) & ((u < 1-tol) & (u > tol)) & (cross != 0))[0]
    if line_has_inters.shape[0] != 0:
        intersections = pv + t[line_has_inters].reshape(-1, 1) * rv
        return intersections, line_has_inters
    else:
        return None, None


def tracing_outline_robust(verts, faces):
    """Return ture outline points(includes intersections) and indices of verts lie on outline 

    Args:
        verts: (n,2)
            array of verts
        faces: (m,3)
            array of faces

    Returns:
        out_points: (-1,2) float
            array of true outline
        out_id:(-1,2) int
            indices of verts lie on outline 
    """
    start_id = np.argmin(verts[:, 0])
    center_pt = verts[start_id]
    pre_pt = center_pt.copy()
    pre_pt[0] = pre_pt[0] - 1  # start from left
    next_id = start_id
    break_c = verts.shape[0]
    i = 0
    edges_list, adj = get_edges_adj(verts, faces)
    edge_arr = np.vstack(edges_list)
    edge_arr = edge_arr.astype('int')
    out_points = []
    connect_id = []
    out_id = []
    out_id.append(next_id)
    out_points.append(verts[next_id])

    while True and i < break_c:
        i += 1
        if len(connect_id) == 0:
            connect_id = adj[next_id]
        vector_a = norm_vectors(pre_pt - center_pt)
        vector_b_arr = verts[connect_id] - center_pt
        vector_b_arr = norm_vectors(vector_b_arr)
        # filter point in connect_id that lies in vector_a (same direction)
        not_lie_in_vector_a = np.where(
            np.abs(np.dot(vector_b_arr, vector_a)-1) > tol)[0]
        connect_id = np.ndarray.tolist(
            np.asarray(connect_id)[not_lie_in_vector_a])
        vector_b_arr = verts[connect_id] - center_pt
        clockwise_nearest = find_clockwise_nearest(vector_a, vector_b_arr)
        next_id = int(connect_id[clockwise_nearest])
        if next_id == start_id:
            break
        pre_pt = center_pt
        center_pt = verts[next_id]
        arr_q = verts[edge_arr[:, 0]]
        arr_r = verts[edge_arr[:, 1]] - arr_q
        inters, inter_edge_id = find_inters(
            center_pt, pre_pt-center_pt, arr_q, arr_r)
        # double check, filter inters that is actually the node of the edge
        if inters is not None:
            inters_filter = []
            for j, inter in enumerate(inters):
                if np.linalg.norm(inter-center_pt) < tol or np.linalg.norm(inter-pre_pt) < tol:
                    continue
                else:
                    inters_filter.append(inter)
            if len(inters_filter) > 0:
                inters = np.vstack(inters_filter)
            else:
                inters = None

        if inters is not None:
            nearest = np.argmin(np.linalg.norm(inters - pre_pt, axis=1))
            center_pt = inters[nearest]
            connect_id = np.ndarray.tolist(edge_arr[inter_edge_id[nearest]])
            inters = None
        else:
            connect_id = []
            inters = None
            out_id.append(next_id)
        out_points.append(center_pt)
    out_points = np.asarray(out_points)
    return out_points, out_id
