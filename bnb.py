from simplex import *
import time
from pysmps import smps_loader
from heapq import *
from itertools import count


def read_mps(file):
    m = smps_loader.load_mps(file)
    I = np.array([i == 'integral' for i in m[4]])
    s_dict = { 'E': EQ, 'L': LE, 'G': GE }
    s = np.array([s_dict[i] for i in m[5]])
    c = m[6]
    A = m[7]
    b = next(iter(m[9].values()))


    lo, up = None, None
    if m[11]:
        bnd = next(iter(m[11].values()))

        if 'LO' in bnd:
            lo = bnd['LO']
            lo_mask = lo > 0
            lo_vals = lo
            lo = lo_mask, lo_vals

        if 'UP' in bnd:
            up = bnd['UP']
            up_mask = np.isfinite(up)
            up_vals = up
            up = up_mask, up_vals

    return A, b, c, s, I, lo, up


def solve_relaxation(A, b, c, s, lo, up, fixed):
    lo_mask, lo_vals = lo 
    up_mask, up_vals = up

    fixed_mask, fixed_vals = fixed
    old_fv = fixed_vals
    fixed_vals = fixed_vals[fixed_mask]

    m, n, fixed_z = len(b), len(c), c[fixed_mask] @ fixed_vals
    any_fixed = len(fixed_vals) > 0

    if any_fixed:
        not_fixed = ~fixed_mask
        lo_mask = lo_mask & not_fixed
        up_mask = up_mask & not_fixed

    A = np.vstack((A, np.eye(n)[lo_mask, :], np.eye(n)[up_mask, :]))
    b = np.hstack((b, lo_vals[lo_mask], up_vals[up_mask]))
    s = np.hstack((s, GE * np.ones(np.sum(lo_mask)), 
                   LE * np.ones(np.sum(up_mask))))

    if any_fixed:
        b[:m] -= A[:m, fixed_mask] @ fixed_vals
        A[:m, fixed_mask] = 0
        c = np.copy(c)
        c[fixed_mask] = 0

    sln = two_phase_simplex(A, b, c, s)
    if sln:
        x, z = sln
        x[fixed_mask] = fixed_vals

        return x, z + fixed_z

    return None


def tighten_up(i, bnd, up):
    up_mask, up_vals = np.copy(up[0]), np.copy(up[1])
    up_mask[i] = True
    up_vals[i] = min(up_vals[i], bnd)

    return up_mask, up_vals


def tighten_lo(i, bnd, lo):
    lo_mask, lo_vals = np.copy(lo[0]), np.copy(lo[1])
    lo_mask[i] = True
    lo_vals[i] = max(lo_vals[i], bnd)

    return lo_mask, lo_vals


def fix_var(i, fx, fixed):
    fixed_mask, fixed_vals = np.copy(fixed[0]), np.copy(fixed[1])
    fixed_mask[i] = True
    fixed_vals[i] = fx

    return fixed_mask, fixed_vals


def branch_and_bound(A, b, c, s, I, lo=None, up=None, target_gap=1e-5):
    m, n = A.shape
    EPS = 1e-8

    if lo is None:
        lo = np.array([True] * n), np.zeros(n)
    if up is None:
        up = np.array([False] * n), np.ones(n) * np.inf
    fixed = np.array([False] * n), np.zeros(n)

    counter = count()
    q = []
    q.append((n, 0, -1e20, next(counter), lo, up, fixed))

    if I is None:
        I = np.array([True] * n)

    best_sol, primal_bound = None, 1e20

    while q:
        n_vars, parent_z, _, k, lo, up, fixed = heappop(q)

        sln = solve_relaxation(A, b, c, s, lo, up, fixed)
        if sln is None:
            continue

        x, z = sln
        if z >= primal_bound:
            continue

        nonint_mask = np.abs(x - np.round(x)) > EPS
        non_ints = np.where(nonint_mask & I)[0]

        if k % 1000 == 0:
            print(k, z, n_vars, len(non_ints))

        lo_mask, lo_vals = lo
        up_mask, up_vals = up
        fixed_mask, fixed_vals = fixed

        if len(non_ints) == 0:
            if best_sol is None or z < primal_bound:
                best_sol, primal_bound = x, z

                # gap = primal_bound - parent_z
                # gap /= max(EPS, min(abs(primal_bound), 
                #                       abs(parent_z)))

                print(f'new best feasible {z} {k}')
            continue

        for i in non_ints:
            bnd = np.floor(x[i])
            if lo_vals[i] <= bnd and bnd <= up_vals[i]:
                if bnd - lo_vals[i] <= EPS:
                    new_fixed = fix_var(i, bnd, fixed)
                    heappush(q, (n_vars - 1, z, -(x[i] - bnd), next(counter), lo, up, new_fixed))
                else:
                    new_up = tighten_up(i, bnd, up)
                    heappush(q, (n_vars, z, -(x[i] - bnd), next(counter), lo, new_up, fixed))
            bnd += 1
            if lo_vals[i] <= bnd and bnd <= up_vals[i]:
                if up_vals[i] - bnd <= EPS:
                    new_fixed = fix_var(i, bnd, fixed)
                    heappush(q, (n_vars - 1, z, -(bnd - x[i]), next(counter), lo, up, new_fixed))
                else:
                    new_lo = tighten_lo(i, bnd, lo)
                    heappush(q, (n_vars, z, -(bnd - x[i]), next(counter), new_lo, up, fixed))


    return best_sol, primal_bound #, gap


def main():
    #A = np.array([[-2, 2], [-8, 10]])
    #b = np.array([1, 13])
    #c = -np.ones(2)
    #s = np.array([GE, LE])
    #I = np.array([True, True])

    weight = np.array([10, 20, 30])
    cost = np.array([60, 100, 120])

    A = np.array(
        [[ 10, 20, 30],
         [  1,  0,  0],
         [  0,  1,  0],
         [  0,  0,  1]])
    b = np.array([50, 1, 1, 1])
    c = -cost
    s = np.array([LE] * 4)
    I = np.array([True] * 3)

    up = np.array([True] * 3), np.ones(3)

    branch_and_bound(A, b, c, s, I, up=up)


if __name__ == '__main__':
    #np.set_printoptions(edgeitems=30, linewidth=100000) 

    # A, b, c, s, I, lo, up = read_mps('markshare_4_0.mps')
    # branch_and_bound(A, b, c, s, I, lo=lo, up=up)

    main()

