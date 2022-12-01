from ngsolve import *
import netgen.meshing as ngmeshing
import numpy as np
import scipy.sparse as sp

def main(filename, order, curve, bi):
    # Loading the object file
    ngmesh = ngmeshing.Mesh(dim=3)
    ngmesh.Load(filename)

    # Creating the mesh and assigning curvature
    mesh = Mesh(filename)
    mesh.Curve(curve)
    numelements = mesh.ne  # Count the number elements
    print(" mesh contains " + str(numelements) + " elements")

    # Setting up complex finite element space.
    fes = HCurl(mesh, order=order, dirichlet="default", complex=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()

#   trig function
    fx = CoefficientFunction(3*sin(x + y)*(2+1j))
    fy = CoefficientFunction(-3*sin(x + y)*(2+1j))
    fz = CoefficientFunction(0*(2+1j))
    f = CoefficientFunction((fx, fy, fz))

    ex = CoefficientFunction(sin(x + y)*(2+1j))
    ey = CoefficientFunction(-sin(x + y)*(2+1j))
    ez = CoefficientFunction(0*(2+1j))
    e = CoefficientFunction((ex, ey, ez))

#   linear function - should be exact for p=0,1,2,3,etc
    A = BilinearForm(fes)
    A += SymbolicBFI(curl(u) * curl(v),bonus_intorder=bi)
    A += SymbolicBFI(u * v,bonus_intorder=bi)
    A.Assemble()

    F = LinearForm(fes)
    F += SymbolicLFI(f * v,bonus_intorder=bi)
    F.Assemble()

    gfu = GridFunction(fes)
    gfu.Set((ex, ey, ez), BND)

    # Applying BC:
    r = F.vec.CreateVector()
    r.data = F.vec - A.mat * gfu.vec

    gfu.vec.data += A.mat.Inverse(freedofs=fes.FreeDofs()) * r

#   Compare some integration of inner products
    Intval = Integrate(gfu*gfu,mesh,order=2*(order+1))
    M = BilinearForm(fes)
    M += SymbolicBFI(u * v,bonus_intorder=bi)
    M.Assemble()

    rows, cols, vals = M.mat.COO()
    Msci = sp.csr_matrix((vals, (rows, cols)))
    SciInt = gfu.vec.FV().NumPy()[:] @ (Msci @ gfu.vec.FV().NumPy()[:])

    print(f'Integral: {Intval}')
    print(f'Matrix: {SciInt}')
    print ("Order =",order,"Rel Error in (gfu,gfu)_\Omega", np.abs(SciInt-Intval) / np.abs(Intval))

    return 0

if __name__ == '__main__':

    # Running function for a unit radius sphere with a purely tetrahedral mesh
    filename = 'sphere.vol'
    order = 1
    bi = 0
    for curve in [1,2,3]:
        print(f'solving for curve = {curve}')
        main(filename, order, curve, bi)
    print('\n\n\n')
    # Running function for unit radius sphere with an increased bonus_intval applied to the bilinear and linear forms
    bi = 2
    for curve in [1, 2, 3]:
        print(f'solving for curve = {curve}')
        main(filename, order, curve, bi)
    print('\n\n\n')

    # Running similar example for a unit sphere with a thin layer of prismatic elements.
    filename = 'sphere_prism.vol'
    bi = 0
    for curve in [1, 2, 3]:
        print(f'solving for curve = {curve}')
        main(filename, order, curve, bi)