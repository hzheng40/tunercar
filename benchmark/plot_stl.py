import numpy
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.colors import LightSource

def plotSTL(filename):
    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL mesh
    stlmesh = mesh.Mesh.from_file(filename)
    polymesh = mplot3d.art3d.Poly3DCollection(stlmesh.vectors)

    # Create light source
    ls = LightSource(azdeg=225, altdeg=45)

    # Darkest shadowed surface, in rgba
    dk = numpy.array([0.2, 0.0, 0.0, 1])
    # Brightest lit surface, in rgba
    lt = numpy.array([0.7, 0.7, 1.0, 1])
    # Interpolate between the two, based on face normal
    shade = lambda s: (lt-dk) * s + dk

    # Set face colors
    sns = ls.shade_normals(stlmesh.get_unit_normals(), fraction=1.0)
    rgba = numpy.array([shade(s) for s in sns])
    polymesh.set_facecolor(rgba)

    axes.add_collection3d(polymesh)

    # Adjust limits of axes to fill the mesh, but keep 1:1:1 aspect ratio
    pts = stlmesh.points.reshape(-1,3)
    ptp = max(numpy.ptp(pts, 0))/2
    ctrs = [(min(pts[:,i]) + max(pts[:,i]))/2 for i in range(3)]
    lims = [[ctrs[i] - ptp, ctrs[i] + ptp] for i in range(3)]
    axes.auto_scale_xyz(*lims)

    pyplot.show()

if __name__ == '__main__':
    plotSTL('./Geom.stl')