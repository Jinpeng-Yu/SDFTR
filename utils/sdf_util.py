import numpy as np
from numpy import mat
import trimesh
from skimage import measure
import open3d as o3d

# coordinate array
def sdCube(points):

    points = mat(points)
    semi_x, semi_y, semi_z = 0.1, 0.1, 0.1
    # print(semi_z)
    sdf_value = np.sqrt(np.sum(np.square(np.maximum(np.concatenate((abs(points[:,0])-semi_x, abs(points[:,1])-semi_y, abs(points[:,2])-semi_z) ,axis = 1),0)), axis=1))\
    + np.minimum(np.maximum(abs(points[:,0])-semi_x,np.maximum(abs(points[:,1])-semi_y, abs(points[:,2])-semi_z)), 0)

    return sdf_value

def genCube(params):

    # print(np.ravel(params))
    delta_x, delta_y, delta_z, scale_x, scale_y, scale_z, theta_x, theta_y, theta_z = [item for item in np.ravel(params)]
    # Translation Matrix
    TMatrix = np.array([[1., 0., 0., -delta_x],
                        [0., 1., 0., -delta_y],
                        [0., 0., 1., -delta_z],
                        [0., 0., 0., 1.]])

    # Scale Matrix
    SMatrix = np.array([[1/scale_x, 0., 0., 0.],
                        [0., 1/scale_y, 0., 0.],
                        [0., 0., 1/scale_z, 0.],
                        [0., 0., 0., 1.]])

    # Rotation Matrix
    # theta_x = np.pi * theta_x / 180
    # RxMatrix = np.array([[1., 0., 0., 0.],
    #                      [0., np.cos(theta_x), -np.sin(theta_x), 0.],
    #                      [0., np.sin(theta_x),  np.cos(theta_x), 0.],
    #                      [0., 0., 0., 1.]])
    RxMatrix = np.array([[1., 0., 0., 0.],
                         [0., np.cos(theta_x), np.sin(theta_x), 0.],
                         [0., -np.sin(theta_x),  np.cos(theta_x), 0.],
                         [0., 0., 0., 1.]])

    # theta_y = np.pi * theta_y / 180
    # RyMatrix = np.array([[np.cos(theta_y), 0., np.sin(theta_y), 0.],
    #                      [0., 1., 0., 0.],
    #                      [-np.sin(theta_y), 0.,  np.cos(theta_y), 0.],
    #                      [0., 0., 0., 1.]])
    RyMatrix = np.array([[np.cos(theta_y), 0., -np.sin(theta_y), 0.],
                         [0., 1., 0., 0.],
                         [np.sin(theta_y), 0.,  np.cos(theta_y), 0.],
                         [0., 0., 0., 1.]])

    # theta_z = np.pi * theta_z / 180
    # RzMatrix = np.array([[np.cos(theta_z), -np.sin(theta_z), 0., 0.],
    #                      [np.sin(theta_z),  np.cos(theta_z), 0., 0.],
    #                      [0., 0., 1., 0.],
    #                      [0., 0., 0., 1.]])
    RzMatrix = np.array([[np.cos(theta_z), np.sin(theta_z), 0., 0.],
                         [-np.sin(theta_z),  np.cos(theta_z), 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])

    # single coordinate
    # return lambda point: sdCube([item for item in RxMatrix.dot( RyMatrix.dot( RzMatrix.dot( SMatrix.dot( TMatrix.dot( np.append(point, 1).reshape(4,1)))))).T[0][0:3]])
    
    # coordinate array # Affine transform
    return lambda points: sdCube(np.delete(SMatrix.dot( RxMatrix.dot( RyMatrix.dot( RzMatrix.dot( TMatrix.dot( np.insert(points,3,values=1,axis=1).T))))).T, -1, axis=1))

def genBackground(params):
    
    def func(points):
        bg_render = np.empty([points.shape[0], params.shape[0]])
        for i in range(params.shape[0]):
            bg_render[:,[i]] = genCube(params[i])(points)
        return mat(np.min(bg_render, axis=1)).T

    return func

def genForeground(params):

    def func(points):
        fg_render = np.empty([points.shape[0], params.shape[0]])
        for i in range(params.shape[0]):
            fg_render[:,[i]] = genCube(params[i])(points)
        return mat(np.min(fg_render, axis=1)).T
    
    return func

def genTarget(params):

    def func(points):
        tg_render = np.empty([points.shape[0], params.shape[0]])
        for i in range(params.shape[0]):
            tg_render[:,[i]] = genCube(params[i])(points)
        return mat(np.min(tg_render, axis=1)).T
    
    return func

def sdf2Obj(filename, verts, faces, normals, values, affine = None, one = True):
    """
    Write a .obj file for the output of marching cube algorithm.

    Parameters
    -----------
    filename : str
        Ouput file name.
    verts : array
        Spatial coordinates for vertices as returned by skimage.measure.marching_cubes_lewiner().
    faces : array
        List of faces, referencing indices of verts as returned by skimage.measure.marching_cubes_lewiner().
    normals : array
        Normal direction of each vertex as returned by skimage.measure.marching_cubes_lewiner().
    affine : array,optional
        If given, vertices coordinates are affine transformed to create mesh with correct origin and size.
    one : bool
        Specify if faces values should start at 1 or at 0. Different visualization programs use different conventions.

    """
    if one: faces = faces + 1
    thefile = open(filename, 'w')
    if affine is not None:
        for item in verts:
            # transformed = f(item[0],item[1],item[2],affine) # True
            transformed = item[0],item[1],item[2] # False
            thefile.write('v {0} {1} {2}\n'.format(transformed[0], transformed[1], transformed[2]))
    else:
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))
    for item in normals:
        thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))
    for item in faces:
        thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))
    thefile.close()

def sdf2ply(filename, verts):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    o3d.io.write_point_cloud(filename, pcd)
    # o3d.visualization.draw_geometries([pcd])

# def sdf2ply(filename, verts, faces, text = True):
    
#     mesh = trimesh.Trimesh(verts, faces, process=False)
#     points, face_idx = mesh.sample(100000, return_index=True)
#     points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
#     vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
#     el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
#     PlyData([el], text=text).write(filename)

def scene_render(output_filename_prefix, 
                    bg_params, 
                    fg_params, 
                    tg_params, 
                    save_obj = False, 
                    save_ply = False):
    x_count = 257
    y_count = 257
    z_count = 257

    x = np.linspace(-1, 1, x_count)
    y = np.linspace(-1, 1, y_count)
    z = np.linspace(-1, 1, z_count)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    points = np.concatenate((mat(np.ravel(xv)), mat(np.ravel(yv)), mat(np.ravel(zv))), axis= 0).T

    bg = genBackground(bg_params)(points)
    fg = genForeground(fg_params)(points)
    tg = genTarget(tg_params)(points)

    bg_volume = np.array(bg).reshape(x_count, y_count, z_count)
    fg_volume = np.array(fg).reshape(x_count, y_count, z_count)
    tg_volume = np.array(tg).reshape(x_count, y_count, z_count)

    bg_verts, bg_faces, bg_normals, values = measure.marching_cubes(bg_volume,0)
    fg_verts, fg_faces, fg_normals, values = measure.marching_cubes(fg_volume,0)
    tg_verts, tg_faces, tg_normals, values = measure.marching_cubes(tg_volume,0)
    step_size = 2 / (x_count - 1)
    bg_verts = bg_verts * step_size - 1.0 # voxel grid coordinates to world coordinates
    fg_verts = fg_verts * step_size - 1.0 # voxel grid coordinates to world coordinates
    tg_verts = tg_verts * step_size - 1.0 # voxel grid coordinates to world coordinates
    if save_obj:
        sdf2Obj(output_filename_prefix+'_bg.obj', bg_verts, bg_faces, bg_normals, 0)
        sdf2Obj(output_filename_prefix+'_fg.obj', fg_verts, fg_faces, fg_normals, 0)
        sdf2Obj(output_filename_prefix+'_target.obj', tg_verts, tg_faces, tg_normals, 0)
    if save_ply:
        sdf2ply(output_filename_prefix+'_bg.ply', bg_verts, bg_faces)
        sdf2ply(output_filename_prefix+'_fg.ply', fg_verts, fg_faces)
        sdf2ply(output_filename_prefix+'_target.ply', tg_verts, tg_faces)