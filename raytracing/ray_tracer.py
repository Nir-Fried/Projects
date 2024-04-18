import argparse
from PIL import Image
import numpy as np
import random

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

import time

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array, path):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image in the given path
    image.save(path)

def computeCCS(camera):
    # Compute the camera coordinate system
    up_vector = camera.up_vector / np.linalg.norm(camera.up_vector) # normalize the up vector
    up_vector = np.array(up_vector)

    direction = np.array(camera.look_at) - np.array(camera.position) # [lx - px, ly - py, lz - pz]
    Vx = np.cross(direction, up_vector) # will give a vector perpendicular to both direction and up_vector
    Vx = Vx / np.linalg.norm(Vx) # normalize Vx

    up_vector = np.cross(Vx, direction) # will give a vector perpendicular to both Vx and direction

    Vy = up_vector / np.linalg.norm(up_vector)
    Vz = direction / np.linalg.norm(direction)

    return Vx, Vy, Vz

def computeP(camera, Vx, Vy, Vz, height, width):
    ratio = height / width
    
    plane_width = camera.screen_width
    plane_height = plane_width * ratio 

    Pc = np.array(camera.position) + camera.screen_distance * Vz # the center of the screen is the camera position + the screen distance in the direction of Vz
    P = Pc - (plane_width / 2) * Vx - (plane_height / 2) * Vy # the bottom left corner of the screen is the center of the screen - half the screen width in the direction of Vx - half the screen height in the direction of Vy

    return P, plane_height

def computeIntersectionPlane(V, camera, planes):

    best_t = np.inf
    best_plane = (None, best_t, None, None)

    for i, plane in enumerate(planes):
        N = np.array(plane[0:3]) # the normal of the plane
        N = N / np.linalg.norm(N) # normalize N
        c = plane[3] # the offset of the plane
        t = -(np.dot(np.array(camera), N) - c) / np.dot(V, N) # the intersection point of the plane
        if t < best_t and t > 0:
            best_t = t
            best_plane = (plane, best_t, "pln", i)

    return best_plane

def computeIntersectionSphere(V, camera, spheres):
    # Using the geometric solution from the slides
    best_t = np.inf
    best_sphere = (None, best_t, None, None)

    for i, sphere in enumerate(spheres):

        center = np.array(sphere[0:3]) # the center of the sphere
        radius = sphere[3] # the radius of the sphere

        L = center - np.array(camera) # the vector from the camera position to the center of the sphere
        t_ca = np.dot(L, V) # the distance from the camera position to the closest point on the ray to the center of the sphere

        if t_ca < 0: # if the distance is negative, then the ray is pointing away from the sphere
            continue
        d2 = np.dot(L, L) - (t_ca**2) # the distance from the closest point on the ray to the center of the sphere to the center of the sphere
        if d2 - (radius**2) > 0: # if the distance is greater than the radius, then the ray does not intersect the sphere
            continue
        
        t_hc = (radius**2 - d2)**0.5 # the distance from the closest point on the ray to the center of the sphere to the intersection point
        t = min(t_ca - t_hc, t_ca + t_hc, best_t) # the distance from the camera position to the intersection point

        if t < best_t and t > 0:
            best_t = t
            best_sphere = (sphere, best_t, "sph", i)

    return best_sphere

def cubeIntersection(cube, camera, V):
    # using the slab method 
    cube_center = np.array(cube[0:3])
    cube_edge = cube[3]

    tx1 = (cube_center[0] + cube_edge/2 - camera[0]) / V[0] # the x coordinate of the right side of the cube
    tx2 = (cube_center[0] - cube_edge/2 - camera[0]) / V[0]

    tx_min = min(tx1, tx2)
    tx_max = max(tx1, tx2)

    ty1 = (cube_center[1] + cube_edge/2 - camera[1]) / V[1] # the y coordinate of the top of the cube
    ty2 = (cube_center[1] - cube_edge/2 - camera[1]) / V[1] 

    ty_min = min(ty1, ty2)
    ty_max = max(ty1, ty2)

    if tx_min > ty_max or ty_min > tx_max: # if the ray misses the cube
        return -1
    
    # we want the intersection of the two slabs
    t_min = max(tx_min, ty_min)
    t_max = min(tx_max, ty_max)

    tz1 = (cube_center[2] + cube_edge/2 - camera[2]) / V[2] # the z coordinate of the front of the cube
    tz2 = (cube_center[2] - cube_edge/2 - camera[2]) / V[2]

    tz_min = min(tz1, tz2)
    tz_max = max(tz1, tz2)

    if t_min > tz_max or tz_min > t_max: # if the ray misses the cube
        return -1
    
    t = max(t_min, tz_min)

    return t

def computeIntersectionCube(V, camera, cubes):
    best_t = np.inf
    best_cube = (None, best_t, None, None)
    for i, cube in enumerate(cubes):
        t = cubeIntersection(cube, camera, V)
        if t < best_t and t > 0:
            best_t = t
            best_cube = (cube, best_t, "box", i)
    
    return best_cube

def computeNormalPlane(plane, intersection_point):
    N = np.array(plane[0:3]) # the normal of the plane
    N = N / np.linalg.norm(N) # normalize N
    return N

def computeNormalSphere(sphere, intersection_point):
    center = np.array(sphere[0:3]) # the center of the sphere
    N = intersection_point - center # the normal of the sphere
    N = N / np.linalg.norm(N) # normalize N
    return N

def float_equals(a, b):
    return abs(a - b) < 0.001

def computeNormalCube(cube, intersection_point):
    cube_center = np.array(cube[0:3])
    cube_edge = cube[3]
    x = intersection_point[0]
    y = intersection_point[1]
    z = intersection_point[2]

    # check if the intersection point is on the surface of the cube
    # if it is, then the normal is the direction of the surface

    if float_equals(x, cube_center[0] + cube_edge/2):
        N = np.array([1, 0, 0], dtype='float')
    elif float_equals(x, cube_center[0] - cube_edge/2):
        N = np.array([-1, 0, 0], dtype='float')
    elif float_equals(y, cube_center[1] + cube_edge/2):
        N = np.array([0, 1, 0], dtype='float')
    elif float_equals(y, cube_center[1] - cube_edge/2):
        N = np.array([0, -1, 0], dtype='float')
    elif float_equals(z, cube_center[2] + cube_edge/2):
        N = np.array([0, 0, 1], dtype='float')
    elif float_equals(z, cube_center[2] - cube_edge/2):
        N = np.array([0, 0, -1], dtype='float')
    else:
        print("should not reach here")
        N = np.array([0, 0, 0], dtype='float')

    return N

def computeSoftShadow(light, normal, intersection_point, num_shadow_rays, planes, spheres, cubes):
    
    light_position = np.array(light[0:3])
    light_radius = light[-1]

    c = np.dot(normal, light_position) # the constant in the plane equation
    if normal[2] != 0: # if the normal is not parallel to the xy plane
        z = -(normal[0] + normal[1] + c) / normal[2]
    else:
        z = 0
    v = np.array([1, 1, z]) - light_position # the vector from the light to the point (1, 1, z),
    v = v / np.linalg.norm(v) # normalize v

    u = np.cross(normal, v) # the vector perpendicular to both normal and v
    u = u / np.linalg.norm(u) # normalize u

    # calculate the starting point of the light using the light radius:
    starting_point = light_position.copy() - v * (0.5 * light_radius) - u * (0.5 * light_radius) 

    # grid is N x N
    interval = 1 / num_shadow_rays # the interval between each shadow ray

    v, u = v * light_radius, u * light_radius # scale v and u by the light radius
    v, u = v * interval, u * interval # scale v and u by the interval

    res = 0
    for i in range(num_shadow_rays):
        for j in range(num_shadow_rays):
            x = random.random()
            y = random.random()

            p = starting_point + v * (i + x) + u * (j + y) # the point on the light
            direction = p - intersection_point # the direction of the shadow ray

            plength = np.dot(direction, direction) ** 0.5 # the length of the direction vector
            direction = direction / np.linalg.norm(direction) # normalize direction

            pos = intersection_point + (direction * 0.001) # to avoid self intersection
            tempCube = computeIntersectionCube(direction, pos, cubes)
            tempSphere = computeIntersectionSphere(direction, pos, spheres)
            tempPlane = computeIntersectionPlane(direction, pos, planes)

            best_intersection = min(tempCube, tempSphere, tempPlane, key=lambda x: x[1])
            
            if (best_intersection[1] < plength):
                pass
            else: # if the shadow ray does not intersect with any object
                res += 1
    
    # return how many rays hit the required point out of the total number of rays
    return res / (num_shadow_rays ** 2)
         
def computeColor(best_intersection, intersection_point, N, V, lights, settings, materials, rec_depth, planes, spheres, cubes):
    settings_colors = [settings.background_color[0], settings.background_color[1], settings.background_color[2]]
    if rec_depth == 0:
        return settings_colors
    
    cur_mat = materials[int(best_intersection[0][-1]) - 1] # the material of the current object

    diff_color = np.array(cur_mat[0:3]) # the diffuse color of the object
    spec_color = np.array(cur_mat[3:6]) # the specular color of the object
    phong = cur_mat[9] # the phong exponent of the object
    trans = cur_mat[10] # the transparency of the object

    output_color = [0,0,0]

    for light in lights:
        L = np.array(light[0:3]) - intersection_point # the direction of the light
        L = L / np.linalg.norm(L) # normalize L
        lgt_color = np.array(light[3:6]) # the color of the light
        lgt_spec = light[6] # the specular intensity of the light

        theta = np.dot(N, L) # the cosine of the angle between N and L
        if theta < 0: # if the light is behind the object
            continue
        
        # calculate the diffuse component of the color
        r = theta * diff_color[0] * lgt_color[0] 
        g = theta * diff_color[1] * lgt_color[1]
        b = theta * diff_color[2] * lgt_color[2]

        R = N * (np.dot(2*L, N)) - L # the direction of the reflected ray

        spec_comp = np.dot(R, -V) ** phong # the specular component of the color

        # calculate the specular component of the color
        r += spec_comp * spec_color[0] * lgt_color[0] * lgt_spec 
        g += spec_comp * spec_color[1] * lgt_color[1] * lgt_spec 
        b += spec_comp * spec_color[2] * lgt_color[2] * lgt_spec 

        # find the ratio of the light that is blocked by other objects
        soft_shadow = computeSoftShadow(light, -L, intersection_point, int(settings.root_number_shadow_rays), planes, spheres, cubes)

        # (1-shadow_intensity) + (shadow_intensity * (% of rays that hit the point))
        output_color[0] += r * ((1 - light[-2]) + light[-2] * soft_shadow) 
        output_color[1] += g * ((1 - light[-2]) + light[-2] * soft_shadow) 
        output_color[2] += b * ((1 - light[-2]) + light[-2] * soft_shadow) 

    trans_color = np.zeros(3)
    if trans > 0: # if the object is transparent, compute the transparency color
        direction = V.copy()
        direction = direction / np.linalg.norm(direction) # normalize direction
        pos = intersection_point + (direction * 0.001) # avoid self intersection
        tempCube = computeIntersectionCube(direction, pos, cubes)
        tempSphere = computeIntersectionSphere(direction, pos, spheres)
        tempPlane = computeIntersectionPlane(direction, pos, planes)

        best_intersection2 = min(tempCube, tempSphere, tempPlane, key=lambda x: x[1])
        if best_intersection2[1] < np.inf: # if the ray intersects with an object
            #update the intersection point and normal
            intersection_point2 = pos + (direction * best_intersection2[1])
            if best_intersection2[2] == "box":
                N = computeNormalCube(best_intersection2[0], intersection_point2)
            elif best_intersection2[2] == "sph":
                N = computeNormalSphere(best_intersection2[0], intersection_point2)
            elif best_intersection2[2] == "pln":
                N = computeNormalPlane(best_intersection2[0], intersection_point2)

            temp_output_color = computeColor(best_intersection2, intersection_point2, N, V, lights, settings, materials, rec_depth - 1, planes, spheres, cubes)
        else:
            temp_output_color = settings_colors
        
        temp_output_color[0] = min(temp_output_color[0], 1) 
        temp_output_color[1] = min(temp_output_color[1], 1)
        temp_output_color[2] = min(temp_output_color[2], 1)

        trans_color = np.array(temp_output_color)
    
    reflect_color = np.zeros(3)
    if cur_mat[6:9].any() > 0:  # if the object is reflective, compute the reflection color
        direction = V.copy()
        direction = direction / np.linalg.norm(direction) # normalize direction
        ret = [0,0,0]

        # reflection angle: (from slide 20 in the ray tracing slides)
        R = direction + (N * -2 * (np.dot(direction, N))) # the direction of the reflected ray
        R = R / np.linalg.norm(R) # normalize R

        pos = intersection_point + (R * 0.001) # avoid self intersection
        
        tempCube = computeIntersectionCube(R, pos, cubes)
        tempSphere = computeIntersectionSphere(R, pos, spheres)
        tempPlane = computeIntersectionPlane(R, pos, planes)

        best_intersection2 = min(tempCube, tempSphere, tempPlane, key=lambda x: x[1])
        if best_intersection2[1] < np.inf: # if the ray intersects with an object
            #update the intersection point and normal
            intersection_point2 = pos + (R * best_intersection2[1])
            if best_intersection2[2] == "box":
                N = computeNormalCube(best_intersection2[0], intersection_point2)
            elif best_intersection2[2] == "sph":
                N = computeNormalSphere(best_intersection2[0], intersection_point2)
            elif best_intersection2[2] == "pln":
                N = computeNormalPlane(best_intersection2[0], intersection_point2)

            temp_output_color = computeColor(best_intersection2, intersection_point2, N, R, lights, settings, materials, rec_depth - 1, planes, spheres, cubes)
            
            # to find the reflection color, multiply the color of the reflected ray by the reflectivity of the material
            ret[0] = temp_output_color[0] * cur_mat[6] 
            ret[1] = temp_output_color[1] * cur_mat[7] 
            ret[2] = temp_output_color[2] * cur_mat[8] 
        else:
            # if the ray does not intersect with an object, the reflection color is the background color multiplied by the reflectivity of the material
            ret[0] = settings_colors[0] * cur_mat[6]
            ret[1] = settings_colors[1] * cur_mat[7] 
            ret[2] = settings_colors[2] * cur_mat[8] 
        
        # clamp the reflection color to be between 0 and 1 to avoid overflow
        ret[0] = min(ret[0], 1)
        ret[1] = min(ret[1], 1)
        ret[2] = min(ret[2], 1)
        reflect_color = np.array(ret)
    
    # (diffuse + specular) * (1-transparency) + (background color * transperency) + (reflection color)
    output_color[0] = output_color[0] * (1-trans) + trans_color[0] * trans + reflect_color[0]
    output_color[1] = output_color[1] * (1-trans) + trans_color[1] * trans + reflect_color[1] 
    output_color[2] = output_color[2] * (1-trans) + trans_color[2] * trans + reflect_color[2] 

    # clamp the output color to be between 0 and 1 to avoid overflow
    output_color[0] = min(output_color[0], 1)
    output_color[1] = min(output_color[1], 1)
    output_color[2] = min(output_color[2], 1)

    return output_color

def main():
    start = time.time()

    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # process the objects in the scene:
    materials = np.empty((0, 11))
    lights = np.empty((0, 9))
    planes = np.empty((0, 5))
    spheres = np.empty((0, 5))
    cubes = np.empty((0, 5))

    for o in objects:
        if isinstance(o, Material):
            temp = [o.diffuse_color[0], o.diffuse_color[1], o.diffuse_color[2], o.specular_color[0], o.specular_color[1], o.specular_color[2], o.reflection_color[0], o.reflection_color[1], o.reflection_color[2], o.shininess, o.transparency]
            row = np.asarray([float(num) for num in temp])
            materials = np.vstack([materials, row])
        elif isinstance(o, Light):
            temp = [o.position[0], o.position[1], o.position[2], o.color[0], o.color[1], o.color[2], o.specular_intensity, o.shadow_intensity, o.radius]
            row = np.asarray([float(num) for num in temp])
            lights = np.vstack([lights, row])
        elif isinstance(o, InfinitePlane):
            temp = [o.normal[0], o.normal[1], o.normal[2], o.offset, o.material_index]
            row = np.asarray([float(num) for num in temp])
            planes = np.vstack([planes, row])
        elif isinstance(o, Sphere):
            temp = [o.position[0], o.position[1], o.position[2], o.radius, o.material_index]
            row = np.asarray([float(num) for num in temp])
            spheres = np.vstack([spheres, row])
        elif isinstance(o, Cube):
            temp = [o.position[0], o.position[1], o.position[2], o.scale, o.material_index]
            row = np.asarray([float(num) for num in temp])
            cubes = np.vstack([cubes, row])
        else:
            raise ValueError("Unknown object type: {}".format(o.__class__.__name__))
    
    output_image_path = args.output_image
    width = args.width
    height = args.height
    
    bgr, bgg, bgb = scene_settings.background_color

    img_output = np.full((height, width, 3),[bgr, bgg, bgb])

    Vx, Vy, Vz = computeCCS(camera) # compute the camera coordinate system
    P, plane_height = computeP(camera, Vx, Vy, Vz, height, width) # compute the initial P (the top left corner of the image plane)

    for i in range(height):
        temp_P = P.copy()
        # if ((i+1) % 10 == 0):
        #     print("iteration {} out of {}".format(i+1, height))

        for j in range(width):

            # V = (P - P0) / ||P - P0|| 
            V = temp_P - np.array(camera.position) # V is the directional vector from the camera position to the pixel that we are currently at
            V = V / np.linalg.norm(V) # normalize V

            camera_pos = [camera.position[0], camera.position[1], camera.position[2]]

            # compute the intersection of the ray with the objects in the scene
            best_cube = computeIntersectionCube(V, camera_pos, cubes)
            best_sphere = computeIntersectionSphere(V, camera_pos, spheres)
            best_plane = computeIntersectionPlane(V, camera_pos, planes)

            # find the closest intersection
            best_intersection = min(best_cube, best_sphere, best_plane, key=lambda x: x[1])

            # if there is no intersection, then the pixel is the background color
            if best_intersection[1] == np.inf:
                img_output[i,j] = np.array([bgr, bgg, bgb]) * 255
            else:
                # compute the intersection point
                P_int = np.array(camera.position) + best_intersection[1] * V

                # compute the normal at the intersection point
                if best_intersection[2] == "box":
                    N = computeNormalCube(best_intersection[0], P_int)
                elif best_intersection[2] == "sph":
                    N = computeNormalSphere(best_intersection[0], P_int)
                elif best_intersection[2] == "pln":
                    N = computeNormalPlane(best_intersection[0], P_int)
                else:
                    raise ValueError("Unknown object type: {}".format(best_intersection[2]))
                
                if np.dot(N, V) > 0: # if the dot product of N and V is positive, then we need to flip N since we want it to point towards the camera
                    N = -N
                N = N / np.linalg.norm(N) # normalize N

                # compute the color at the intersection point
                color = computeColor(best_intersection, P_int, N, V, lights, scene_settings, materials, scene_settings.max_recursions, planes, spheres, cubes)

                # set the pixel to the color
                img_output[i,j] = np.array(color) * 255
            
            temp_P = temp_P + Vx * (camera.screen_width / width) # update temp_P so that we are at the next column of pixels in the same row
        
        P = P + Vy * (plane_height / height) # update P so that we are at the next column of pixels in the same row

    img_output = np.flip(img_output, axis=(0,1)) # flip the image so that it is right side up
    end = time.time()
    print("Time elapsed: {} minutes".format((end - start) / 60))
    save_image(img_output, output_image_path)


if __name__ == '__main__':
    main()
