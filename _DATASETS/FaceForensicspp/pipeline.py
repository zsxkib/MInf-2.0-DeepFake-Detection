import cv2
import face_recognition
from PIL import Image
from tqdm import tqdm
import numpy as np
from operator import itemgetter

def get_every_frame(video , interval=1):
    frames = []
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()

    count = 0
    while success:
        if not count % interval:
            frames.append(image[:, :, ::-1])
        success, image = vidcap.read()
        count += 1
        
    return frames


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)


def save_faces(face_list, saving_dir):
    for i, face in enumerate(face_list):
        Image.fromarray(face, mode='RGB').save(f'{saving_dir}_{i}.png', 'PNG')


def filterNones(frames, faces):
    cleaned_frames, cleaned_face = [], []
    for fr, fa in zip(frames, faces):
        if fa is not None:
            cleaned_frames.append(fr)
            cleaned_face.append(fa)
    return cleaned_frames, cleaned_face
    

def get_face_locations(frames, GPU=False, batch_size=64):
    face_coordinates = []
    if GPU:
        for i in range(0, len(frames), batch_size):
            batch_of_frames = frames[i:i+batch_size]
            batch_face_locations = face_recognition.batch_face_locations(batch_of_frames, number_of_times_to_upsample=0)
            face_coordinates += batch_face_locations
        face_coordinates = [f[-1] if f is not None and len(f) else None for f in face_coordinates]
    else:
        for frame in tqdm(frames):
            coordinates_found = face_recognition.face_locations(frame)
            if coordinates_found is not None and len(coordinates_found):
                face_coordinates.append(coordinates_found[-1])
            else:
                face_coordinates.append(None)
        
    return face_coordinates


def get_centroid(face_coordinates):
    cleaned_face_coordinates = np.asarray([f for f in face_coordinates if f is not None])

    length = cleaned_face_coordinates.shape[0]
    sum_t = np.sum(cleaned_face_coordinates[:, 0])
    sum_r = np.sum(cleaned_face_coordinates[:, 1])
    sum_b = np.sum(cleaned_face_coordinates[:, 2])
    sum_l = np.sum(cleaned_face_coordinates[:, 3])

    return np.asarray((sum_t, sum_r, sum_b, sum_l)) / length


def get_distance_from_centroid(centroid, face_coordinates):
    dist_from_centroid = []
    for coord in face_coordinates:
        if coord is not None:
            dist_from_centroid.append(np.linalg.norm(centroid-coord))
        else:
            dist_from_centroid.append(coord)
            
    return dist_from_centroid


def get_stable_faces(movement_thershold, dist_from_centroid, face_coordinates, frames, zoomed=False):
    stable_faces = []
    if zoomed:
        for dist, coord, frame in zip(dist_from_centroid, face_coordinates, frames):
            if dist is not None:
                if dist <= movement_thershold:
                    t,r,b,l = coord
                    stable_faces.append(frame[t:b, l:r])
    else:
        face_coordinates_in_thrshld = []
        for dist, coord in zip(dist_from_centroid, face_coordinates):
            if dist is not None and dist <= movement_thershold:
                face_coordinates_in_thrshld.append(coord)

        face_coordinates_in_thrshld = [f for f in face_coordinates_in_thrshld if f is not None]
        min_top = min(face_coordinates_in_thrshld, key=itemgetter(0))[0]
        max_right = max(face_coordinates_in_thrshld, key=itemgetter(1))[1]
        max_bottom = max(face_coordinates_in_thrshld, key=itemgetter(2))[2]
        min_left = min(face_coordinates_in_thrshld, key=itemgetter(3))[3]

        for dist, frame in zip(dist_from_centroid, frames):
            if dist is not None and dist <= movement_thershold:
                stable_faces.append(frame[min_top:max_bottom, min_left:max_right])
                
    return stable_faces


def stabilise(face_coordinates, frames):
    stable_faces = []
    
    cleaned_face_coordinates = [f for f in face_coordinates if f is not None]
    min_top = min(cleaned_face_coordinates, key=itemgetter(0))[0]
    max_right = max(cleaned_face_coordinates, key=itemgetter(1))[1]
    max_bottom = max(cleaned_face_coordinates, key=itemgetter(2))[2]
    min_left = min(cleaned_face_coordinates, key=itemgetter(3))[3]

    for frame in frames:
        stable_faces.append(frame[min_top:max_bottom, min_left:max_right])
                
    return stable_faces


def sub_average(frames, face_coordinates, interval):
    avg = []

    for sub_frames, sub_face_coordinates in zip(grouped(frames, n=interval), grouped(face_coordinates, n=interval)):
        sub_frames, sub_face_coordinates = filterNones(sub_frames, sub_face_coordinates)
        if sub_face_coordinates is not None and len(sub_face_coordinates):
            crop = stabilise(sub_face_coordinates, sub_frames)
            avg.append(average(crop))

    return avg


def sub_difference(frames, face_coordinates, interval):
    diff = []

    for sub_frames, sub_face_coordinates in zip(grouped(frames, n=interval), grouped(face_coordinates, n=interval)):
        sub_frames, sub_face_coordinates = filterNones(sub_frames, sub_face_coordinates)
        if sub_face_coordinates is not None and len(sub_face_coordinates):
            crop = stabilise(sub_face_coordinates, sub_frames)
            diff.append(difference(crop))

    return diff


def sub_frames_every_interval(frames, face_coordinates, interval):
    frames_at_end_of_interval = []

    for sub_frames, sub_face_coordinates in zip(grouped(frames, n=interval), grouped(face_coordinates, n=interval)):
        sub_frames, sub_face_coordinates = filterNones(sub_frames, sub_face_coordinates)
        if sub_face_coordinates is not None and len(sub_face_coordinates):
            last_crop = stabilise(sub_face_coordinates, sub_frames)[-1]
            frames_at_end_of_interval.append(last_crop)

    return frames_at_end_of_interval


def average(frames):
    h,w,c = frames[0].shape
    N = len(frames)
    avg = np.zeros((h,w,c), np.float)

    for frame in frames:
        frame = np.array(frame, dtype=np.float)
        avg += (frame/N)
    avg = np.array(np.round(avg), dtype=np.uint8)

    return avg


def difference(frames, interval=1):
    h,w,c = frames[0].shape
    diff = np.zeros((h,w,c), np.float)

    for current_frame, next_frame in zip(frames[::interval], frames[interval::interval]):
        current_frame = np.array(current_frame, dtype=np.float)
        next_frame = np.array(next_frame, dtype=np.float)
        diff += np.absolute(next_frame - current_frame)
    diff = np.array(np.round(diff), dtype=np.uint8)

    return diff