import sys
import argparse
import cv2
import os 

sys.path.append(os.getcwd())

from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
from time import time
from lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer
import onnxruntime as ort
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.gridspec as gridspec
#import triton_backend.trt_client as triton_client

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

"""
Keypoints corespondance list : 

0: central hip
1: right hip
2: right knee
3: right ankle
4: left hip
5: left knee
6: left ankle
7: spine
8: neckbase
9: nose
10: forehead
11: left shoulder
12: left elbow
13: left wrist
14: right shoulder
15: right elbow
16: right wrist
"""


def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def get_pose2D(video_path, output_dir):
    #cap = cv2.VideoCapture(video_path)
    #width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    #print('keypoints_before : ',keypoints.shape)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    #print('keypoints_after :',keypoints.shape)
    #print('traj : ',traj)
    # plt.plot([i[0] for i in traj], [i[1] for i in traj])
    # plt.savefig(output_dir + 'traj.png')
    # plt.show()
    
    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir += '/input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_name = video_path.split('/')[-1].split('.')[0]
    names = sorted(glob.glob(os.path.join(output_dir + '/pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + '/'+video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample

def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]

    return new_keypoints


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.copy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data

def run_motionagformer_onnx(input_array,onnx_session,onnx_input_name):
    """
    Runs ONNX inference for MotionAGFormer.

    Args:
        input_array (np.ndarray): Input of shape (B, T, J, C), dtype float32

    Returns:
        np.ndarray: Output of shape (B, T, J, 3)
    """
    if input_array.dtype != np.float32:
        input_array = input_array.astype(np.float32)

    return onnx_session.run(None, {onnx_input_name: input_array})[0]

@torch.no_grad()
def get_pose3D(video_path, output_dir,keypoints,model):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    # ## Reload 
    # model = nn.DataParallel(MotionAGFormer(**args))#.cuda()

    # # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    # model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]

    # pre_dict = torch.load(model_path,map_location=torch.device('cpu'))
    # model.load_state_dict(pre_dict['model'], strict=True)

    # model.eval()

    onnx_model_path = os.path.join("checkpoint", "motionagformer.onnx")
    onnx_session = ort.InferenceSession(onnx_model_path)

    # Get input name for ONNX runtime
    onnx_input_name = onnx_session.get_inputs()[0].name

# Prepare inference function


    ## input
    #keypoints = np.load(output_dir + '/input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    # keypoints = np.load('demo/lakeside3.npy')
    # keypoints = keypoints[:240]
    # keypoints = keypoints[None, ...]
    # keypoints = turn_into_h36m(keypoints)
    

    clips, downsample = turn_into_clips(keypoints)


    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_valid_frames = keypoints.shape[1]
    print('video_length:', video_length)
    print('num_valid_frames:', num_valid_frames)
    ## 3D
    print('\nGenerating 2D pose image...')
    for i in tqdm(range(num_valid_frames)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape
        img_size = (2160,3840)
        input_2D = keypoints[0][i]

        image = show2Dpose(input_2D, copy.deepcopy(img))

        output_dir_2D = output_dir +'/pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)

    all_frames = []
    print('\nGenerating 3D pose...')
    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)
        
        input_2D = torch.from_numpy(input_2D.astype('float32'))#.cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32'))#.cuda()

        # output_3D_non_flip = model(input_2D) 
        # output_3D_flip = flip_data(model(input_2D_aug))
        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0]#.cpu().detach().numpy()
        #print('post_out_all shape:', post_out_all.shape)

        #np.save(output_dir + 'post_out_all_0.5.npy', post_out_all)

        for j, post_out in enumerate(post_out_all):
            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            frame_data = {
                "frame_index": j + idx * post_out_all.shape[0],
                "keypoints": {
                    str(i): {
                        "x": float(post_out[i][0]),
                        "y": float(post_out[i][1]),
                        "z": float(post_out[i][2])
                    }
                    for i in range(post_out.shape[0])
                }
            }
            all_frames.append(frame_data)

            # Build final JSON structure
            

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)
            #print('post_out shape:', post_out.shape)
            output_dir_3D = output_dir +'/pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            str(('%04d'% (idx * 243 + j)))
            plt.savefig(output_dir_3D + str(('%04d'% (idx * 243 + j))) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)
            os.makedirs(output_dir, exist_ok=True)

        json_output = {
                #"video_name": video_name,
                #"label": label,
                "frames": all_frames
            }
        json_path = os.path.join(output_dir, 'keypoints_xyz.json')
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=4)

        print(f"Saved keypoints to {json_path}")
        return json_path    

        
    print('Generating 3D pose successful!')

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)
        fig.patch.set_facecolor('lightgrey')
        ## save
        output_dir_pose = output_dir +'/pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close(fig)

    return json_path

import json
import os

@torch.no_grad()
def get_3D_keypoints(output_dir, model, keypoints, model_path=None, args=None,label=None,video_path=None):
    clips, downsample = turn_into_clips(keypoints)
    #cap = cv2.VideoCapture(video_path)
    #ret, img = cap.read()
    #if img is not None:
    img_size = (384, 288)
    #cap.release()
    print('\nGenerating 3D pose...')
    
    #video_name = os.path.basename(video_path)

    all_frames = []

    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)

        input_2D = torch.from_numpy(input_2D.astype('float32'))
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32'))

        output_3D_non_flip = model(input_2D) 
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0]

        rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        
        for j in range(post_out_all.shape[0]):
            keypoints_frame = camera_to_world(post_out_all[j], R=rot, t=0)
            keypoints_frame[:, 2] -= np.min(keypoints_frame[:, 2])
            max_value = np.max(keypoints_frame)
            keypoints_frame /= max_value

            frame_data = {
                "frame_index": j + idx * post_out_all.shape[0],
                "keypoints": {
                    str(i): {
                        "x": float(keypoints_frame[i][0]),
                        "y": float(keypoints_frame[i][1]),
                        "z": float(keypoints_frame[i][2])
                    }
                    for i in range(keypoints_frame.shape[0])
                }
            }
            all_frames.append(frame_data)

    # Build final JSON structure
    json_output = {
        #"video_name": video_name,
        #"label": label,
        "frames": all_frames
    }

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'keypoints_xyz.json')
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=4)

    print(f"Saved keypoints to {json_path}")

def get_3D_keypoints_all(directory,show_anim=False):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    ## Reload 
    model = nn.DataParallel(MotionAGFormer(**args))#.cuda()

    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]

    pre_dict = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()
    counter = 0
    output_dir_main = './demo/processed_videos'
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files):
            if not file.endswith('.mp4'):
                continue
            print(f"Processing {file}")
            # if 'take-off_86' in file:
            #     print(f"Skipping {file} as it contains 'take_off_86'.")
            #     continue
            #label = file.split('_')[0]
            output_dir = os.path.join(root)
            if os.path.exists(os.path.join(output_dir, 'keypoints_xyz.json')):
                print(f"Skipping {file} as keypoints_xyz.json already exists.")
                continue
            start_time = time()
            
            
            video_path = os.path.join(root, file)
            print("video path : ",video_path)
            get_pose2D(video_path, output_dir)
            if show_anim:
                get_pose3D(video_path, output_dir)
                img2video(video_path, output_dir)
                print('Generating demo successful!')
            keypoints = np.load(output_dir + '/input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
            #get_3D_keypoints(video_path, output_dir, model, model_path,args,keypoints)
            get_3D_keypoints(output_dir,model,keypoints)
            print(f"Processing {file} took {time() - start_time:.2f} seconds")
            # counter += 1
            # if counter == 1:
            #     return

    # raw_maneuvers_data_dir = './raw_maneuvers_data'
    # all_filenames = get_all_filenames(raw_maneuvers_data_dir)
    # print(all_filenames)
    ## input
    
def get_all_filenames(directory):
        filenames = []
        dirs_all = []
        for root, dirs, files in os.walk(directory):
            dirs_all.append(root)
            for file in files:
                filenames.append(os.path.join(root, file))
        return dirs_all

def rename_files_in_subfolders(directory):
    for root, dirs, files in os.walk(directory):
        files.sort()  # Ensure files are sorted to avoid jumps in IDs
        id = 0
        for file in files:
            subfolder_name = os.path.basename(root)
            file_extension = os.path.splitext(file)[1]
            new_name = f"{subfolder_name}_{id}{file_extension}"
            
            # Check if the file is already named correctly
            if file == new_name:
                id += 1
                continue
            
            # Check if the new name is already taken
            while os.path.exists(os.path.join(root, new_name)):
                id += 1
                new_name = f"{subfolder_name}_{id}{file_extension}"
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
            id += 1

def load_3d_keypoints_from_json(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        frames = data.get('threed_pose', data.get('frames', []))
       #frames = data_3D['frames']
        T = len(frames)
        num_joints = 17

        keypoints = np.zeros((T, num_joints, 3), dtype=np.float32)

        for t, frame in enumerate(frames):
            kp_dict = frame['keypoints']
            for j in range(num_joints):
                kp = kp_dict[str(j)]
                keypoints[t, j] = np.array([kp['x'], kp['y'], kp['z']], dtype=np.float32)

        return keypoints

# def load_2d_keypoints_from_json(json_path):
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#         frames = data['twod_pose']
#         print(frames)
#        #frames = data_3D['frames']
#         T = len(frames)
#         num_joints = 17

#         keypoints = np.zeros((T, num_joints, 3), dtype=np.float32)

#         for t, frame in enumerate(frames):
#             print(frame)
#             kp_dict = frame['keypoints']
#             for j in range(num_joints):
#                 kp = kp_dict[str(j)]
#                 keypoints[t, j] = kp

#         return keypoints

def load_2d_keypoints_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['twod_pose']
    frame_ids = sorted(frames.keys(), key=int)
    T = len(frame_ids)
    num_joints = 17

    keypoints = np.zeros((T, num_joints, 3), dtype=np.float32)

    for t, frame_id in enumerate(frame_ids):
        kp_list = frames[frame_id]['keypoints']
        scores = frames[frame_id]['score']
        for j in range(min(len(kp_list), num_joints)):
            keypoints[t, j, :2] = np.array(kp_list[j][:2], dtype=np.float32)
            keypoints[t, j, 2] = scores[j]

    return keypoints

def vis_kpts_3d(kpts_3d,output_dir,idx):
    fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05) 
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose(kpts_3d, ax)
    #print('post_out shape:', post_out.shape)
    output_dir_3D = output_dir +'/pose3D/'
    os.makedirs(output_dir_3D, exist_ok=True)
    #str(('%04d'% (idx * 243 + j)))
    plt.savefig(output_dir_3D + str(('%04d'% (idx))) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
    plt.close(fig)

def vis_kpts_video(json_path):
    keypoints = load_3d_keypoints_from_json(json_path)
    T, J, D = keypoints.shape
    output_dir = os.path.join(os.path.dirname(json_path), os.path.splitext(os.path.basename(json_path))[0])
    os.makedirs(output_dir, exist_ok=True)
    new_json_path = os.path.join(output_dir, os.path.basename(json_path))
    #os.rename(json_path, new_json_path)
    for i in tqdm(range(T)):
        vis_kpts_3d(keypoints[i],output_dir,i)

    print(f"Visualized 3D keypoints and saved to {output_dir}")
    #plt.show()

def prepare_input(clip,img_size):

    
    input_2D = normalize_screen_coordinates(clip, w=img_size[0], h=img_size[1]) 
    input_2D_aug = flip_data(input_2D)

    input_2D = torch.from_numpy(input_2D.astype('float32'))
    input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32'))

    return input_2D, input_2D_aug

@torch.no_grad()
def process_2D_seq(keypoints,video_path,output_dir,model):

        clips, downsample = turn_into_clips(keypoints)
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        if img is not None:
            img_size = img.shape
            img_size = (img.shape[1], img.shape[0])
            print('img_size:', img_size)
            
        cap.release()

        #img_size = (2160,3840)
        #img_size =  (3840, 2160)
        video_name = os.path.basename(video_path)

        all_frames = []

        for idx, clip in enumerate(clips):
            batch_input = []


            input_2D,input_2D_aug = prepare_input(clip,img_size)
            
            batch_input.append(input_2D.squeeze(0))       # Original
            batch_input.append(input_2D_aug.squeeze(0))   # Augmented

            # Convert list to a single stacked NumPy array (batch_size = 2 * len(clips))
            batch_input = np.stack(batch_input)  # Shape: (2 * len(clips), 243, 17, 3)
            batch_input = torch.from_numpy(batch_input).float()  # Convert to tensor
            # Perform batch inference in one go
            output_3D = model(batch_input)  # Shape: (2 * len(clips), ..., ...)

            # Split outputs back into original and augmented versions
            output_3D_non_flip = output_3D[0::2]  # Take even indices (original inputs)
            output_3D_flip = flip_data(output_3D[1::2]) 

            #output_3D_non_flip = self.inference(input_2D) 
            #output_3D_flip = flip_data(self.inference(input_2D_aug))
            output_3D = (output_3D_non_flip + output_3D_flip) / 2

            if idx == len(clips) - 1:
                output_3D = output_3D[:, downsample]

            output_3D[:, :, 0, :] = 0
            post_out_all = output_3D[0]#.cpu().detach().numpy()


            #post_process(post_out_all,video_name)
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            
            for j in range(post_out_all.shape[0]):
                keypoints_frame = camera_to_world(post_out_all[j], R=rot, t=0)
                
                keypoints_frame[:, 2] -= np.min(keypoints_frame[:, 2])
                max_value = np.max(keypoints_frame)
                keypoints_frame /= max_value
                

                frame_data = {
                    "frame_index": j + idx * post_out_all.shape[0],
                    "keypoints": {
                        str(i): {
                            "x": float(keypoints_frame[i][0]),
                            "y": float(keypoints_frame[i][1]),
                            "z": float(keypoints_frame[i][2])
                        }
                        for i in range(keypoints_frame.shape[0])
                    }
                }
                all_frames.append(frame_data)

        # Build final JSON structure
        json_output = {
            "video_name": video_name,
            "label": None,
            "frames": all_frames
        }

        # Save JSON
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'keypoints_xyz.json')
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=4)

        print(f"Saved keypoints to {json_path}")
        return json_path

@torch.no_grad()
def process_2D_seq_curr(keypoints,video_path,output_dir,model):

        clips, downsample = turn_into_clips(keypoints)
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        if img is not None:
            img_size = img.shape
            print('img_size:', img_size)
            
        cap.release()

        img_size = (228,384)

        video_name = os.path.basename(video_path)

        all_frames = []

        for idx, clip in enumerate(clips):
            batch_input = []


            input_2D,input_2D_aug = prepare_input(clip,img_size)
            
            batch_input.append(input_2D.squeeze(0))       # Original
            batch_input.append(input_2D_aug.squeeze(0))   # Augmented

            # Convert list to a single stacked NumPy array (batch_size = 2 * len(clips))
            batch_input = np.stack(batch_input)  # Shape: (2 * len(clips), 243, 17, 3)
            batch_input = torch.from_numpy(batch_input).float()  # Convert to tensor
            # Perform batch inference in one go
            output_3D = model(batch_input)  # Shape: (2 * len(clips), ..., ...)

            # Split outputs back into original and augmented versions
            output_3D_non_flip = output_3D[0::2]  # Take even indices (original inputs)
            output_3D_flip = flip_data(output_3D[1::2]) 

            

            #output_3D_non_flip = self.inference(input_2D) 
            #output_3D_flip = flip_data(self.inference(input_2D_aug))
            output_3D = (output_3D_non_flip + output_3D_flip) / 2

            if idx == len(clips) - 1:
                output_3D = output_3D[:, downsample]

            output_3D[:, :, 0, :] = 0
            post_out_all = output_3D[0]#.cpu().detach().numpy()


            #post_process(post_out_all,video_name)
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            
            for j in range(post_out_all.shape[0]):
                keypoints_frame = camera_to_world(post_out_all[j], R=rot, t=0)
                keypoints_frame[:, 2] -= np.min(keypoints_frame[:, 2])
                max_value = np.max(keypoints_frame)
                keypoints_frame /= max_value

                frame_data = {
                    "frame_index": j + idx * post_out_all.shape[0],
                    "keypoints": {
                        str(i): {
                            "x": float(keypoints_frame[i][0]),
                            "y": float(keypoints_frame[i][1]),
                            "z": float(keypoints_frame[i][2])
                        }
                        for i in range(keypoints_frame.shape[0])
                    }
                }
                all_frames.append(frame_data)

        # Build final JSON structure
        json_output = {
            "video_name": video_name,
            "label": None,
            "frames": all_frames
        }

        # Save JSON
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'keypoints_xyz.json')
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=4)

        print(f"Saved keypoints to {json_path}")
        return json_path

def filter_confident_keypoints(kpts_2d: np.ndarray, threshold: float):
    """
    Filters frames where all keypoints have a confidence score above the given threshold.

    Parameters:
    - kpts_2d (np.ndarray): Input keypoints with shape (1, num_frames, 17, 3)
    - threshold (float): Confidence threshold for filtering

    Returns:
    - valid_ids (List[int]): List of frame indices with all keypoints above threshold
    - filtered_kpts (np.ndarray): Filtered keypoints with shape (N_valid, 17, 3)
    """
    # Remove first singleton dimension: (num_frames, 17, 3)

    # Extract confidence values: shape (num_frames, 17)
    confidences = kpts_2d[..., 2]

    # Identify frames where all keypoints have confidence > threshold
    valid_mask = np.all(confidences > threshold, axis=1)

    # Get indices of valid frames
    valid_ids = np.where(valid_mask)[0].tolist()

    # Filter keypoints for valid frames
    filtered_kpts = kpts_2d[valid_mask]

    return valid_ids, filtered_kpts

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    # parser.add_argument('--gpu', type=str, default='0', help='input video')
    # args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # video_path = './demo/video/' + args.video
    # video_name = video_path.split('/')[-1].split('.')[0]
    # output_dir = './demo/output/' + video_name + '/'

    # get_pose2D(video_path, output_dir)
    # get_pose3D(video_path, output_dir)
    # img2video(video_path, output_dir)
    # print('Generating demo successful!')

    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    ## Reload 
    model = nn.DataParallel(MotionAGFormer(**args))#.cuda()

    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]

    pre_dict = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()
    #video_path = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/demo/video/b_2022-11-05-12-20-48_558.mp4"
    #output_dir = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/demo/processed_videos"
    #keypoints = np.load("/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/demo/output/b_2022-11-05-12-20-48_558/input_2D/keypoints.npz", allow_pickle=True)['reconstruction']

    '''directory = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/3d-based-reid/data/custom_reid/img_experimental_data"
    get_3D_keypoints_all(directory=directory,show_anim=False)'''
    #json_path = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/meta/pdg-left-10-05/2025-05-10-11-07-08_4.json"
    json_path = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/demo/samples_json/2025-05-23-17-11-53_24.json"
    #vis_kpts_video(json_path)
    kpts_2d = load_2d_keypoints_from_json(json_path)

    valid_ids,filtered_kpts_2d = filter_confident_keypoints(kpts_2d, threshold=0.65)
    print('valid_ids nb :', len(valid_ids))
    print('valid_ids:', valid_ids)

    kpts_2d = np.expand_dims(kpts_2d, axis=0)  # Add batch dimension
    print('kpts_2d shape:', kpts_2d.shape)
    video_path = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/demo/video/test_sliding_window.mp4"
    ## Reload 
    output_dir = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/demo/samples_json/processed_360_14"
    #json_path = process_2D_seq(kpts_2d,video_path,output_dir,model)
    json_path = '/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/demo/samples_json/2025-05-28-07-35-11_135.json'


    json_path = '/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionBERT_custom/lib/data/processed_videos/360/360_14/keypoints_xyz.json'
    json_path = '/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/samples_json/2025-06-08-11-28-57_2183.json'
    #json_path = get_pose3D(video_path, output_dir, kpts_2d, model)
    #vis_kpts_video(json_path)
