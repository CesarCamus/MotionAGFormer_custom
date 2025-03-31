import numpy as np
import logging as logger
import time
from datetime import datetime
import argparse
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.gridspec as gridspec
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import torch.nn as nn
from tritonclient.http import InferenceServerClient
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from triton_backend.utils import normalize_screen_coordinates, camera_to_world, flip_data, turn_into_clips, wrap, qrot, resample
import cv2 
import json
import matplotlib.pyplot as plt
import tqdm
import copy
import glob

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)

class Pose3DTritonInferencer:
    """
    This class is used to perform 3D pose estimation using the triton model. The class is initialized with the model name, model version, batch size, fixed input size, confidence threshold, model type and search region ratio. The class has methods to process the 2D keypoints sequence, prepare the input, perform inference, process the output, draw the pose, draw the heatmap, draw all and get the input details and output details. The class has a method to load the triton client.
    """
    def __init__(
            self,
            model_name='motion_former_3d_pose', 
            model_version='2', 
            batch_size=1,
            fixed_input_size=(243,17,3),
            conf_thres = 0.35,
            model_type = '',
            search_region_ratio = 0.1) -> None:
        
        self.conf_threshold = conf_thres
        self.model_type = model_type
        self.search_region_ratio = search_region_ratio

        self.inputs = []
        self.outputs = []

        self.model_name = model_name
        self.model_version = model_version 
        
        INPUT_NAMES = ["input"]
        # OUTPUT_NAMES = ["det_indices", "det_boxes", "det_pose", "det_scores"]
        OUTPUT_NAMES = ['output']
        input_size = (batch_size, *fixed_input_size)
        print("input size : ",input_size)
        self.input_size = input_size  
        self.INPUT_NAMES = INPUT_NAMES  
        self.OUTPUT_NAMES = OUTPUT_NAMES 

        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        # self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        # self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        # self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))
        
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        self.triton_client = self.load_triton()

    def load_triton(self):
        print("Initializing Detector TRT Client..")
        triton_client = None

        # Attempt to create a Triton client
        for attempt in range(2000):
            try:
                triton_client = grpcclient.InferenceServerClient(
                    url='localhost:8001',
                    verbose=False,
                    ssl=False,
                )
                print("Triton Client initialized successfully..")
                break
            except Exception as e:
                logger.exception(f"Exception creating Triton Client: {e}")
                time.sleep(30)
        else:
            logger.error("Failed to initialize Triton Client after 2000 attempts.")
            return None

        for attempt in range(5000):
            try:
                if not triton_client.is_server_live():
                    logger.exception("FAILED : is_server_live")
                    time.sleep(20)
                else:
                    print("Triton Server is live.")
                    break
            except Exception as e:
                logger.exception(f"Exception checking if Triton Server is live: {e}")
                time.sleep(20)
        else:
            logger.error("Triton Server is not live after 5000 attempts.")
            return None
        
        for attempt in range(5000):
            try:
                if not triton_client.is_server_ready():
                    logger.exception("FAILED : is_server_ready")
                    time.sleep(20)
                else:
                    print("Triton Server is ready.")
                    break
            except Exception as e:
                logger.exception(f"Exception checking if Triton Server is ready: {e}")
                time.sleep(20)
        else:
            logger.error("Triton Server is not ready after 5000 attempts.")
            return None
        print("Triton Client connected successfully..")

        try:
            model_metadata = triton_client.get_model_metadata(model_name=self.model_name)
            print(f"Model Metadata: {model_metadata}")
        except InferenceServerException as e:
            logger.error(f"Failed to retrieve model metadata: {e}")
            return None

        return triton_client
    

    def __call__(self, image, detections=None):

        if detections is None:
            return self.update(image)
        else:
            return self.update_with_detections(image, detections)

    def show3Dpose(self,vals, ax):
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


    @torch.no_grad()
    def get_pose3D(self,video_path, output_dir,keypoints):
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
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        
        img_size = img.shape
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
        
        print('\nGenerating 3D pose...')
        for idx, clip in enumerate(clips):
            input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
            input_2D_aug = flip_data(input_2D)
            
            input_2D = torch.from_numpy(input_2D.astype('float32'))#.cuda()
            input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32'))#.cuda()

            # output_3D_non_flip = model(input_2D) 
            # output_3D_flip = flip_data(model(input_2D_aug))
            output_3D_non_flip = self.inference(input_2D.numpy())
            output_3D_flip = flip_data(self.inference(input_2D_aug.numpy()))
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

                

                fig = plt.figure(figsize=(9.6, 5.4))
                gs = gridspec.GridSpec(1, 1)
                gs.update(wspace=-0.00, hspace=0.05) 
                ax = plt.subplot(gs[0], projection='3d')
                self.show3Dpose(post_out, ax)
                #print('post_out shape:', post_out.shape)
                output_dir_3D = output_dir +'/pose3D/'
                os.makedirs(output_dir_3D, exist_ok=True)
                str(('%04d'% (idx * 243 + j)))
                plt.savefig(output_dir_3D + str(('%04d'% (idx * 243 + j))) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
                plt.close(fig)
            

           
        print('Generating 3D pose successful!')

        ## all
        #image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
        image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

        print('\nGenerating demo...')
        for i in tqdm.tqdm(range(len(image_3d_dir))):
            #image_2d = plt.imread(image_2d_dir[i])
            image_3d = plt.imread(image_3d_dir[i])

            ## crop
            #edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
            #image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

            edge = 130
            image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

            ## show
            font_size = 12
            fig = plt.figure(figsize=(15.0, 5.4))
            ax = plt.subplot(121)
            #showimage(ax, image_2d)
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


    def process_2D_seq(self,keypoints,video_path,output_dir):

        clips, downsample = turn_into_clips(keypoints)
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        if img is not None:
            img_size = img.shape
        cap.release()

        video_name = os.path.basename(video_path)

        all_frames = []

        for idx, clip in enumerate(clips):
            batch_input = []


            input_2D,input_2D_aug = self.prepare_input(clip,img_size)
            
            # batch_input.append(input_2D.squeeze(0))       # Original
            # batch_input.append(input_2D_aug.squeeze(0))   # Augmented

            output_3D_non_flip = self.inference(input_2D) 
            output_3D_flip = flip_data(self.inference(input_2D_aug))
            output_3D = (output_3D_non_flip + output_3D_flip) / 2

            # # Convert list to a single stacked NumPy array (batch_size = 2 * len(clips))
            # batch_input = np.stack(batch_input)  # Shape: (2 * len(clips), 243, 17, 3)

            # # Perform batch inference in one go
            # output_3D = self.inference(batch_input)  # Shape: (2 * len(clips), ..., ...)

            # # Split outputs back into original and augmented versions
            # output_3D_non_flip = output_3D[0::2]  # Take even indices (original inputs)
            # output_3D_flip = flip_data(output_3D[1::2]) 

            # #output_3D_non_flip = self.inference(input_2D) 
            # #output_3D_flip = flip_data(self.inference(input_2D_aug))
            # output_3D = (output_3D_non_flip + output_3D_flip) / 2

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
        

    def prepare_input(self, clip,img_size):

        

        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)

        #input_2D = torch.from_numpy(input_2D.astype('float32'))
        #input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32'))

        return input_2D, input_2D_aug

    def inference(self, input_tensor):
        start = time.perf_counter()
        
        #(batch_size,T,nb_keypoints,nb_coord) = self.input_size
        
       
        self.inputs = [grpcclient.InferInput(self.INPUT_NAMES[0], list(self.input_size), "FP32")]
        
        self.inputs[0].set_data_from_numpy(input_tensor)

        results = self.triton_client.infer(model_name=self.model_name,inputs=self.inputs,outputs=self.outputs,model_version=self.model_version)
        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        outputs = results.as_numpy("output")  
        return outputs

    def process_output(self, heatmaps):
        total_heatmap = cv2.resize(heatmaps.sum(axis=1)[0], (self.img_width, self.img_height))
        map_h, map_w = heatmaps.shape[2:]

        # Find the maximum value in each of the heatmaps and its location
        max_vals = np.array([np.max(heatmap) for heatmap in heatmaps[0, ...]])
        peaks = np.array([np.unravel_index(heatmap.argmax(), heatmap.shape)
                          for heatmap in heatmaps[0, ...]])
        peaks[max_vals < self.conf_threshold] = np.array([np.NaN, np.NaN])

        # Scale peaks to the image size
        peaks = peaks[:, ::-1] * np.array([self.img_width / map_w,
                                          self.img_height / map_h])

        return total_heatmap, peaks

    def draw_pose(self, image):

        if self.poses is None:
            return image
        return draw_skeletons(image, self.poses, self.model_type)

    def draw_heatmap(self, image, mask_alpha=0.4):
        if self.poses is None:
            return image
        return draw_heatmap(image, self.total_heatmap, mask_alpha)

    def draw_all(self, image, mask_alpha=0.4):
        if self.poses is None:
            return image
        return self.draw_pose(self.draw_heatmap(image, mask_alpha))

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

