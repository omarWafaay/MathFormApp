import cv2
import torch
import numpy as np
import time
import torchvision


def initialize_model(weights):
    '''
    Description:
        Initialize the torch script and load it to an object
    Parameters:
    Inputs:
        weights : (String) directory path for the model
    Output:
        model : (torch Object) model object loaded in the memory
    '''
    model = torch.jit.load(weights)
    return model

def xywh2xyxy(x):
    '''
    Description:
        Transform the representation from center-x, center-y, width, height to top-left-x top-left-y bottom-right-x bottom-right-y
    Parameters:
    Inputs:
        x: (list) contain the input representation of center width height
    Outputs:
        y: (list) contain the output representation of top-left and bottom-right points
    '''
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    '''
    Description:
        Apply the Non-Maximum Suppression (NMS) on the resultant prediction while thresholding on the IOU and Conf
        it takes the resultant predictions and returns the merged and filtered predictions 
    Parameters:
    Inputs:
        prediction: (nd-array) array containing the predictions of the model 
        conf_thres: (float) Confidence threshold to filter the predictions
        iou_thres: (float) Merging IOU threshold any predictions overlapped on this ratio will be merged else will be considered
                   as separate prediction
        max_det: (int) maximum number of detections that will be taken into consideration
    Outputs:
        output: (nd-array) array containing the predictions after being filtered and considered as final predicitons
    '''

    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def clip_coords(boxes, shape):
    '''
    Description:
        clip the boundary boxes to the image shape to make sure that points inside the given image shape
    Parameters:
    Inputs:
        boxes: (list) containing the coordinates of each box
        shape: (tuple) image shape (height,width,channels)
    Outputs:
        boxes: (list) boxes after being clipped
    '''
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    return boxes

def scale_coords(img1_shape, coords, img0_shape):
    '''
    Description:
        scale the coordinates of the img1 to fit the same area in img0
    Parameters:
    Inputs:
        img1_shape: (tuple) containing the height, width, channels of the image1
        coords: (list) contain the 2 points (x,y) in the form of 1x4 to be scaled
        img0_shape: (tuple) containing the height, width, channels of the image0 that will be considered as targeted shape
    Outputs:
        scaled_coords: (list) same shape as input coords but scaled on image0
    '''

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    scaled_coords = clip_coords(coords, img0_shape)
    return scaled_coords

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    '''
    Description:
        Resize the given image on the minimum ratio of the original dimension and the given new shape while keeping the aspect ratio
        of the image and pad the sides to match the given new shape dimension
        It takes the image and the new size and returns resized padded image, ratio of scaling and padded width and height
    Parameters:
    Inputs:
        im: (nd-array) image in the shape of numpy array
        new_shape: (int tuple) containing the needed shape of height and width default value is 640,640
        color: (tuple) the color of the padding for the 3 channels RGB
    '''
    shape = im.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def predict_formulas(img, model):
    '''
    Description:
        Pipeline to detect the math formulas inside given image
        It takes an image containing one or more math formulas and return the bounding boxes of thr
    ''' 
    device = torch.device('cpu')
    mod_img, scale_ratio, pad_wh = letterbox(img,new_shape = (640,640))
    mod_img = mod_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x460x460
    mod_img = np.ascontiguousarray(mod_img)
    mod_img = torch.from_numpy(mod_img).to(device)
    mod_img = mod_img.float() # float32
    mod_img /= 255.0  

    if mod_img.ndimension() == 3:
        mod_img = mod_img.unsqueeze(0)

    # Inference model
    pred = model(mod_img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, 0.5, 0.45)
    
    bboxes = []
    
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(mod_img.shape[2:], det[:, :4], img.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if conf < 0.5 or int(cls.item()) != 0:
                    continue
                bboxes.append([xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item(),conf.item(), cls.item()])
    return bboxes


