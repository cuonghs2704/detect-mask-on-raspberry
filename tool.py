import torch
import torchvision
import cv2
import random
import numpy as np
def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
    total_bboxes, output_bboxes  = [], []
    # 
    N, C, H, W = preds.shape
    bboxes = torch.zeros((N, H, W, 6))
    pred = preds.permute(0, 2, 3, 1)
    # 
    pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
    # 
    preg = pred[:, :, :, 1:5]
    # 
    pcls = pred[:, :, :, 5:]

    # 
    bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
    bboxes[..., 5] = pcls.argmax(dim=-1)

    # 
    gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)])
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid() 
    bcx = (preg[..., 0].tanh() + gx.to(device)) / W
    bcy = (preg[..., 1].tanh() + gy.to(device)) / H

    # 
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H*W, 6)
    total_bboxes.append(bboxes)
        
    batch_bboxes = torch.cat(total_bboxes, 1)

    # 
    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        # 
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # Torchvision NMS
        if len(b) > 0:
            b = torch.Tensor(b).to(device)
            c = torch.Tensor(c).squeeze(1).to(device)
            s = torch.Tensor(s).squeeze(1).to(device)
            keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes

def normalize_image(ori_image, W, H):
    res_img = cv2.resize(ori_image, (W, H), interpolation = cv2.INTER_LINEAR) 
    img = res_img.reshape(1, W, H, 3)
    img = img.transpose(0, 3, 1, 2)
    im = img.astype(np.float32)
    im /= 255
 
    return im



    
def put_box(ori_image, output, LABEL_NAMES):
    for box in output[0]:
       
        box = box.tolist()
        H, W, _ = ori_image.shape
        
        obj_score = box[4]
        
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(ori_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_image, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
        cv2.putText(ori_image, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        cv2.imwrite("result.png", ori_image)
    return ori_image
def letterbox(im, new_shape=(352, 352), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)
def predict(ori_image, W, H, inname, outname, session):
  
    im = normalize_image(ori_image, W, H)
    inp = {inname[0]:im}

    #predict output
    outputs = session.run(outname, inp)[0]
    
    outputs = torch.tensor(outputs) #output shape = 1, 8, 22, 22

    #
    outputs = handle_preds(outputs, 'cpu', 0.65)
    LABEL_NAMES = ['with_mask' , 'without_mask','mask_weared_incorrect']

    preds_image = put_box(ori_image, outputs, LABEL_NAMES)
    return preds_image
def normalize_image4yolo(img):
    

    image = img.copy()
    
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    return im
def predict4yolo(img, W, H, inname, outname, session):
    ori_images = [img.copy()]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image, ratio, dwdh = letterbox(img, (W, H), auto=False)
    im = normalize_image4yolo(image)
    inp = {inname[0]:im}

    # ONNX inference
    outputs = session.run(outname, inp)[0]
    
    
    names =  ['with_mask' , 'without_mask','mask_weared_incorrect']
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

    cv2.imwrite('result_yolo.png', ori_images[0])
    return ori_images[0]