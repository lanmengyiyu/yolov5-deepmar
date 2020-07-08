import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
from PIL import Image, ImageDraw
from deepmar.baseline.model.DeepMAR import DeepMAR_ResNet50

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print(os.path.isfile(weights))
    # Load model
 #   google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                person_count = 0
                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        if int(cls)==0: #change by miao   plot only person in image
                            #label = '%s %.2f' % (names[int(cls)], conf)
                            label = '%.2f' % (conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            # print(im0.shape)
                            bbox_img = im0[y1:y2,x1:x2]
                            # print(type(bbox_img))
                            shortname, extension = os.path.splitext(Path(p).name)
                            person_count += 1
                            bbox_img_name = shortname + '_person%d'%(person_count) + extension
                            # print(bbox_img_name)
                            # bbox_img_path = str(Path(out) / Path(bbox_img_name))
                            # cv2.imwrite(bbox_img_path, bbox_img)
                            # bbox_img_PIL = transforms.ToPILImage()(bbox_img)
                            # bbox_img_draw = ImageDraw.Draw(bbox_img_PIL)
                            # bbox_img_PIL = Image.open(bbox_img_path)
                            # img_trans = input_transform(bbox_img_PIL)
                            # img_trans = torch.unsqueeze(img_trans, dim=0)
                            # img_var = Variable(img_trans).cuda()
                            #
                            # score = model_deepmar(img_var).data.cpu().numpy()
                            # print(score)

                            bbox_img_convt_PIL = Image.fromarray(cv2.cvtColor(bbox_img,cv2.COLOR_BGR2RGB))
                            img_trans_conv_PIL = input_transform(bbox_img_convt_PIL)
                            img_trans_conv_PIL = torch.unsqueeze(img_trans_conv_PIL, dim=0)
                            img_trans_conv_PIL_var = Variable(img_trans_conv_PIL).cuda()
                            score = model_deepmar(img_trans_conv_PIL_var).data.cpu().numpy()
                            # print(score)
                            # result_att = []
                            # print(bbox_img_path+" done attribution recognition")
                            att_count = -1
                            for idx in range(len(att_list)):
                                if score[0, idx] >= 0:
                                    txt = '%s: %.2f' % (att_list[idx], score[0, idx])
                                    # print(txt)
                                    att_count += 1
                                    # print(att_count)
                                    cv2.putText(im0, att_list[idx], (x1, y1 + 5 + 10*att_count ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, [225, 255, 255],
                                                thickness=1, lineType=cv2.LINE_AA)
                                    # cv2.imshow('deepmar',im0)
                                    # cv2.waitKey()
                                    # cv2.destroyAllWindows()
                                    # result_att.append(txt)
                                    # print(txt)
                            # print(att_list)
                            # print('-'*30)
                            # bbox_img_PIL.save(bbox_img_savepath)
                # cv2.imshow('deepmar',im0)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)
    resize = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    with open('./deepmar/attribute_list.pickle', 'rb') as file_handle:
        att_list = pickle.load(file_handle)
#     model_deepmar = torch.load('./deepmar/deepmar.pth')
    model_deepmar = DeepMAR_ResNet50()
    model_deepmar.load_state_dict(torch.load('deepmar/deepmar_statedict.pth'))
    model_deepmar.eval()
    model_deepmar.cuda()

    with torch.no_grad():
        detect()

        # Update all models
        # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt.weights, opt.weights)
