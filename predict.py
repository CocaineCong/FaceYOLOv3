import time
from utils.datasets import *
from utils.utils import *
from utils.parse_config import parse_data_cfg
from yolov3 import Yolov3
from utils.torch_utils import select_device


def process_data(img, img_size=416):  # 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0
    return img


def refine_hand_bbox(bbox, img_shape):
    height, width, _ = img_shape
    x1, y1, x2, y2 = bbox
    expand_w = (x2 - x1)
    expand_h = (y2 - y1)
    x1 -= expand_w * 0.06
    y1 -= expand_h * 0.1
    x2 += expand_w * 0.06
    y2 += expand_h * 0.1
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(x2, width - 1))
    y2 = int(min(y2, height - 1))
    return x1, y1, x2, y2


def detect(ModelPath, cfg, data_cfg, ImgSize=416, ConfThres=0.5, NMSThres=0.5, VideoPath=0):
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    num_classes = len(classes)
    # 初始化模型
    weights = ModelPath
    A_Scalse = 416. / ImgSize
    anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
    anchors_new = [(int(anchors[j][0] / A_Scalse), int(anchors[j][1] / A_Scalse)) for j in range(len(anchors))]
    model = Yolov3(num_classes, anchors=anchors_new)
    device = select_device()  # 运行硬件选择
    use_cuda = torch.cuda.is_available()
    # Load weights
    if os.access(weights, os.F_OK):  # 判断模型文件是否存在
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        print('error model not exists')
        return False
    model.to(device).eval()  # 模型模式设置为 eval
    colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, num_classes + 1)][::-1]
    video_capture = cv2.VideoCapture(VideoPath)
    video_writer = None
    loc_time = time.localtime()
    str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
    save_video_path = "./demo/demo_{}.mp4".format(str_time)
    # -------------------------------------------------
    while True:
        ret, im0 = video_capture.read()
        if ret:
            t = time.time()
            # im0 = cv2.imread("picture/1.png")
            img = process_data(im0, ImgSize)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.time()
            # print("process time:", t1 - t)
            img = torch.from_numpy(img).unsqueeze(0).to(device)

            pred, _ = model(img)  # 图片检测
            if use_cuda:
                torch.cuda.synchronize()
            t2 = time.time()
            # print("inference time:", t2 - t1)
            detections = non_max_suppression(pred, ConfThres, NMSThres)[0]  # nms
            if use_cuda:
                torch.cuda.synchronize()
            t3 = time.time()
            # print("get res time:", t3 - t2)
            if detections is None or len(detections) == 0:
                cv2.namedWindow('image', 0)
                cv2.imshow("image", im0)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                continue
            # Rescale boxes from 416 to true image size
            detections[:, :4] = scale_coords(ImgSize, detections[:, :4], im0.shape).round()
            result = []
            for res in detections:
                result.append(
                    (classes[int(res[-1])], float(res[4]), [int(res[0]), int(res[1]), int(res[2]), int(res[3])]))
            if use_cuda:
                torch.cuda.synchronize()
            for r in result:
                print(r)
            for *xyxy, conf, cls_conf, cls in detections:
                label = '%s %.2f' % (classes[int(cls)], conf)
                xyxy = int(xyxy[0]), int(xyxy[1]) + 6, int(xyxy[2]), int(xyxy[3])
                if int(cls) == 0:
                    plot_one_box(xyxy, im0, label=label, color=(255, 255, 95), line_thickness=3)
                else:
                    plot_one_box(xyxy, im0, label=label, color=(15, 155, 255), line_thickness=3)
            s2 = time.time()
            # print("detect time: {} \n".format(s2 - t))
            str_fps = ("{:.2f} FPS".format(1. / (s2 - t + 0.00001)))
            cv2.putText(im0, str_fps, (5, im0.shape[0] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 4)
            cv2.putText(im0, str_fps, (5, im0.shape[0] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)
            cv2.namedWindow('image', 0)
            cv2.imshow("image", im0)
            key = cv2.waitKey(1)
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps=25, frameSize=(im0.shape[1], im0.shape[0]))
            video_writer.write(im0)
            if key == 27:
                break
        else:
            break
    cv2.destroyAllWindows()
    video_writer.release()


if __name__ == '__main__':
    FaceConfig = 'cfg/face.data'     # 模型相关配置文件
    ModelPath = './weights/face.pt'  # 检测模型路径
    ModelCfg = 'yolo'                # 模型结构
    VideoPath = "./video/face2.mov"  # 测试视频

    ImgSize = 416  # 图像尺寸
    ConfThres = 0.5  # 检测置信度
    NMSThres = 0.6  # nms 阈值

    with torch.no_grad():  # 设置无梯度运行模型推理
        detect(
            ModelPath=ModelPath,
            cfg=ModelCfg,
            data_cfg=FaceConfig,
            ImgSize=ImgSize,
            ConfThres=ConfThres,
            NMSThres=NMSThres,
            # VideoPath = VideoPath,
        )
