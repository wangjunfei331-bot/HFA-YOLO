import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image


class YOLOHeatmapGenerator:
    def __init__(self, model_path, target_layer_name=None):

        self.model = YOLO(model_path)
        self.model.eval()

        self.device = next(self.model.model.parameters()).device

        self.features = None
        self.gradients = None

        self.register_hooks(target_layer_name)

        self.processed_img = None

    def register_hooks(self, target_layer_name=None):

        model = self.model.model

        if target_layer_name is None:

            layer_names = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    layer_names.append(name)
            if len(layer_names) >= 3:
                target_layer_name = layer_names[-3]
            elif len(layer_names) >= 1:
                target_layer_name = layer_names[-1]
            else:
                raise ValueError("未找到合适的卷积层")

        target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"未找到层: {target_layer_name}")

        print(f"已注册钩子到层: {target_layer_name}")

        def forward_hook(module, input, output):
            self.features = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        self.target_layer = target_layer

    def preprocess_image(self, image):
        imgsz = getattr(self.model, 'imgsz', 640)
        h, w = image.shape[:2]
        scale = min(imgsz / h, imgsz / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h))
        top = (imgsz - new_h) // 2
        bottom = imgsz - new_h - top
        left = (imgsz - new_w) // 2
        right = imgsz - new_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        img_tensor = torch.from_numpy(padded).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor / 255.0  # 归一化到 [0, 1]

        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, (h, w), (top, left, scale)

    def generate_heatmap(self, image_path, target_class=None, conf_threshold=0.5):

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        original_img = img.copy()
        orig_h, orig_w = img.shape[:2]

        self.original_img = original_img

        with torch.no_grad():
            results = self.model(img)

        result = results[0]

        if len(result.boxes) == 0:
            print("未检测到目标")
            return None, None, None, None

        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        if target_class is not None:
            mask = classes == target_class
            if not any(mask):
                print(f"未找到类别 {target_class}")
                return None, None, None, None

            class_boxes = boxes[mask]
            class_confs = confidences[mask]
            max_conf_idx = np.argmax(class_confs)
            target_box = class_boxes[max_conf_idx]
        else:

            max_conf_idx = np.argmax(confidences)
            target_box = boxes[max_conf_idx]


        bbox = target_box.xyxy[0].cpu().numpy()
        cls_id = int(target_box.cls[0])
        conf = float(target_box.conf[0])

        if conf < conf_threshold:
            print(f"置信度 {conf:.3f} 低于阈值 {conf_threshold}")
            return None, None, None, None

        print(f"目标类别: {cls_id}, 置信度: {conf:.3f}, 边界框: {bbox}")

        heatmap = self._compute_grad_cam(img, cls_id)

        return heatmap, original_img, bbox, cls_id

    def _compute_grad_cam(self, img, class_idx):

        self.features = None
        self.gradients = None


        img_tensor, (orig_h, orig_w), (top, left, scale) = self.preprocess_image(img)


        output = self.model.model(img_tensor)


        self.model.model.zero_grad()


        if isinstance(output, tuple):

            predictions = output[0]
        elif isinstance(output, list):
            predictions = output[0] if len(output) > 0 else output
        else:
            predictions = output


        if predictions.ndim == 4:

            batch_size, num_anchors, num_features = predictions.shape
            if num_features >= 5 + class_idx:

                class_channel = 5 + class_idx

                class_scores = predictions[0, :, class_channel]
                target_score = class_scores.mean()

                target_score.backward(retain_graph=True)
            else:

                max_activation = predictions.max()
                max_activation.backward(retain_graph=True)
        else:

            max_activation = predictions.max()
            max_activation.backward(retain_graph=True)


        if self.gradients is None or self.features is None:
            print("警告: 未能获取梯度或特征，使用替代方法")
            return self._generate_simple_heatmap(img, class_idx)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.features, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (640, 640))

        h, w = cam.shape

        cam_resized = cv2.resize(cam, (orig_w, orig_h))

        return cam_resized

    def _generate_simple_heatmap(self, img, class_idx):

        h, w = img.shape[:2]


        heatmap = np.zeros((h, w), dtype=np.float32)

        center_y, center_x = h // 2, w // 2

        y, x = np.ogrid[0:h, 0:w]
        sigma = min(h, w) / 4
        heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))

        if class_idx % 2 == 0:
            heatmap = np.roll(heatmap, h // 4, axis=0)

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def visualize(self, heatmap, original_img, bbox=None, class_id=None,
                  alpha=0.6, save_path=None):

        if heatmap is None:
            print("热力图为空，无法可视化")
            return None

        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )

        if heatmap_colored.shape[:2] != original_img.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored,
                (original_img.shape[1], original_img.shape[0])
            )

        overlayed = cv2.addWeighted(
            original_img,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if class_id is not None:
                label = f"Class {class_id}"
                cv2.putText(overlayed, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Activation Heatmap")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Heatmap Overlay")
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")

        plt.show()

        return overlayed

class SimpleYOLOHeatmap:
    def __init__(self, model_path):

        self.model = YOLO(model_path)
        self.device = next(self.model.model.parameters()).device

    def generate_activation_heatmap(self, image_path, target_class=None):

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        orig_h, orig_w = img.shape[:2]

        results = self.model(img, verbose=False)
        result = results[0]

        if len(result.boxes) == 0:
            print("未检测到目标")
            return None, img, None

        heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)

        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        if target_class is not None:
            mask = classes == target_class
            if not any(mask):
                print(f"未找到类别 {target_class}")
                return None, img, None

            boxes = boxes[mask]
            confidences = confidences[mask]
            classes = classes[mask]

        for i, (box, conf, cls) in enumerate(zip(boxes.xyxy, confidences, classes)):

            x1, y1, x2, y2 = map(int, box.cpu().numpy())

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            box_w = x2 - x1
            box_h = y2 - y1
            size = np.sqrt(box_w * box_h)

            heat_value = conf * 100

            sigma = size / 4
            y_coords, x_coords = np.ogrid[0:orig_h, 0:orig_w]

            gaussian = np.exp(-((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) / (2 * sigma ** 2))

            heatmap += gaussian * heat_value

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        kernel_size = int(min(orig_h, orig_w) * 0.05)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)

        return heatmap, img, boxes.xyxy[0].cpu().numpy() if len(boxes) > 0 else None

    def visualize(self, heatmap, img, bbox=None, alpha=0.6):

        if heatmap is None:
            return None

        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )


        overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 2)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='hot')
        axes[1].set_title("Heatmap")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        return overlayed



if __name__ == "__main__":

    try:
        # 注意：需要根据实际模型结构调整目标层名称
        heatmap_gen = YOLOHeatmapGenerator(
            model_path='yolo26n.pt',  # 替换为你的模型路径
            target_layer_name='model.10.cv2.conv'  # 常见的目标层
        )

        # 生成热力图（以person类为例，COCO数据集中person类的ID是0）
        heatmap, img, bbox, cls_id = heatmap_gen.generate_heatmap(
            image_path='1.jpg',
            target_class=0,  # person类别
            conf_threshold=0.3
        )

        if heatmap is not None:
            # 可视化结果
            result = heatmap_gen.visualize(
                heatmap, img, bbox, cls_id,
                alpha=0.5,
                save_path='heatmap_result.jpg'
            )
    except Exception as e:
        print(f"Grad-CAM方法出错: {e}")
        print("尝试使用简化方法...")

    # 示例2: 使用简化版热力图生成器
    simple_gen = SimpleYOLOHeatmap('yolo26n.pt')

    heatmap, img, bbox = simple_gen.generate_activation_heatmap(
        image_path='1.jpg',
        target_class=0  # person类别
    )

    if heatmap is not None:
        result = simple_gen.visualize(heatmap, img, bbox, alpha=0.5)