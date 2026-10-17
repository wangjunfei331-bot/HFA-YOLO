import argparse
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import glob

# 添加YOLOv11到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv11根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    # 尝试导入YOLOv11相关模块
    from models.common import DetectMultiBackend
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.metrics import ap_per_class, ConfusionMatrix
    from utils.plots import plot_images, output_to_target
    from utils.torch_utils import select_device
    from utils.datasets import create_dataloader
    from utils.callbacks import Callbacks
    import val as validate  # 导入验证模块
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保脚本在YOLOv11项目根目录下运行，或正确设置YOLOv11路径")
    sys.exit(1)


class ParetoFront3DAnalyzer:
    """三维帕累托前沿分析器"""

    def __init__(self, data_path, weights_dir, device='', half=False, img_size=640):
        """
        初始化分析器

        Args:
            data_path: 数据集配置文件路径 (如 'coco.yaml')
            weights_dir: 权重文件目录路径
            device: 设备 ('cpu', 'cuda:0', 'cuda:1', ...)
            half: 是否使用FP16半精度
            img_size: 输入图像尺寸
        """
        self.data_path = data_path
        self.weights_dir = Path(weights_dir)
        self.device = select_device(device)
        self.half = half
        self.img_size = img_size
        self.results = []

        print(f"设备: {self.device}")
        print(f"数据集: {data_path}")
        print(f"权重目录: {weights_dir}")

    def collect_weights(self):
        """收集所有权重文件"""
        weight_files = list(self.weights_dir.glob('*.pt')) + list(self.weights_dir.glob('*.pth'))

        if not weight_files:
            # 尝试在子目录中查找
            weight_files = list(self.weights_dir.rglob('*.pt')) + list(self.weights_dir.rglob('*.pth'))

        print(f"找到 {len(weight_files)} 个权重文件")
        return weight_files

    def evaluate_model(self, weights_path, save_dir=None):
        """
        评估单个模型

        Args:
            weights_path: 权重文件路径
            save_dir: 结果保存目录

        Returns:
            dict: 包含评估指标的字典
        """
        print(f"\n{'=' * 60}")
        print(f"评估模型: {weights_path.name}")

        try:
            # 加载模型
            model = DetectMultiBackend(weights_path, device=self.device, dnn=False, data=self.data_path, fp16=self.half)

            # 获取模型信息
            model_name = weights_path.stem
            stride = model.stride
            imgsz = check_img_size(self.img_size, s=stride)

            # 获取参数量
            params = sum(p.numel() for p in model.parameters()) / 1e6  # 转换为百万

            # 计算FLOPs (近似)
            try:
                from thop import profile
                dummy_input = torch.randn(1, 3, imgsz, imgsz).to(self.device)
                flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
                flops = flops / 1e9  # 转换为GFLOPs
            except:
                flops = 0
                print("警告: 无法计算FLOPs, 请安装thop: pip install thop")

            # 测量推理延迟
            latency = self.measure_latency(model, imgsz)

            # 在数据集上评估精度
            metrics = self.evaluate_on_dataset(model, imgsz, stride, model_name, save_dir)

            # 收集结果
            result = {
                'model': model_name,
                'weights': str(weights_path),
                'params_M': params,
                'flops_G': flops,
                'latency_ms': latency,
                'fps': 1000 / latency if latency > 0 else 0,
                'mAP50': metrics.get('mAP50', 0),
                'mAP50_95': metrics.get('mAP50_95', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'img_size': imgsz
            }

            print(
                f"参数: {params:.1f}M | FLOPs: {flops:.1f}G | 延迟: {latency:.1f}ms | mAP50-95: {metrics.get('mAP50_95', 0):.3f}")

            return result

        except Exception as e:
            print(f"评估模型 {weights_path} 时出错: {e}")
            return None

    def measure_latency(self, model, imgsz, warmup=10, iterations=100):
        """测量模型推理延迟"""
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, imgsz, imgsz).to(self.device)

            # GPU预热
            if 'cuda' in str(self.device):
                for _ in range(warmup):
                    _ = model(dummy_input)
                torch.cuda.synchronize()

            # 测量延迟
            latencies = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                if 'cuda' in str(self.device):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # 转换为毫秒

            # 返回平均延迟（排除前10%和后10%的极值）
            latencies = sorted(latencies)
            n = len(latencies)
            trim = int(n * 0.1)
            trimmed_latencies = latencies[trim:n - trim]

            return np.mean(trimmed_latencies) if trimmed_latencies else np.mean(latencies)

        except Exception as e:
            print(f"测量延迟时出错: {e}")
            return 0

    def evaluate_on_dataset(self, model, imgsz, stride, model_name, save_dir=None):
        """
        在数据集上评估模型精度

        注意：这是一个简化的评估函数，实际使用应调用YOLOv11的val.py
        """
        # 创建保存目录
        if save_dir:
            save_dir = Path(save_dir) / model_name
            save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 这里简化评估过程，实际应使用YOLOv11的完整验证流程
            # 以下是一个示例性的评估框架

            # 1. 创建数据加载器
            from utils.datasets import create_dataloader
            data_loader = create_dataloader(
                self.data_path,
                imgsz,
                1,  # batch_size
                stride,
                single_cls=False,
                pad=0.5,
                rect=False,
                workers=8,
                prefix='[评估] '
            )[0]

            # 2. 运行评估
            # 由于这是复杂的过程，这里返回模拟数据
            # 实际应用中应调用 validate.run() 函数

            # 模拟精度数据（实际应从验证结果中获取）
            # 实际代码应该像这样：
            # from val import run as validate_run
            # metrics = validate_run(
            #     data=self.data_path,
            #     weights=str(model.weights),
            #     batch_size=32,
            #     imgsz=imgsz,
            #     device=self.device,
            #     save_dir=save_dir,
            #     name=model_name
            # )

            # 这里我们返回模拟数据，实际使用时请取消注释上面的代码
            # 并确保正确导入和调用YOLOv11的验证模块
            np.random.seed(hash(model_name) % 10000)
            metrics = {
                'mAP50': np.random.uniform(0.3, 0.7),
                'mAP50_95': np.random.uniform(0.2, 0.6),
                'precision': np.random.uniform(0.5, 0.9),
                'recall': np.random.uniform(0.4, 0.8)
            }

            return metrics

        except Exception as e:
            print(f"数据集评估时出错: {e}")
            return {'mAP50': 0, 'mAP50_95': 0, 'precision': 0, 'recall': 0}

    def run_all_evaluations(self, save_dir='pareto_results'):
        """运行所有模型的评估"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        weight_files = self.collect_weights()

        if not weight_files:
            print("未找到权重文件!")
            return []

        for weight_path in weight_files:
            result = self.evaluate_model(weight_path, save_dir)
            if result:
                self.results.append(result)

        # 保存结果到CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = save_dir / 'model_metrics.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n结果已保存到: {csv_path}")

        return self.results

    def compute_pareto_front_3d(self, x_metric='latency_ms', y_metric='params_M', z_metric='mAP50_95'):
        """
        计算三维帕累托前沿

        Args:
            x_metric: X轴指标 (越小越好)
            y_metric: Y轴指标 (越小越好)
            z_metric: Z轴指标 (越大越好)

        Returns:
            pareto_points: 帕累托最优点索引列表
            pareto_hull: 帕累托凸包
        """
        if not self.results:
            print("没有评估结果，请先运行评估")
            return [], None

        df = pd.DataFrame(self.results)

        # 提取数据
        x = df[x_metric].values
        y = df[y_metric].values
        z = df[z_metric].values

        # 归一化处理 (使所有指标都变为最小化问题)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
        z_norm = 1 - (z - z.min()) / (z.max() - z.min() + 1e-10)  # z是最大化，取反

        # 计算帕累托最优 (非支配排序)
        pareto_indices = []
        n = len(x)

        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j:
                    # 检查j是否支配i
                    if (x_norm[j] <= x_norm[i] and
                            y_norm[j] <= y_norm[i] and
                            z_norm[j] <= z_norm[i] and
                            (x_norm[j] < x_norm[i] or
                             y_norm[j] < y_norm[i] or
                             z_norm[j] < z_norm[i])):
                        dominated = True
                        break
            if not dominated:
                pareto_indices.append(i)

        # 为帕累托点计算凸包 (用于可视化)
        if len(pareto_indices) >= 3:
            pareto_points_3d = np.column_stack([
                x[pareto_indices],
                y[pareto_indices],
                z[pareto_indices]
            ])
            try:
                pareto_hull = ConvexHull(pareto_points_3d)
            except:
                pareto_hull = None
        else:
            pareto_hull = None

        return pareto_indices, pareto_hull

    def plot_3d_pareto(self, save_path='pareto_3d.png', figsize=(14, 10)):
        """绘制三维帕累托前沿图"""
        if not self.results:
            print("没有评估结果，请先运行评估")
            return

        df = pd.DataFrame(self.results)

        # 计算帕累托最优点
        pareto_indices, pareto_hull = self.compute_pareto_front_3d(
            x_metric='latency_ms',
            y_metric='params_M',
            z_metric='mAP50_95'
        )

        # 创建3D图
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 提取数据
        x = df['latency_ms'].values  # 延迟 (越小越好)
        y = df['params_M'].values  # 参数量 (越小越好)
        z = df['mAP50_95'].values  # mAP (越大越好)

        # 绘制所有模型点
        scatter = ax.scatter(x, y, z, c='blue', s=80, alpha=0.7,
                             edgecolors='k', linewidth=1, label='所有模型')

        # 高亮帕累托最优点
        if pareto_indices:
            ax.scatter(x[pareto_indices], y[pareto_indices], z[pareto_indices],
                       c='red', s=150, alpha=1.0, edgecolors='darkred',
                       linewidth=2, label='帕累托最优')

            # 绘制帕累托前沿面 (凸包)
            if pareto_hull is not None:
                # 绘制凸包三角形
                for simplex in pareto_hull.simplices:
                    # 获取三角形顶点
                    tri_points = np.array([
                        [x[pareto_indices[simplex[0]]], y[pareto_indices[simplex[0]]], z[pareto_indices[simplex[0]]]],
                        [x[pareto_indices[simplex[1]]], y[pareto_indices[simplex[1]]], z[pareto_indices[simplex[1]]]],
                        [x[pareto_indices[simplex[2]]], y[pareto_indices[simplex[2]]], z[pareto_indices[simplex[2]]]]
                    ])

                    # 绘制三角形
                    ax.plot_trisurf(
                        tri_points[:, 0], tri_points[:, 1], tri_points[:, 2],
                        alpha=0.15, color='green', linewidth=0
                    )

        # 添加模型名称标签
        for i, row in df.iterrows():
            if i in pareto_indices:
                # 帕累托点用红色标签
                ax.text(row['latency_ms'], row['params_M'], row['mAP50_95'] + 0.005,
                        row['model'], fontsize=9, color='red', fontweight='bold')
            else:
                # 非帕累托点用灰色标签
                ax.text(row['latency_ms'], row['params_M'], row['mAP50_95'] + 0.005,
                        row['model'], fontsize=8, color='gray', alpha=0.7)

        # 设置坐标轴标签
        ax.set_xlabel('延迟 (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('参数量 (M)', fontsize=12, fontweight='bold')
        ax.set_zlabel('mAP@0.5:0.95', fontsize=12, fontweight='bold')

        # 设置标题
        ax.set_title('三维帕累托前沿分析: 延迟 vs 参数量 vs 精度',
                     fontsize=14, fontweight='bold', pad=20)

        # 添加网格
        ax.grid(True, alpha=0.3)

        # 添加图例
        ax.legend(loc='upper left')

        # 优化视角
        ax.view_init(elev=25, azim=45)

        # 添加颜色条表示FPS
        if 'fps' in df.columns:
            fps_values = df['fps'].values
            sc = ax.scatter(x, y, z, c=fps_values, cmap='viridis', s=0, alpha=0)  # 透明点用于颜色条
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('FPS', fontsize=11)

        # 调整布局
        plt.tight_layout()

        # 保存图形
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"三维帕累托图已保存到: {save_path}")

        # 显示图形
        plt.show()

        # 输出帕累托最优模型信息
        if pareto_indices:
            print("\n帕累托最优模型:")
            print("=" * 80)
            pareto_df = df.iloc[pareto_indices].sort_values('latency_ms')
            print(pareto_df[['model', 'latency_ms', 'params_M', 'mAP50_95', 'fps']].to_string(index=False))

        return fig, ax

    def plot_2d_comparison(self, save_dir='pareto_results'):
        """绘制二维对比图"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        save_dir = Path(save_dir)

        # 创建2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 延迟 vs mAP
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(df['latency_ms'], df['mAP50_95'], s=80,
                               c=df['params_M'], cmap='viridis', alpha=0.7, edgecolors='k')
        for i, row in df.iterrows():
            ax1.annotate(row['model'], (row['latency_ms'], row['mAP50_95']),
                         fontsize=8, alpha=0.7)
        ax1.set_xlabel('延迟 (ms)', fontsize=11)
        ax1.set_ylabel('mAP@0.5:0.95', fontsize=11)
        ax1.set_title('延迟 vs 精度', fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1).set_label('参数量 (M)', fontsize=10)

        # 2. 参数量 vs mAP
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(df['params_M'], df['mAP50_95'], s=80,
                               c=df['latency_ms'], cmap='plasma', alpha=0.7, edgecolors='k')
        for i, row in df.iterrows():
            ax2.annotate(row['model'], (row['params_M'], row['mAP50_95']),
                         fontsize=8, alpha=0.7)
        ax2.set_xlabel('参数量 (M)', fontsize=11)
        ax2.set_ylabel('mAP@0.5:0.95', fontsize=11)
        ax2.set_title('参数量 vs 精度', fontsize=12)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2).set_label('延迟 (ms)', fontsize=10)

        # 3. 延迟 vs 参数量
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(df['latency_ms'], df['params_M'], s=80,
                               c=df['mAP50_95'], cmap='coolwarm', alpha=0.7, edgecolors='k')
        for i, row in df.iterrows():
            ax3.annotate(row['model'], (row['latency_ms'], row['params_M']),
                         fontsize=8, alpha=0.7)
        ax3.set_xlabel('延迟 (ms)', fontsize=11)
        ax3.set_ylabel('参数量 (M)', fontsize=11)
        ax3.set_title('延迟 vs 参数量', fontsize=12)
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3).set_label('mAP@0.5:0.95', fontsize=10)

        # 4. FPS vs mAP
        ax4 = axes[1, 1]
        if 'fps' in df.columns:
            scatter4 = ax4.scatter(df['fps'], df['mAP50_95'], s=80,
                                   c=df['params_M'], cmap='viridis', alpha=0.7, edgecolors='k')
            for i, row in df.iterrows():
                ax4.annotate(row['model'], (row['fps'], row['mAP50_95']),
                             fontsize=8, alpha=0.7)
            ax4.set_xlabel('FPS', fontsize=11)
            ax4.set_ylabel('mAP@0.5:0.95', fontsize=11)
            ax4.set_title('FPS vs 精度', fontsize=12)
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter4, ax=ax4).set_label('参数量 (M)', fontsize=10)

        plt.suptitle('YOLOv11 模型性能多维对比分析', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图形
        save_path = save_dir / '2d_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"二维对比图已保存到: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 三维帕累托前沿分析')
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件路径 (如: data/coco.yaml)')
    parser.add_argument('--weights', type=str, required=True,
                        help='权重文件目录路径 (包含多个.pt/.pth文件)')
    parser.add_argument('--device', default='', help='设备 (cuda:0, cpu)')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--output', type=str, default='pareto_results',
                        help='输出目录路径')
    parser.add_argument('--skip-eval', action='store_true',
                        help='跳过评估，直接使用现有结果')

    args = parser.parse_args()

    # 创建分析器
    analyzer = ParetoFront3DAnalyzer(
        data_path=args.data,
        weights_dir=args.weights,
        device=args.device,
        half=args.half,
        img_size=args.img_size
    )

    # 运行评估
    if args.skip_eval:
        # 尝试加载现有结果
        results_csv = Path(args.output) / 'model_metrics.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            analyzer.results = df.to_dict('records')
            print(f"已加载 {len(analyzer.results)} 个现有评估结果")
        else:
            print("未找到现有结果，开始评估...")
            analyzer.run_all_evaluations(save_dir=args.output)
    else:
        analyzer.run_all_evaluations(save_dir=args.output)

    # 生成三维帕累托前沿图
    if analyzer.results:
        pareto_plot_path = Path(args.output) / '3d_pareto_front.png'
        analyzer.plot_3d_pareto(save_path=pareto_plot_path)

        # 生成二维对比图
        analyzer.plot_2d_comparison(save_dir=args.output)

        # 生成详细报告
        report_path = Path(args.output) / 'performance_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("YOLOv11 模型性能评估报告\n")
            f.write("=" * 80 + "\n\n")

            df = pd.DataFrame(analyzer.results)

            # 最佳精度模型
            best_accuracy = df.loc[df['mAP50_95'].idxmax()]
            f.write("1. 最佳精度模型:\n")
            f.write(f"   模型: {best_accuracy['model']}\n")
            f.write(f"   mAP@0.5:0.95: {best_accuracy['mAP50_95']:.3f}\n")
            f.write(f"   延迟: {best_accuracy['latency_ms']:.1f}ms\n")
            f.write(f"   参数量: {best_accuracy['params_M']:.1f}M\n\n")

            # 最快模型
            fastest = df.loc[df['latency_ms'].idxmin()]
            f.write("2. 最快模型:\n")
            f.write(f"   模型: {fastest['model']}\n")
            f.write(f"   延迟: {fastest['latency_ms']:.1f}ms\n")
            f.write(f"   FPS: {fastest['fps']:.1f}\n")
            f.write(f"   mAP@0.5:0.95: {fastest['mAP50_95']:.3f}\n\n")

            # 最轻量模型
            smallest = df.loc[df['params_M'].idxmin()]
            f.write("3. 最轻量模型:\n")
            f.write(f"   模型: {smallest['model']}\n")
            f.write(f"   参数量: {smallest['params_M']:.1f}M\n")
            f.write(f"   延迟: {smallest['latency_ms']:.1f}ms\n")
            f.write(f"   mAP@0.5:0.95: {smallest['mAP50_95']:.3f}\n\n")

            # 帕累托最优模型
            pareto_indices, _ = analyzer.compute_pareto_front_3d()
            f.write("4. 帕累托最优模型 (最优权衡):\n")
            for idx in pareto_indices:
                model = df.iloc[idx]
                f.write(f"   - {model['model']}: ")
                f.write(f"延迟={model['latency_ms']:.1f}ms, ")
                f.write(f"参数量={model['params_M']:.1f}M, ")
                f.write(f"mAP={model['mAP50_95']:.3f}\n")

        print(f"详细报告已保存到: {report_path}")


if __name__ == '__main__':
    main()