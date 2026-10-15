import os
import cv2
import tempfile
import numpy as np
from pathlib import Path

# 导入底层算法包
from algorithm.iretinex import enhance_image
from medical_agent import MedicalAgent
from evaluator import calculate_dice

def save_image_cv2(path, img):
    """支持中文路径与Path对象的保存函数"""
    # 确保路径被转换为字符串
    cv2.imencode('.jpg', img)[1].tofile(str(path))

def main():
    # ================= 1. 动态路径溯源 (核心技术点) =================
    # CURRENT_DIR 指向 D:\Miller's Work\project
    CURRENT_DIR = Path(__file__).resolve().parent
    
    # ROOT_DIR 指向 D:\Miller's Work (向上跳一级)
    ROOT_DIR = CURRENT_DIR.parent
    
    # 根据你的描述，数据在 Miller's Work\data 下
    DATA_DIR = ROOT_DIR / "data"
    
    # 修正：定位数据集具体路径
    IMG_DIR = DATA_DIR / "images"
    GT_DIR = DATA_DIR / "masks"
    
    # 修正：将权重也改为基于 ROOT_DIR 的相对路径，确保可复刻性
    YOLO_WEIGHT_PATH = CURRENT_DIR / "weights" / "polyp_yolov8.pt"
    SAM_WEIGHT_PATH = CURRENT_DIR / "weights" / "medsam_vit_b.pth"
    
    # 结果保存路径
    RESULTS_DIR = ROOT_DIR / "results"
    
    # 定义子文件夹变量并自动创建
    ENHANCED_DIR = RESULTS_DIR / "enhanced"
    BBOX_DIR = RESULTS_DIR / "bbox"
    MASK_DIR = RESULTS_BASE = RESULTS_DIR / "mask" # 统一命名
    
    for d in [ENHANCED_DIR, BBOX_DIR, MASK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ================= 2. 初始化智能体 =================
    print(f"正在加载 AI 模型，锚点根目录: {ROOT_DIR}")
    # 传入字符串格式的路径
    agent = MedicalAgent(yolo_path=str(YOLO_WEIGHT_PATH), sam_path=str(SAM_WEIGHT_PATH))
    
    # ================= 3. 获取测试图片 =================
    if not IMG_DIR.exists():
        print(f"找不到图片文件夹: {IMG_DIR}")
        return
        
    all_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    test_files = all_files[:10] 
    
    print(f"\n找到 {len(all_files)} 张图片，准备测试前 {len(test_files)} 张...\n")
    
    # ================= 4. 批量测试与分类保存 =================
    total_dice = 0.0
    valid_count = 0
    
    for idx, img_name in enumerate(test_files, 1):
        img_path = IMG_DIR / img_name
        gt_mask_path = GT_DIR / img_name
        original_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if original_bgr is None:
            print("  -> Failed: could not read original image.")
            continue
        pred_box, pred_mask = agent.diagnose(original_bgr)
        
        print(f"[{idx}/{len(test_files)}] 正在处理: {img_name} ...")
        
        # A. 图像增强 (IRetinex)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = enhance_image(str(img_path), output_dir=tmpdir, weights=(0.7, 0.3))
            # 读取增强后的图像
            enhanced_bgr = cv2.imdecode(np.fromfile(result.fusion_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # 【保存任务 1】：保存增强后的纯净图片
        save_image_cv2(ENHANCED_DIR / img_name, enhanced_bgr)
            
        # B. 智能体识别 (YOLO + MedSAM)
        
        if pred_mask is not None:
            x1, y1, x2, y2 = map(int, pred_box)
            original_h, original_w = original_bgr.shape[:2]
            enhanced_h, enhanced_w = enhanced_bgr.shape[:2]
            if (original_w, original_h) != (enhanced_w, enhanced_h):
                scale_x = enhanced_w / original_w
                scale_y = enhanced_h / original_h
                draw_box = [
                    int(round(x1 * scale_x)),
                    int(round(y1 * scale_y)),
                    int(round(x2 * scale_x)),
                    int(round(y2 * scale_y)),
                ]
                draw_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (enhanced_w, enhanced_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                draw_box = [x1, y1, x2, y2]
                draw_mask = pred_mask > 0

            # 【保存任务 2】：绘制 YOLO 方框并保存
            bbox_img = enhanced_bgr.copy()
            dx1, dy1, dx2, dy2 = draw_box
            cv2.rectangle(bbox_img, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            save_image_cv2(BBOX_DIR / img_name, bbox_img)
            
            # 【保存任务 3】：绘制 MedSAM 掩膜并保存
            mask_img = enhanced_bgr.copy()
            overlay = enhanced_bgr.copy()
            overlay[draw_mask] = [0, 0, 255] 
            mask_img = cv2.addWeighted(overlay, 0.4, mask_img, 0.6, 0)
            save_image_cv2(MASK_DIR / img_name, mask_img)

            # C. 评分验证 (Dice)
            if gt_mask_path.exists():
                gt_img = cv2.imdecode(np.fromfile(str(gt_mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                gt_img = cv2.resize(gt_img, (pred_mask.shape[1], pred_mask.shape[0]))
                
                dice_score = calculate_dice(pred_mask, gt_img)
                total_dice += dice_score
                valid_count += 1
                print(f"  -> 成功! SSIM: {result.ssim:.4f} | Dice: {dice_score:.4f}")
            else:
                print(f"  -> 成功! (未找到Ground Truth，跳过打分)")
        else:
            print(f"  -> 失败: YOLO 未检测到病变区域。")

    # ================= 5. 输出最终报告 =================
    print("\n" + "="*40)
    print(" 🏥 医学影像智能体 - 批量测试报告")
    print("="*40)
    print(f"总计测试: {len(test_files)} 张")
    print(f"成功识别: {valid_count} 张")
    if valid_count > 0:
        avg_dice = total_dice / valid_count
        print(f"平均精确度 (Average Dice): {avg_dice:.4f}")
    print(f"图像分类保存路径: {RESULTS_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()
