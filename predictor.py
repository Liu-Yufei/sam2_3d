import json
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from torch.nn import functional as F
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import argparse
import re
from sklearn.cluster import KMeans
from sam2.utils.misc import fill_holes_in_mask_scores
from sam2.sam2_video_predictor import SAM2VideoPredictor
def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="/media/DATA1/lyf/few-shot-training-free/data")
    parser.add_argument('--topic_path', type=str, default='CHAOS')
    parser.add_argument('--outdir', type=str, default='test_1')
    parser.add_argument('--ckpt', type=str, default="/media/DATA1/lyf/few-shot-training-free/sam2_2d/checkpoints/sam2.1_hiera_large.pt")
    # parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--model_cfg', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--dino', type=int, default=3)
    parser.add_argument('--dinov3', type=bool, default=True)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--delta_1', type=float, default=0.25)
    parser.add_argument('--delta_2', type=float, default=0.25)
    parser.add_argument('--multiple', type=int, default=4)
    parser.add_argument('--left', type=float, default=-1.0)
    parser.add_argument('--right', type=float, default=0.25)
    parser.add_argument('--step', type=float, default=0.25)
    parser.add_argument('--p_t', type=int, default=0)
    args = parser.parse_args()
    return args

def image_reprocess(feat_img: np.ndarray): # img: HWC
    '''将feature化的image转为可视化的image'''
    img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min())
    img = (img * 255).astype('uint8')
    return img


def save_heatmap(test_image, sim, output_path, test_idx, sim_bg_fr, feat_type='fusion'):
    image = image_reprocess(test_image)
    heatmap_data = sim.cpu().numpy()
    # 保存路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path_heatmap = os.path.join(output_path, f'frame_{test_idx:04d}_heatmap_{feat_type}.png')
    # 创建绘图
    plt.figure(figsize=(10, 10))
    # 显示原始图片
    plt.imshow(image, cmap='gray', interpolation='nearest')  # 原始图片
    # 显示热力图，alpha控制透明度
    plt.imshow(heatmap_data, cmap='jet', alpha=0.5, interpolation='nearest') # interpolation: 
    plt.colorbar(label='Intensity')
    # 隐藏坐标轴
    plt.axis('off')
    plt.title('mean: '+str(sim_bg_fr.mean().item())+' std: '+str(sim_bg_fr.std().item()))
    # 保存结果到指定路径
    plt.savefig(output_path_heatmap, bbox_inches='tight', pad_inches=0)
    # 关闭绘图以释放内存
    plt.close()

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    # ax.save(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def save_all_in_one(test_image, sim, ann_obj_frame_points, test_idx, sim_bg_fr, output_path, feat_type='fusion'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    out_path = os.path.join(output_path, f'frame_{test_idx:04d}_all_{feat_type}.png')
    
    # ----------------------------------------
    # 创建一个 1×3 的大画布
    # ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # -------------- 1. Heatmap ----------------
    image = image_reprocess(test_image)
    heatmap_data = sim.cpu().numpy()
    axes[0].imshow(image, cmap='gray')
    axes[0].imshow(heatmap_data, cmap='jet', alpha=0.5)
    axes[0].set_title(f'Heatmap\nmean={sim_bg_fr.mean().item():.4f}  std={sim_bg_fr.std().item():.4f}')
    axes[0].axis('off')

    # -------------- 2. Prompts ----------------
    axes[1].imshow(image)
    show_points(ann_obj_frame_points[test_idx][0]*4,
                ann_obj_frame_points[test_idx][1],
                axes[1])
    axes[1].set_title('Prompts')
    axes[1].axis('off')

    # -------------- 3. Histogram ----------------
    data = sim.flatten().cpu().numpy()
    axes[2].hist(data, bins=50)
    axes[2].set_title('sim_bg_fr Distribution')
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Count")

    # ----------------------------------------
    # 保存一次即可
    # ----------------------------------------
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # print(f"Saved combined figure to: {out_path}")


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_prompts(test_image, test_idx, ann_obj_frame_points):
    test_image = image_reprocess(test_image)
    plt.imshow(test_image)
    show_points(ann_obj_frame_points[test_idx][0], ann_obj_frame_points[test_idx][1], plt.gca())
    plt.savefig(f'frame_{test_idx:04d}_prompts.png')
    plt.close()

def show_straits(sim, test_idx):

    data = sim.flatten().cpu().numpy()
    plt.hist(data, bins=50)
    plt.title("sim_bg_fr strait distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.savefig(f'frame_{test_idx:04d}_zhifangtu.png')

def load_video_predictor(sam2_checkpoint, model_cfg):
    
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    return predictor

def readjson(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_selected_frame(frame_list, n_shot = 3):
    # 从frame_list中均匀选择n_shot个frame
    n = len(frame_list)
    size = n // n_shot
    remainder = n % n_shot
    parts = []
    start_idx = 0
    for i in range(n_shot):
        #前remainder个size+1
        extra = 1 if i < remainder else 0
        end_idx = start_idx + size + extra
        parts.append(frame_list[start_idx:end_idx])
        start_idx = end_idx
    selected_frames = []
    for part in parts:
        m = len(part)
        if m % 2 == 1:
            # 奇数个，取中间
            selected_frames.append(part[m // 2])
        else:
            # 偶数个，取靠前的中间
            selected_frames.append(part[m // 2 - 1])
    selected_frames = [f'{x:05d}.jpg' for x in selected_frames]
    return selected_frames

def get_fg_feat(support_feat, feat_mask):
    '''
    input:
        support_feat: (H, W, C)
        feat_mask: (H, W)
    output:
        fg_feat: (1, C)
    '''
    fg_feat = support_feat[feat_mask>0]
    fg_feat_embedding = fg_feat.mean(0).unsqueeze(0)  # (1, C)
    fg_feat = fg_feat_embedding / fg_feat_embedding.norm(dim=-1, keepdim=True)
    return fg_feat

def get_bg_feat(support_feat, feat_mask):
    '''
    input:
        support_feat: (H, W, C)
        feat_mask: (H, W)
    output:
        bg_feat: (C, N)
    '''
    bg_feat = support_feat[feat_mask<=0]
    bg_feat = bg_feat.permute(1,0)
    bg_feat = bg_feat / bg_feat.norm(dim=0, keepdim=True)
    return bg_feat

def norm_query_feat(query_feat, C, h, w):
    '''
    input:
        query_feat: (C, H, W)
    output:
        query_feat: (C, H*W)
    '''
    query_feat = query_feat / query_feat.norm(dim=0, keepdim=True)
    query_feat = query_feat.reshape(C, h*w)
    return query_feat

def similarity(feat1, feat2):
    '''
    inputs:
        feat1: (1, C) or (C, )
        feat2: (C, N)
    outputs:
        sim: (1, N)
    '''
    sim = (feat1 @ feat2).to(feat1.dtype)  # (N, M)
    return sim

def get_all_similarity(support_feat, query_feat, feat_mask, org_hw):
    '''
    inputs:
        support_feat: (H, W, C)
        query_feat: (C, H, W)
        feat_mask: (H, W)
        org_hw: (org_H, org_W)
    outputs:
        fg_bg_sim: (1, 1, 1, N)
        que_sup_sim: (org_H, org_W)
    '''
    C, h, w = query_feat.shape
    fg_feat = get_fg_feat(support_feat, feat_mask)
    bg_feat = get_bg_feat(support_feat, feat_mask)
    query_feat= norm_query_feat(query_feat, C, h, w)  # (C, H, W)
    
    fg_bg_sim = similarity(fg_feat, bg_feat)
    fg_bg_sim = fg_bg_sim.unsqueeze(0).unsqueeze(0)  # (1, N) -> (1,1,1,4034)

    que_sup_sim = similarity(fg_feat, query_feat)
    que_sup_sim = que_sup_sim.view(1, 1, h, w) # (1, 1, H, W)
    que_sup_sim = F.interpolate(que_sup_sim, size=org_hw, mode="bilinear", align_corners=False)
    que_sup_sim = que_sup_sim.squeeze()

    return fg_bg_sim, que_sup_sim

def select_prompts(args, support_features, selected_query_features, support_dino_features, selected_query_dino_features, feat_masks, prompt_list, frame_start_idx, org_hw, org_image,frame_path,frame_0):
    # 对相应的帧取点。
    ann_obj_frame_points_organ = {}
    que_sup_sim_list = []
    que_sup_dino_sim_list = []
    que_sup_fusion_sim_list = []
    fg_bg_sim_list = []
    fg_bg_dino_sim_list = []
    fg_bg_fusion_sim_list = []
    for i, (support_feat, query_feat, support_dino_feat, query_dino_feat, feat_mask) in enumerate(zip(support_features, selected_query_features,support_dino_features,selected_query_dino_features,feat_masks)):
        feat_mask_tmp = feat_mask[0].clone().detach().unsqueeze(0).unsqueeze(0)
        feat_mask_tmp = F.interpolate(feat_mask_tmp, size=support_feat.shape[0: 2], mode="bilinear")
        feat_mask_tmp = feat_mask_tmp.squeeze()

        # ViT_feature:
        query_feat = query_feat.permute(2,0,1) # (H, W, C) -> (C, H, W)
        fg_bg_sim, que_sup_sim = get_all_similarity(support_feat, query_feat, feat_mask_tmp, org_hw)
        
        fg_bg_sim_list.append(fg_bg_sim)
        que_sup_sim_list.append(que_sup_sim)

        # dino_feature:
        support_dino_feat = support_dino_feat.permute(1,2,0) # (C, H, W) -> (H, W, C)
        fg_bg_dino_sim, que_sup_dino_sim = get_all_similarity(support_dino_feat, query_dino_feat, feat_mask_tmp, org_hw)

        fg_bg_dino_sim_list.append(fg_bg_dino_sim)
        que_sup_dino_sim_list.append(que_sup_dino_sim)

        # fusion 
        fg_bg_fusion_sim = fg_bg_sim * fg_bg_dino_sim *(1 - args.delta_1 - args.delta_2) + fg_bg_sim * args.delta_1 + fg_bg_dino_sim * args.delta_2
        que_sup_fusion_sim = que_sup_sim * que_sup_dino_sim *(1 - args.delta_1 - args.delta_2) + que_sup_sim * args.delta_1 + que_sup_dino_sim * args.delta_2
        h_f, w_f = que_sup_fusion_sim.shape
        que_sup_fusion_sim = que_sup_fusion_sim.reshape(1, 1, h_f, w_f)
        que_sup_fusion_sim = F.interpolate(que_sup_fusion_sim, size=org_hw, mode="bilinear", align_corners=False).squeeze()
        que_sup_fusion_sim_list.append(que_sup_fusion_sim)
        fg_bg_fusion_sim_list.append(fg_bg_fusion_sim)

        # select fusion prompts
        if args.p_t >0:
            p_t = args.p_t
        else:
            p_t = min((((que_sup_fusion_sim[que_sup_fusion_sim > 0.75].sum() / que_sup_fusion_sim.sum())* 10).int().item() + 2),3) # 前景占比
        topk_xy, topk_label, last_xy, last_label = point_selection_fusion(args, que_sup_fusion_sim, fg_bg_fusion_sim, p_t=p_t)

        # select prompts
        # topk_xy, topk_label, last_xy, last_label = point_selection(que_sup_sim, fg_bg_sim)
        prompt_xy = np.concatenate((topk_xy, last_xy), axis=1).squeeze(0)
        prompt_label = np.concatenate((topk_label, last_label), axis=1).squeeze(0)
        ann_obj_frame_points_organ[int(prompt_list[i].split('.')[0])-frame_start_idx] = (prompt_xy, prompt_label)

    return ann_obj_frame_points_organ, fg_bg_fusion_sim_list, que_sup_fusion_sim_list

def kmeans_sklearn(data: torch.Tensor, K:int, max_iters = 100):
    data = data.cpu().numpy()
    kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=max_iters, random_state=0).fit(data)
    return torch.tensor(kmeans.cluster_centers_)

def point_selection(sim:torch.Tensor, fg_bg_sim:torch.Tensor, topk=1000, part=2, p_t=2, t=6):
    # Top-1 point selection21
    mask_sim = sim.clone()
    w, h = mask_sim.shape
    topk_sim, topk_xy = mask_sim.flatten(0).topk(topk) # 返回最大的topk个值的索引
    topk_x = (topk_xy // h).unsqueeze(0) # 返回索引对应的行
    topk_y = (topk_xy - topk_x * h) # 返回索引对应的列
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0) # 将x,y拼接
    feature_pos = torch.cat([topk_xy, topk_sim.unsqueeze(1)],dim=1)
    centers_p = []
    center_p = kmeans_sklearn(feature_pos, p_t, max_iters=20) 
    centers_p.append(center_p[:, :2])
    topk_xy_torch = torch.cat(centers_p, dim=0).to(sim.device)
    topk_xy = topk_xy_torch.unsqueeze(0).cpu().numpy() # 转为numpy
    topk_label = np.array([1] * topk_xy.shape[1]).reshape(1,topk_xy.shape[1]) # 生成topk个1

    # Top-last point selection
    sim_bg_fr_mean = fg_bg_sim.mean()
    sim_bg_fr_std = fg_bg_sim.std()
    threshold_low = sim_bg_fr_mean + (-0.75)*sim_bg_fr_std
    threshold_high = sim_bg_fr_mean + (-0.25)*sim_bg_fr_std
    centers_n = []
    for i in range(part):
        threshold_low = threshold_low + 0.25 * i * sim_bg_fr_std
        threshold_high = threshold_high + 0.25 * i * sim_bg_fr_std
        mask_sim_flat = mask_sim.flatten(0)
        threshold_mask = (mask_sim_flat >= threshold_low) & (mask_sim_flat <= threshold_high)
        threshold_indices = torch.nonzero(threshold_mask).squeeze(1) # 返回在阈值范围内的所有值的索引
        # threshold_indices = threshold_indices[torch.randperm(threshold_indices.size(0))[:topk]] # 随机选取topk个点
        threshold_x = (threshold_indices // h).unsqueeze(0) # 返回索引对应的行
        threshold_y = (threshold_indices - threshold_x * h) # 返回索引对应的列
        threshold_xy = torch.cat((threshold_y, threshold_x), dim=0).permute(1, 0) # 将x,y拼接
        sim_vals = mask_sim_flat[threshold_indices].unsqueeze(1)
        feature_n =  torch.cat([threshold_xy, sim_vals],dim=1)
        # 随机选取threshold_mask中的100个点
        if threshold_xy.size(0)>=2*t//part:
            center = kmeans_sklearn(feature_n, K=t//part, max_iters=20) #t 2topk_xy, p_t, max_iters=10
            centers_n.append(center[:, :2].cpu())
        else:
            centers_n.append(feature_n[:, :2].cpu())

    max_centers = torch.cat(centers_n, dim=0)
    last_xy = max_centers.cpu().unsqueeze(0).numpy() # 转为numpy
    last_label = np.array([0] * last_xy.shape[1]).reshape(1,last_xy.shape[1]) # 生成对应数量的0

    return topk_xy, topk_label, last_xy, last_label


def point_selection_fusion(args, sim, sim_bg_fr_fusion, topk=50, part=2, p_t= 2, n_t=4):

    # Top-1 point selection
    mask_sim = sim.clone()
    w, h = mask_sim.shape
    topk_sim, topk_xy = mask_sim.flatten(0).topk(p_t * args.multiple) # 返回最大的topk个值的索引
    topk_x = (topk_xy // h).unsqueeze(0) # 返回索引对应的行
    topk_y = (topk_xy - topk_x * h) # 返回索引对应的列
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0) # 将x,y拼接
    centers_p = []
    center_p = kmeans_sklearn(topk_xy, p_t, max_iters=20) 
    centers_p.append(center_p)

    topk_xy_torch = torch.cat(centers_p, dim=0).to(sim.device)
    topk_xy = topk_xy_torch.unsqueeze(0).cpu().numpy() # 转为numpy
    # topk_xy = torch.cat([topk_xy[0:1],topk_xy_torch]).unsqueeze(0).cpu().numpy() # 转为numpy

    topk_label = np.array([1] * topk_xy.shape[1]).reshape(1, topk_xy.shape[1]) # 生成topk个1

    # 计算positive点之间的距离
    distances_p = np.linalg.norm(center_p[:, np.newaxis] - center_p, axis=-1)
    max_distance = torch.tensor(np.max(distances_p)).to(sim.device)
    np.fill_diagonal(distances_p, np.inf)
    # min_distance = torch.tensor(np.min(distances_p)).to(sim.device)

    # Top-last point selection
    sim_bg_fr_mean = sim_bg_fr_fusion.mean()
    sim_bg_fr_std = sim_bg_fr_fusion.std()
    threshold_low = sim_bg_fr_mean + args.left * sim_bg_fr_std
    threshold_high = sim_bg_fr_mean + args.right * sim_bg_fr_std
    centers_n = []
    # t_final = n_t
    for i in range(part):
        threshold_low = threshold_low + args.step * i * sim_bg_fr_std
        threshold_high = threshold_high + args.step * i * sim_bg_fr_std
        mask_sim_flat = mask_sim.flatten(0)
        threshold_mask = (mask_sim_flat >= threshold_low) & (mask_sim_flat <= threshold_high)
        threshold_indices = torch.nonzero(threshold_mask).squeeze(1) # 返回在阈值范围内的所有值的索引
        # threshold_indices = threshold_indices[torch.randperm(threshold_indices.size(0))[:topk]] # 随机选取topk个点
        # threshold_indices = threshold_indices[torch.randperm(threshold_indices.size(0))[:n_t//part*20]] # 随机选取topk个点
        threshold_x = (threshold_indices // h).unsqueeze(0) # 返回索引对应的行
        threshold_y = (threshold_indices - threshold_x * h) # 返回索引对应的列
        threshold_xy = torch.cat((threshold_y, threshold_x), dim=0).permute(1, 0) # 将x,y拼接
        if topk_xy.shape[1] >1:
            distances_n_p = torch.cdist(threshold_xy.float(), topk_xy_torch.float(),p=2) # 计算negative点到positive点的距禂
            valid_mask = torch.all(distances_n_p >= (max_distance), dim=1) # 保证negative点到positive点的距离大于min_distance
            threshold_xy = threshold_xy[valid_mask]
        threshold_xy = threshold_xy[torch.randperm(threshold_xy.size(0))[:n_t//part*args.multiple]]
        # 随机选取threshold_mask中的100个点-
        threshold_indices_new = threshold_xy[:,0] * h + threshold_xy[:,1]
         # 将坐标转为索引
        sim_vals = mask_sim_flat[threshold_indices_new].unsqueeze(1)
        feature = torch.cat([threshold_xy, sim_vals],dim=1)
        if threshold_xy.size(0)>2*n_t//part:
            center = kmeans_sklearn(feature, K=n_t//part, max_iters=20)
            centers_n.append(center[:, :2].cpu())
        else:
            centers_n.append(feature[:, :2].cpu())
    max_centers = torch.cat(centers_n, dim=0)
    last_xy = max_centers.cpu().unsqueeze(0).numpy() # 转为numpy
    last_label = np.array([0] * last_xy.shape[1]).reshape(1,last_xy.shape[1]) # 生成对应数量的0
    return topk_xy, topk_label, last_xy, last_label
    



def sam2_video_inference(ann_obj_frame_points, inference_state, predictor, frame_names, video_dir=None, show_result=False, vis_frame_stride=1, output_path=None):
    # Add all points
    for ann_obj_id,frame_points in ann_obj_frame_points.items():
    # if True:
        # ann_obj_id = 0
        frame_points = ann_obj_frame_points
        
        for i, (ann_frame_idx, (points, labels)) in enumerate(frame_points.items()):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                clear_old_points=False,
                labels=labels,
                support_feature = inference_state['support_features'][i],
                support_mask = inference_state['support_masks'][i],
            )
            if show_result and (output_path is not None):
                output_path_fig = os.path.join(output_path, f'frame_{ann_frame_idx:04d}_after_clicks.png')
                frame_idx = frame_names[ann_frame_idx]
                # show the results on the current (interacted) frame
                plt.figure(figsize=(9, 6))
                plt.title(f"frame {ann_frame_idx}")
                plt.imshow(Image.open(os.path.join(video_dir, f'{frame_idx:05d}.jpg')))
                show_points(points, labels, plt.gca())
                # for i in range(len(out_obj_ids)):
                #     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[i])
                show_mask((out_mask_logits[out_obj_ids.index(ann_obj_id)] > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)
                plt.savefig(output_path_fig)
                plt.close()
            # break

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(  
        inference_state,  
        # start_frame_idx=150,  
        reverse=True  # 向前传播  
    ):  
        video_segments[out_frame_idx] = {  
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()  
            for i, out_obj_id in enumerate(out_obj_ids)  
        }
    if show_result and (video_dir is not None):
        # render the segmentation results every few frames
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            frame_idx = frame_names[out_frame_idx]

            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, f'{frame_idx:05d}.jpg')))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            output_path_fig = os.path.join(output_path, f'frame_{out_frame_idx:04d}_segmentation.png')
            plt.savefig(output_path_fig)
            plt.close()
            
def main():
    args = get_arguments()
    # scan all the JPEG frame names in this directory
    sam2_checkpoint = "/media/DATA1/lyf/few-shot-training-free/sam2_2d/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_dir = "/media/DATA1/lyf/few-shot-training-free/data/CHAOS/SAM2_jpgs_1/"
    
    if args.topic_path=='RawData':
        images_path = args.data_path +'/'+ args.topic_path + '/sabs_CT_normalized' + '/images/'
        masks_path = args.data_path + '/'+ args.topic_path + '/sabs_CT_normalized' + '/labels/'
        json_path = args.data_path + '/'+ args.topic_path + '/sabs_CT_normalized' + '/classmap_1.json'
        label_to_organ = {1 : 'SPLEEN', 2 : 'KID_R', 3 : 'KID_l', 6 : 'LIVER'}
    elif args.topic_path=='CHAOS':
        images_path = args.data_path +'/'+ args.topic_path + '/chaos_MR_T2_normalized' + '/images/'
        masks_path = args.data_path + '/'+ args.topic_path + '/chaos_MR_T2_normalized' + '/labels/'
        dino_feature_path = args.data_path + '/'+ args.topic_path + '/chaos_MR_T2_normalized' + '/dino_features/'
        json_path = args.data_path + '/'+ args.topic_path + '/chaos_MR_T2_normalized' + '/classmap_1.json'
        label_to_organ = {1: 'LIVER', 2: 'RK', 3: 'LK',4: 'SPLEEN'}
    fold = args.fold

    predictor:SAM2VideoPredictor = load_video_predictor(sam2_checkpoint, model_cfg)

    classmap = readjson(json_path)

    # 得到需要标注的frame的frame index: （注意，这里不考虑没有当前要标注器官的帧。
    # 得到support frame 的 frame index
    obj_name_list = [obj_name.split("_")[-1].replace('.nii.gz','') for obj_name in os.listdir(images_path) if ".nii.gz" in obj_name]
    obj_name_list = sorted(obj_name_list, key=lambda x:int(re.search(r'(\d+)',x).group(1)))

    obj_name_list_fold = np.array_split(obj_name_list, fold)
    list_split_len = len(obj_name_list_fold[0])

    support_scan_list = [obj_name_list[i*list_split_len] for i in range(args.fold)]
    support_scan_list.append(support_scan_list.pop(0))
    for organ_idx, organ in label_to_organ.items():
        # 对于每一种organ：
        for fold_idx in range(fold):
            support_case_num = support_scan_list[fold_idx]
            # 找到对应的support frame
            support_video_dir = os.path.join(video_dir, support_case_num, 'images')
            support_mask_dir = os.path.join(video_dir, support_case_num, 'masks', organ)
            support_dino_feature_dir = os.path.join(dino_feature_path, 'feature_' + support_case_num + '.npy')
            support_frame_list = classmap[organ][support_case_num] #　support list of frame, int list, org idx
            support_list = get_selected_frame(support_frame_list, n_shot = 3) # support list of selected frame, str list, org idx
            inference_state, masks = predictor.init_state(video_path=support_video_dir, frame_list=support_frame_list,mask_path=support_mask_dir)
            support_features = []
            support_dino_features = []
            feat_masks = []
            predictor.reset_state(inference_state)
            support_dino_feature = np.load(support_dino_feature_dir)
            support_dino_feature = torch.from_numpy(support_dino_feature).to(predictor.device)
            with torch.no_grad():
                for _, frame_path in enumerate(support_list):
                    frame_idx = int(frame_path.split('.')[0])-support_frame_list[0]
                    # predictor.reset_state(inference_state)
                    all_features:torch.Tensor
                    _,_,all_features,_,feat_sizes = predictor._get_image_feature(inference_state, frame_idx=frame_idx, batch_size=1) # 要对mask对应进行放缩。
                    features_org = all_features[-1].permute(1, 2, 0).view(all_features[-1].size(1), all_features[-1].size(2), *feat_sizes[-1])
                    features = features_org.squeeze().permute(1, 2, 0)
                    support_features.append(features) # (64,64,256)
                    support_dino_features.append(support_dino_feature[frame_idx])
                    feat_masks.append(masks[frame_idx])  # (1, H, W), dist idx
            
                    
            for case_num in iter(obj_name_list_fold[fold_idx]):
            # if True:
                # case_num = '5'
                # 对于其中的每个case：(0-39)
                # 每个organ对应的frame:
                # case_num = case_name.split('_')[-1].replace('.nii.gz','')
                frame_list = classmap[organ][case_num]
                # frame_list.sort(key=lambda p: int(os.path.splitext(p)[0]))
                case_video_dir = os.path.join(video_dir, case_num, 'images')
                case_mask_dir = os.path.join(video_dir, case_num, 'masks', organ)
                case_dino_feature_dir = os.path.join(dino_feature_path, 'feature_' + case_num + '.npy')
                inference_state = predictor.init_state(video_path=case_video_dir, frame_list=frame_list)
                # 更新inference_state，将support_features和support_masks加入进去
                inference_state['support_features'] = support_features # list of (H, W, C)
                inference_state['support_masks'] = feat_masks # list of (1, H, W)
                predictor.reset_state(inference_state)
                selected_query_features = []
                selected_query_dino_features = []
                query_dino_features = np.load(case_dino_feature_dir)
                query_dino_features = torch.from_numpy(query_dino_features).to(predictor.device)
                prompt_list = get_selected_frame(frame_list, n_shot = 3)
                with torch.no_grad():
                    for _, frame_path in enumerate(prompt_list):
                        frame_idx = int(frame_path.split('.')[0]) - frame_list[0]
                        # predictor.reset_state(inference_state)
                        _,_,all_features,_,feat_sizes = predictor._get_image_feature(inference_state, frame_idx=frame_idx, batch_size=1)
                        features_org = all_features[-1].permute(1, 2, 0).view(all_features[-1].size(1), all_features[-1].size(2), *feat_sizes[-1]) # [B 1, C 256, H 64, W 64]
                        features = features_org.squeeze().permute(1, 2, 0) # (64,64,256)
                        selected_query_features.append(features)
                        selected_query_dino_features.append(query_dino_features[frame_idx])
                    # prompt
                    org_hw = inference_state["video_height"], inference_state["video_width"]
                    ann_obj_frame_points, fg_bg_sim_list, que_sup_sim_list = select_prompts(
                        args,
                        support_features, 
                        selected_query_features, 
                        support_dino_features,
                        selected_query_dino_features,
                        feat_masks, 
                        prompt_list, 
                        frame_list[0], 
                        org_hw,
                        inference_state["images"],
                        frame_path,
                        frame_list[0]
                    )
                    for idx, que_sup_sim in enumerate(que_sup_sim_list):
                        test_idx = int(prompt_list[idx].split('.')[0]) - frame_list[0]
                        # save_heatmap(
                        #     test_image = inference_state["images"][frame_idx].permute(1,2,0).cpu().numpy(),
                        #     sim = que_sup_sim,
                        #     output_path = os.path.join(args.outdir, case_num, organ, 'heatmaps'),
                        #     test_idx = test_idx,
                        #     # ref_idx = [int(f.split('.')[0]) - support_frame_list[0] for f in support_list],
                        #     sim_bg_fr = fg_bg_sim_list[idx],
                        #     feat_type = 'ViT'
                        # )
                        # show_prompts(
                        #     inference_state["images"][frame_idx].permute(1,2,0).cpu().numpy(), 
                        #     test_idx, 
                        #     ann_obj_frame_points
                        # )
                        # show_straits(
                        #     fg_bg_sim_list[idx],
                        #     test_idx,
                        # )
                        save_all_in_one(
                            test_image = inference_state["images"][test_idx].permute(1,2,0).cpu().numpy(),
                            sim = que_sup_sim,
                            ann_obj_frame_points = ann_obj_frame_points,
                            test_idx = test_idx,
                            sim_bg_fr = fg_bg_sim_list[idx],
                            # output_path = os.path.join(args.outdir, case_num, organ, 'all_in_one'),
                            output_path = '.',
                            feat_type = 'ViT'
                        )
                output_path = os.path.join(args.outdir, case_num, organ, 'segmentation')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                sam2_video_inference(
                    ann_obj_frame_points, 
                    inference_state, 
                    predictor, 
                    frame_list, 
                    case_video_dir, 
                    show_result=True, 
                    vis_frame_stride=1,
                    output_path = output_path
                )

                # ann_obj_frame_points[organ_idx]
    # frame_names = [
    #     p for p in os.listdir(video_dir)
    #     if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    # ]
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # ann_obj_frame_points = {
    #     1: {
    #         7: (np.array([[35, 85],[40, 125],[100,50]], dtype=np.float32),np.array( [1, 0, 0],dtype=np.int32)),
    #         12: (np.array([[25,75], [48,50], [30,144], [50,100], [100, 150],[200,100],[150,50],[125,175],[60,160]], dtype=np.float32), np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)),
    #     }
    #     # ,
    #     # 2: {
    #     #     20: (np.array([[320, 200],[310, 300]], dtype=np.float32), np.array([1, 1], dtype=np.int32))
    #     # }    
    # }



if __name__ == "__main__":
    main()