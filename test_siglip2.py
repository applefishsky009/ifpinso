import os
import sys
import glob
import torch
import faiss
import pickle
import random
import hashlib
import numpy as np
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
import matplotlib.pyplot as plt
from utils import make_dirs
# 设置字体为 Noto Sans CJK SC
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def split_train_val(path ='DeepFashion/images/', seed=9, test_ratio=0.2):
    """
    Splits the dataset into training and validation sets.
    """
    imgs = glob.glob(os.path.join(path, "*.jpg"))
    random.seed(seed)
    random.shuffle(imgs)
    split_idx = int(len(imgs) * test_ratio)
    test_files = imgs[:split_idx]
    val_files = imgs[split_idx:]
    return test_files, val_files

def get_all_images(path):
    """
    获取指定路径下的所有图片（包括子目录）
    """
    imgs = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                imgs.append(os.path.join(root, file))
    return sorted(imgs)

def get_dataset_hash(src_imgs):
    """
    生成数据集的哈希值，用于检测数据集是否发生变化
    """
    # 将文件路径和修改时间组合生成哈希
    file_info = []
    for img_path in sorted(src_imgs):  # 排序确保一致性
        if os.path.exists(img_path):
            mtime = os.path.getmtime(img_path)
            file_info.append(f"{img_path}:{mtime}")
    
    combined_str = "|".join(file_info)
    return hashlib.md5(combined_str.encode()).hexdigest()

def save_vector_database(embeddings_matrix, src_imgs, faiss_index, cache_dir="vector_cache", \
        dataset_name="database", model_name="siglip2"):
    """
    保存向量数据库到文件
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成数据集哈希
    dataset_hash = get_dataset_hash(src_imgs)
    
    cache_data = {
        'embeddings_matrix': embeddings_matrix,
        'src_imgs': src_imgs,
        'dataset_hash': dataset_hash,
        'timestamp': np.datetime64('now'),
        'embedding_shape': embeddings_matrix.shape
    }
    
    # 保存数据
    cache_file = os.path.join(cache_dir, f"{dataset_name}_{model_name}_vector_database.pkl")
    faiss_file = os.path.join(cache_dir, f"{dataset_name}_{model_name}_faiss_index.index")
    
    try:
        # 保存基本数据
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # 保存FAISS索引
        faiss.write_index(faiss_index, faiss_file)
        
        print(f"向量数据库已保存到: {cache_file}")
        print(f"FAISS索引已保存到: {faiss_file}")
        print(f"数据集哈希: {dataset_hash}")
        return True
    except Exception as e:
        print(f"保存向量数据库时出错: {e}")
        return False

def load_vector_database(src_imgs, cache_dir="vector_cache", dataset_name="database", model_name="siglip2"):
    """
    从文件加载向量数据库
    """
    cache_file = os.path.join(cache_dir, f"{dataset_name}_{model_name}_vector_database.pkl")
    faiss_file = os.path.join(cache_dir, f"{dataset_name}_{model_name}_faiss_index.index")
    
    if not os.path.exists(cache_file) or not os.path.exists(faiss_file):
        print("未找到缓存文件，需要重新构建向量数据库")
        return None, None, None
    
    try:
        # 加载基本数据
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 检查数据集是否发生变化
        current_hash = get_dataset_hash(src_imgs)
        cached_hash = cache_data.get('dataset_hash', '')
        
        if current_hash != cached_hash:
            print("数据集已发生变化，需要重新构建向量数据库")
            print(f"当前哈希: {current_hash}")
            print(f"缓存哈希: {cached_hash}")
            return None, None, None
        
        # 检查图片文件是否仍然存在
        cached_imgs = cache_data['src_imgs']
        missing_files = [img for img in cached_imgs if not os.path.exists(img)]
        if missing_files:
            print(f"发现 {len(missing_files)} 个缓存中的文件已不存在，需要重新构建")
            return None, None, None
        
        # 加载FAISS索引
        faiss_index = faiss.read_index(faiss_file)
        
        embeddings_matrix = cache_data['embeddings_matrix']
        src_imgs_cached = cache_data['src_imgs']
        
        print(f"成功加载缓存的向量数据库")
        print(f"向量矩阵形状: {embeddings_matrix.shape}")
        print(f"图片数量: {len(src_imgs_cached)}")
        print(f"缓存时间: {cache_data.get('timestamp', '未知')}")
        
        return embeddings_matrix, src_imgs_cached, faiss_index
        
    except Exception as e:
        print(f"加载向量数据库时出错: {e}")
        return None, None, None

def build_vector_database(src_imgs, model, processor, cache_dir="vector_cache", \
    model_name = 'siglip2', dataset_name="database", force_rebuild=False):
    """
    构建源图像的向量数据库（带缓存功能）
    """
    # src_imgs = src_imgs[:200]
    # 如果不强制重建，先尝试加载缓存
    if not force_rebuild:
        embeddings_matrix, cached_src_imgs, faiss_index = load_vector_database(src_imgs, \
            cache_dir, dataset_name, model_name)
        if embeddings_matrix is not None and cached_src_imgs is not None and faiss_index is not None:
            return embeddings_matrix, cached_src_imgs, faiss_index
    
    # 缓存不存在或需要重建，开始构建新的向量数据库
    print(f"开始构建向量数据库 [{dataset_name}]，共 {len(src_imgs)} 张图片...")
    
    embeddings_list = []
    valid_src_imgs = []
    for i, src_img in enumerate(src_imgs):
        try:
            image = load_image(src_img)
            inputs = processor(images=[image], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                image_embeddings = model.get_image_features(**inputs)
                embeddings_list.append(image_embeddings.cpu().numpy())
                valid_src_imgs.append(src_img)
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(src_imgs)} 张图片")
                
        except Exception as e:
            print(f"处理图片 {src_img} 时出错: {e}")
            continue
    
    # 将所有向量堆叠成矩阵
    embeddings_matrix = np.vstack(embeddings_list)
    print(f"向量数据库构建完成，形状: {embeddings_matrix.shape}")
    
    # 构建FAISS索引
    dimension = embeddings_matrix.shape[1]
    
    # 使用GPU FAISS索引（如果有GPU）
    if 0:
        res = faiss.StandardGpuResources()  # only for CUDA 11.4 and 12.1
        # 使用IVFFlat索引，适合大规模搜索
        quantizer = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        faiss_index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(valid_src_imgs)//10))
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    else:
        # CPU版本
        faiss_index = faiss.IndexFlatIP(dimension)
    
    # 训练索引（如果是IVF类型）
    if hasattr(faiss_index, 'train'):
        faiss_index.train(embeddings_matrix.astype(np.float32))
    
    # 添加向量到索引
    faiss_index.add(embeddings_matrix.astype(np.float32))
    
    print(f"FAISS索引构建完成，索引中有 {faiss_index.ntotal} 个向量")
    
    # 保存到缓存
    save_vector_database(embeddings_matrix, valid_src_imgs, faiss_index, cache_dir, dataset_name, model_name)
    
    return embeddings_matrix, valid_src_imgs, faiss_index

def find_most_similar_faiss(query_embedding, faiss_index, src_imgs, top_k=1):
    """
    使用FAISS找到最相似的图片
    """
    # 确保查询向量是正确的格式
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # 搜索最相似的向量
    similarities, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    
    results = []
    for i in range(top_k):
        if indices[0][i] < len(src_imgs):  # 确保索引有效
            results.append({
                'image_path': src_imgs[indices[0][i]],
                'similarity': similarities[0][i],
                'index': indices[0][i]
            })
    
    return results

def display_comparison(query_img_path, similar_results, save_path=None):
    """
    在一张图上显示查询图片和最相似的图片
    """
    fig, axes = plt.subplots(1, len(similar_results) + 1, figsize=(15, 5))
    
    # 显示查询图片
    query_img = Image.open(query_img_path)
    axes[0].imshow(query_img)
    axes[0].set_title(f'query image\n{os.path.basename(query_img_path)}', fontsize=10)
    axes[0].axis('off')
    
    # 显示相似图片
    for i, result in enumerate(similar_results):
        similar_img = Image.open(result['image_path'])
        axes[i + 1].imshow(similar_img)
        axes[i + 1].set_title(f'match picture {i+1}\nsimilarity: {result["similarity"]:.4f}\n{os.path.basename(result["image_path"])}',
                             fontsize=10)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
    
    plt.show()

def main(database_path='/data/yili/data/DeepFashion', query_path='/data/yili/data/Re-PolyVore', \
    model_path='/data/yili/model/siglip2-base-patch16-384', force_rebuild=False):
    """
    主函数
    database_path: 数据库图片路径
    query_path: 查询图片路径
    """
    # 获取数据库和查询图片
    database_imgs = get_all_images(database_path)
    query_imgs = get_all_images(query_path)
    
    print(f"数据库图片数量: {len(database_imgs)}")
    print(f"查询图片数量: {len(query_imgs)}")
    
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    ckpt = model_path
    model = AutoModel.from_pretrained(ckpt).to(device).eval()
    processor = AutoProcessor.from_pretrained(ckpt, use_fast=True)
    
    cache_dir = "/data/yili/vector_cache"
    model_name = ckpt.split('/')[-1].replace('-', '_')
    make_dirs(cache_dir)
    # 构建或加载数据库向量
    embeddings_matrix, valid_database_imgs, faiss_index = build_vector_database(database_imgs, \
        model, processor, cache_dir=cache_dir, model_name=model_name, \
        dataset_name=database_path.split('/')[-1], force_rebuild=force_rebuild
    )
    
    # 创建结果保存目录
    classification = query_path.split('/')[-1]
    result_path = f"similarity_results/{model_name}/{classification}"
    make_dirs(result_path)
    
    # 对查询图片进行相似度搜索
    print(f"\n开始处理查询图片，共 {len(query_imgs)} 张...")
    
    for i, query_img in enumerate(query_imgs[:10]):  # 限制处理前10张查询图片
        try:
            print(f"\n处理查询图片 {i+1}/{min(10, len(query_imgs))}: {os.path.basename(query_img)}")
            
            # 提取查询图片特征
            image = load_image(query_img)
            inputs = processor(images=[image], return_tensors="pt").to(device)
            
            with torch.no_grad():
                query_embedding = model.get_image_features(**inputs)
            
            query_embedding_np = query_embedding.cpu().numpy()
            
            # 使用FAISS找到最相似的图片
            similar_results = find_most_similar_faiss(
                query_embedding_np, faiss_index, valid_database_imgs, top_k=3
            )
            
            print(f"最相似的图片: {os.path.basename(similar_results[0]['image_path'])}, 相似度: {similar_results[0]['similarity']:.4f}")
            
            # 显示比较结果
            save_path = f"{result_path}/comparison_{i+1}.png"
            display_comparison(query_img, similar_results, save_path)
            print(f'图片 {i} 处理完成')
            
        except Exception as e:
            print(f"处理查询图片 {query_img} 时出错: {e}")
            continue

if __name__ == "__main__":
    # 使用示例
    folders = ['dress', 'jumpsuit', 'outwear', 'pants', 'skirt', 'top']
    models = ['siglip2-base-patch16-384', 'siglip2-large-patch16-384', 'siglip2-so400m-patch16-384', 'siglip2-giant-opt-patch16-384']
    for folder in folders:
        for model in models[1:]:
            print(f'********* 目录： {folder}, 模型： {model} *********')
            main(
                database_path='/data/yili/data/DeepFashion',            # 数据库图片路径
                query_path=f'/data/yili/data/Re-PolyVore/{folder}',     # 查询图片路径
                model_path=f'/data/yili/model/{model}',                 # 模型路径
                force_rebuild=False                                     # 是否强制重建索引
            )