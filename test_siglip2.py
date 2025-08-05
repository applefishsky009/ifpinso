import os
import sys
import glob
import torch
import pickle
import hashlib
import random
import numpy as np
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


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


def save_vector_database(embeddings_matrix, src_imgs, cache_dir="vector_cache"):
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
  
  cache_file = os.path.join(cache_dir, "vector_database.pkl")
  
  try:
      with open(cache_file, 'wb') as f:
          pickle.dump(cache_data, f)
      print(f"向量数据库已保存到: {cache_file}")
      print(f"数据集哈希: {dataset_hash}")
      return True
  except Exception as e:
      print(f"保存向量数据库时出错: {e}")
      return False


def load_vector_database(src_imgs, cache_dir="vector_cache"):
  """
  从文件加载向量数据库
  """
  cache_file = os.path.join(cache_dir, "vector_database.pkl")
  
  if not os.path.exists(cache_file):
      print("未找到缓存文件，需要重新构建向量数据库")
      return None, None
  
  try:
      with open(cache_file, 'rb') as f:
          cache_data = pickle.load(f)
      
      # 检查数据集是否发生变化
      current_hash = get_dataset_hash(src_imgs)
      cached_hash = cache_data.get('dataset_hash', '')
      
      if current_hash != cached_hash:
          print("数据集已发生变化，需要重新构建向量数据库")
          print(f"当前哈希: {current_hash}")
          print(f"缓存哈希: {cached_hash}")
          return None, None
      
      # 检查图片文件是否仍然存在
      cached_imgs = cache_data['src_imgs']
      missing_files = [img for img in cached_imgs if not os.path.exists(img)]
      if missing_files:
          print(f"发现 {len(missing_files)} 个缓存中的文件已不存在，需要重新构建")
          return None, None
      
      embeddings_matrix = cache_data['embeddings_matrix']
      src_imgs_cached = cache_data['src_imgs']
      
      print(f"成功加载缓存的向量数据库")
      print(f"向量矩阵形状: {embeddings_matrix.shape}")
      print(f"图片数量: {len(src_imgs_cached)}")
      print(f"缓存时间: {cache_data.get('timestamp', '未知')}")
      
      return embeddings_matrix, src_imgs_cached
      
  except Exception as e:
      print(f"加载向量数据库时出错: {e}")
      return None, None


def build_vector_database(src_imgs, model, processor, cache_dir="vector_cache", force_rebuild=False):
  """
  构建源图像的向量数据库（带缓存功能）
  """
  # 如果不强制重建，先尝试加载缓存
  if not force_rebuild:
      embeddings_matrix, cached_src_imgs = load_vector_database(src_imgs, cache_dir)
      if embeddings_matrix is not None and cached_src_imgs is not None:
          return embeddings_matrix, cached_src_imgs
  
  # 缓存不存在或需要重建，开始构建新的向量数据库
  print(f"开始构建向量数据库，共 {len(src_imgs)} 张图片...")
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
  
  # 保存到缓存
  save_vector_database(embeddings_matrix, valid_src_imgs, cache_dir)
  
  return embeddings_matrix, valid_src_imgs

def find_most_similar(query_embedding, embeddings_matrix, src_imgs, top_k=1):
  """
  找到最相似的图片
  """
  # 计算余弦相似度
  similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
  
  # 找到最相似的top_k个
  top_indices = np.argsort(similarities)[-top_k:][::-1]
  
  results = []
  for idx in top_indices:
      results.append({
          'image_path': src_imgs[idx],
          'similarity': similarities[idx],
          'index': idx
      })
  
  return results

def display_comparison(test_img_path, similar_results, save_path=None):
  """
  在一张图上显示测试图片和最相似的图片
  """
  fig, axes = plt.subplots(1, len(similar_results) + 1, figsize=(15, 5))
  
  # 显示测试图片
  test_img = Image.open(test_img_path)
  axes[0].imshow(test_img)
  axes[0].set_title(f'测试图片\n{os.path.basename(test_img_path)}', fontsize=10)
  axes[0].axis('off')
  
  # 显示相似图片
  for i, result in enumerate(similar_results):
      similar_img = Image.open(result['image_path'])
      axes[i + 1].imshow(similar_img)
      axes[i + 1].set_title(f'最相似图片\n相似度: {result["similarity"]:.4f}\n{os.path.basename(result["image_path"])}', 
                           fontsize=10)
      axes[i + 1].axis('off')
  
  plt.tight_layout()
  
  if save_path:
      plt.savefig(save_path, dpi=150, bbox_inches='tight')
      print(f"结果已保存到: {save_path}")
  
  plt.show()

def main(force_rebuild=False):
  # 数据集分割
  test_imgs, src_imgs = split_train_val()
  print(f"测试集图片数量: {len(test_imgs)}")
  print(f"源图片数量: {len(src_imgs)}")
  
  # 加载模型
  ckpt = "siglip2-base-patch16-384"
  model = AutoModel.from_pretrained(ckpt, device_map="cpu").eval()
  processor = AutoProcessor.from_pretrained(ckpt)
  
 # 构建或加载向量数据库
  embeddings_matrix, valid_src_imgs = build_vector_database(
      src_imgs, model, processor, force_rebuild=force_rebuild
  )
  
  # 创建结果保存目录
  os.makedirs("similarity_results", exist_ok=True)
  
  # 对测试图片进行相似度搜索
  print(f"\n开始处理测试图片，共 {len(test_imgs)} 张...")
  
  for i, test_img in enumerate(test_imgs[:10]):  # 限制处理前10张测试图片
      try:
          print(f"\n处理测试图片 {i+1}/{min(10, len(test_imgs))}: {os.path.basename(test_img)}")
          
          # 提取测试图片特征
          image = load_image(test_img)
          inputs = processor(images=[image], return_tensors="pt").to(model.device)
          with torch.no_grad():
              test_embedding = model.get_image_features(**inputs)
          
          test_embedding_np = test_embedding.cpu().numpy()
          
          # 找到最相似的图片
          similar_results = find_most_similar(test_embedding_np, embeddings_matrix, valid_src_imgs, top_k=3)
          
          print(f"最相似的图片: {os.path.basename(similar_results[0]['image_path'])}, 相似度: {similar_results[0]['similarity']:.4f}")
          
          # 显示比较结果
          save_path = f"similarity_results/comparison_{i+1}.png"
          display_comparison(test_img, similar_results, save_path)
          
      except Exception as e:
          print(f"处理测试图片 {test_img} 时出错: {e}")
          continue


if __name__ == "__main__":
  main()