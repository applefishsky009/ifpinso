"""
下载pymongo库：pip install pymongo

collection_name可选以下集合名：main为主集合、images为关联集合
山本耀司品牌：fashion_main_modify、fashion_images_modify
Isabel Marant品牌：Isabel_Marant_main、Isabel_Marant_images
Burberry品牌：burberry_main、burberry_images
Blumarine品牌：Blumarine_main、Blumarine_images
"""
from pymongo import MongoClient
from pymongo import errors
import json  # 用于格式化输出数据


# 配置信息
config = {
    "host": "dev.ifpinso.com",
    "port": 1017,
    "username": "ifpinso",
    "password": "Ifpinso2025.",
    "db_name": "ifpinso",
    # 可更改collection_name
    "collection_name": "fashion_main_modify"
}

try:
    # 连接MongoDB
    client = MongoClient(
        host=config["host"],
        port=config["port"],
        username=config["username"],
        password=config["password"]
    )

    # 验证连接
    client.admin.command('ping')
    print("连接成功！")

    # 获取数据库和集合
    db = client[config["db_name"]]
    collection = db[config["collection_name"]]

    # 获取集合中的所有数据
    print("正在获取数据...")
    all_data = list(collection.find())

    # 显示结果
    print(f"\n在集合 {config['collection_name']} 中找到 {len(all_data)} 条数据：")


    # 只显示前5条，避免数据过多刷屏
    for i, data in enumerate(all_data[:5]):
        print(f"\n数据 {i + 1}:")
        print(data)
    # 剩余更多数据
    if len(all_data) > 5:
        print(f"\n... 还有 {len(all_data) - 5} 条数据未显示")


    # 保存数据到json文件（可选）
    # filename = f"{config['collection_name']}_all_data.json"
    # print(f"正在将数据保存到 {filename}...")
    # with open(filename, 'w', encoding='utf-8') as f:
    #     for i, data in enumerate(all_data):
    #         # 将ObjectId转换为字符串，否则无法正常序列化
    #         if '_id' in data:
    #             data['_id'] = str(data['_id'])
    #             # data['服装ID'] = str(data['服装ID'])
    #         # 写入每条数据，使用JSON格式
    #         f.write(json.dumps(data, ensure_ascii=False, indent=2))  # ensure_ascii=False保证中文正常显示
    #         f.write("\n\n")
    # print(f"所有数据已成功保存到 {filename}")


except errors.ConnectionFailure:
    print("连接失败，请检查服务器地址和端口是否正确")
except Exception as e:
    print(f"发生错误：{str(e)}")

