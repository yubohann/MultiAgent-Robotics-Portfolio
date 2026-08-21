import sys

print("="*60)
print("🔍 检查推荐系统依赖")
print("="*60)

dependencies = [
    ("kafka", "kafka-python", "Kafka消费者/生产者"),
    ("torch", "torch", "PyTorch深度学习"),
    ("pandas", "pandas", "数据处理"),
    ("numpy", "numpy", "数值计算"),
]

all_ok = True

for module, package, desc in dependencies:
    try:
        __import__(module)
        print(f"✅ {desc} - {package} 已安装")
    except ImportError:
        print(f"❌ {desc} - {package} 未安装")
        all_ok = False

print("="*60)

if all_ok:
    print("🎉 所有依赖已安装！")
else:
    print("\n💡 安装缺失依赖:")
    print("   pip install kafka-python torch pandas numpy")
print("="*60)
