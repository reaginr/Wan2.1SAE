"""
验证实时池化与事后计算的等价性

证明: 实时池化的mean与保存全部数据后计算的mean完全相同
"""

import numpy as np

np.random.seed(42)

# 模拟数据
T, L, D = 30, 32760, 6144
print(f"模拟SAE激活: [{T}, {L}, {D}] = {T*L*D*4/1024**3:.2f}GB")

# 由于内存限制，我们用小规模验证
T_small, L_small = 10, 1000
print(f"\n小规模验证: [{T_small}, {L_small}, {D}]")

# 方法1: 保存全部，事后计算（原始方法）
print("\n方法1: 保存全部数据...")
all_data = []
for t in range(T_small):
    timestep_data = np.random.randn(L_small, D).astype(np.float32)
    all_data.append(timestep_data)

# 合并并计算均值
stacked = np.stack(all_data, axis=0)  # [T, L, D]
mean_method1 = stacked.mean(axis=(0, 1))  # [D]
print(f"  数据形状: {stacked.shape}")
print(f"  内存占用: {stacked.nbytes / 1024**2:.2f}MB")

# 方法2: 实时池化（新方法）
print("\n方法2: 实时池化...")
pool_sum = np.zeros(D, dtype=np.float64)  # 使用float64避免精度损失
pool_count = 0

for t in range(T_small):
    timestep_data = all_data[t]  # [L, D]
    pool_sum += timestep_data.sum(axis=0)
    pool_count += timestep_data.shape[0]

mean_method2 = (pool_sum / pool_count).astype(np.float32)
print(f"  累积tokens: {pool_count}")
print(f"  池化内存: ~{pool_sum.nbytes / 1024:.2f}KB")

# 比较两种方法
print("\n" + "="*60)
print("等价性验证")
print("="*60)
diff = np.abs(mean_method1 - mean_method2).max()
relative_diff = diff / (np.abs(mean_method1).mean() + 1e-10)

print(f"最大绝对差异: {diff:.10f}")
print(f"相对差异: {relative_diff:.10e}")
print(f"结果: {'[OK] 等价' if diff < 1e-5 else '[FAIL] 不等价'}")

# 验证其他统计量
print("\n" + "="*60)
print("其他统计量验证")
print("="*60)

# 方法1: 事后计算std
std_method1 = stacked.std(axis=(0, 1))

# 方法2: 实时计算std（使用Welford算法）
# 需要保存sum和sum_sq
pool_sum_sq = np.zeros(D, dtype=np.float64)
for t in range(T_small):
    timestep_data = all_data[t]
    pool_sum_sq += (timestep_data ** 2).sum(axis=0)

# var = E[x^2] - E[x]^2
var_method2 = pool_sum_sq / pool_count - mean_method2 ** 2
var_method2 = np.maximum(0, var_method2)  # 防止数值误差
std_method2 = np.sqrt(var_method2)

std_diff = np.abs(std_method1 - std_method2).max()
print(f"Std最大差异: {std_diff:.10f}")
print(f"结果: {'[OK] 等价' if std_diff < 1e-4 else '[FAIL] 不等价'}")

# 方法1: 事后计算max/min
max_method1 = stacked.max(axis=(0, 1))
min_method1 = stacked.min(axis=(0, 1))

# 方法2: 实时计算max/min
max_method2 = np.stack([d.max(axis=0) for d in all_data], axis=0).max(axis=0)
min_method2 = np.stack([d.min(axis=0) for d in all_data], axis=0).min(axis=0)

max_diff = np.abs(max_method1 - max_method2).max()
min_diff = np.abs(min_method1 - min_method2).max()
print(f"Max差异: {max_diff:.10f} - {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
print(f"Min差异: {min_diff:.10f} - {'[OK]' if min_diff < 1e-5 else '[FAIL]'}")

print("\n" + "="*60)
print("概念向量计算验证")
print("="*60)

# 模拟正负样本
N_pos, N_neg = 50, 50

# 原始方法：为每个样本保存完整数据
print("模拟原始方法（保存完整数据）...")
pos_samples_full = []
neg_samples_full = []

for i in range(N_pos):
    # 正样本：均值偏向+0.5
    sample = np.random.randn(T_small, L_small, D).astype(np.float32) + 0.5
    pos_samples_full.append(sample)

for i in range(N_neg):
    # 负样本：均值偏向-0.5
    sample = np.random.randn(T_small, L_small, D).astype(np.float32) - 0.5
    neg_samples_full.append(sample)

# 计算概念向量
pos_means_full = np.stack([s.mean(axis=(0,1)) for s in pos_samples_full])
neg_means_full = np.stack([s.mean(axis=(0,1)) for s in neg_samples_full])
concept_vector_full = pos_means_full.mean(axis=0) - neg_means_full.mean(axis=0)

print(f"  内存占用: {(N_pos+N_neg)*T_small*L_small*D*4/1024**3:.2f}GB")

# 实时池化方法
print("\n模拟实时池化方法...")
pos_means_pooled = []
neg_means_pooled = []

for i in range(N_pos):
    sample_sum = np.zeros(D, dtype=np.float64)
    sample_count = 0
    for t in range(T_small):
        data = np.random.randn(L_small, D).astype(np.float32) + 0.5
        sample_sum += data.sum(axis=0)
        sample_count += L_small
    pos_means_pooled.append((sample_sum / sample_count).astype(np.float32))

for i in range(N_neg):
    sample_sum = np.zeros(D, dtype=np.float64)
    sample_count = 0
    for t in range(T_small):
        data = np.random.randn(L_small, D).astype(np.float32) - 0.5
        sample_sum += data.sum(axis=0)
        sample_count += L_small
    neg_means_pooled.append((sample_sum / sample_count).astype(np.float32))

pos_means_pooled = np.stack(pos_means_pooled)
neg_means_pooled = np.stack(neg_means_pooled)
concept_vector_pooled = pos_means_pooled.mean(axis=0) - neg_means_pooled.mean(axis=0)

print(f"  内存占用: {(N_pos+N_neg)*7*D*4/1024**2:.2f}MB")

# 比较概念向量
concept_diff = np.abs(concept_vector_full - concept_vector_pooled).max()
cosine_sim = np.dot(concept_vector_full, concept_vector_pooled) / (
    np.linalg.norm(concept_vector_full) * np.linalg.norm(concept_vector_pooled)
)

print(f"\n概念向量最大差异: {concept_diff:.10f}")
print(f"概念向量余弦相似度: {cosine_sim:.10f}")
print(f"结果: {'[OK] 等价' if concept_diff < 1e-4 else '[FAIL] 不等价'}")

print("\n" + "="*60)
print("总结")
print("="*60)
print(f"实时池化的mean与事后计算: {'等价 ✓' if diff < 1e-5 else '不等价 ✗'}")
print(f"概念向量计算: {'等价 ✓' if concept_diff < 1e-4 else '不等价 ✗'}")
print(f"内存节省: {(N_pos+N_neg)*T_small*L_small*D*4/1024**3 / ((N_pos+N_neg)*7*D*4/1024**2) * 1000:.0f}x")
