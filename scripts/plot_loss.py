"""
从训练日志解析Loss数据并绘制曲线
"""
import json, re, os

LOG_PATH = '/private/tmp/claude-501/-Users-huangzs-Medevi-Agent/8f7afe28-44c8-47ca-a309-316bccd9141e/tasks/bwd8zydgj.output'

with open(LOG_PATH) as f:
    text = f.read()

train_iters = []
train_losses = []
val_iters = []
val_losses = []

for line in text.split('\n'):
    m = re.search(r'Iter (\d+): Train loss ([0-9.]+)', line)
    if m:
        train_iters.append(int(m.group(1)))
        train_losses.append(float(m.group(2)))
    m = re.search(r'Iter (\d+): Val loss ([0-9.]+)', line)
    if m:
        val_iters.append(int(m.group(1)))
        val_losses.append(float(m.group(2)))

print(f'解析结果: {len(train_iters)} train points, {len(val_iters)} val points')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_iters, train_losses, 'b-', label='Train Loss', linewidth=2)
if val_iters:
    ax.plot(val_iters, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='o')
ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5, label='Early stopping (~200 iters)')

ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('LoRA Fine-tuning Loss Curve (Qwen2.5-1.5B on M4, batch=1, lr=1e-5)')
ax.legend()
ax.grid(True, alpha=0.3)

os.makedirs('data', exist_ok=True)
plt.savefig('data/loss_curve.png', dpi=150, bbox_inches='tight')
print(f'Loss曲线已保存: data/loss_curve.png')
print(f'Train: {train_losses[0]:.3f} → {train_losses[-1]:.3f}')
print(f'Val:   {val_losses[0]:.3f} → {val_losses[-1]:.3f}')
