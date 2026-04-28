"""V3 训练 Loss 曲线（473 train / 84 val）"""
import os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = '/private/tmp/claude-501/-Users-huangzs-Medevi-Agent/8f7afe28-44c8-47ca-a309-316bccd9141e/tasks/bksnvyfo2.output'

with open(LOG) as f:
    text = f.read()

train_i, train_l = [], []
val_i, val_l = [], []

for line in text.split('\n'):
    m = re.search(r'Iter (\d+): Train loss ([0-9.]+)', line)
    if m:
        train_i.append(int(m.group(1)))
        train_l.append(float(m.group(2)))
    m = re.search(r'Iter (\d+): Val loss ([0-9.]+)', line)
    if m:
        val_i.append(int(m.group(1)))
        val_l.append(float(m.group(2)))

print(f'Train: {len(train_i)} points, Val: {len(val_i)} points')
print(f'Train: {train_l[0]:.3f}→{train_l[-1]:.3f}')
print(f'Val:   {val_l[0]:.3f}→{val_l[-1]:.3f}')
print(f'Train/Val gap: {abs(train_l[-1]-val_l[-1]):.3f}')
print(f'Best val: {min(val_l):.3f} at iter {val_i[val_l.index(min(val_l))]}')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(train_i, train_l, 'b-', label='Train Loss', linewidth=2)
ax.plot(val_i, val_l, 'r-', label='Validation Loss', linewidth=2, marker='o', markersize=6)
ax.axhline(y=min(val_l), color='green', linestyle='--', alpha=0.4, label=f'Best val = {min(val_l):.3f}')

ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('LoRA Fine-tuning V3: Qwen2.5-1.5B, rank=8, dropout=0.1, lr=5e-6\n473 train / 84 val, grad_accum=2, seq_len=1536, M4 16GB', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)

os.makedirs('data', exist_ok=True)
plt.savefig('data/loss_curve_v3.png', dpi=150, bbox_inches='tight')
print(f'Saved: data/loss_curve_v3.png')
