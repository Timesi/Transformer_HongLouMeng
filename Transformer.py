import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap

# 超参
batch_size = 64  # 同时平行处理多少条独立数据（batch）
block_size = 256  # 训练、验证的字符串长度
device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用GPU
n_embd = 256  # 词嵌入的维度,尽量是2的次方数
num_heads = 8  # 多头注意力机制的头数
head_size = n_embd // num_heads  # 多头注意力每个头的维度
n_layer = 6  # 多级残差注意力的层数
learning_rate = 0.0003  # 学习率
max_iters = 1000  # 训练次数
eval_interval = int(max_iters / 10)  # 每隔多少次评估一次
eval_iters = 200  # 抽取多少batch进行评估
dropout_value = 0.2  # dropout比率,即20%的节点归零

wrap_width = 50  # 打印输出时的文本宽度

torch.manual_seed(1337)  # 随机种子
file_name = "Hong_Lou_Meng.txt"

# -------------数据预处理-----------------
# 读取文本
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

# 将文本转换为有序不重复的列表
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 字符和整数之间投影
stoi = {ch: i for i, ch in enumerate(chars)}  # 符号到整数
itos = {i: ch for i, ch in enumerate(chars)}  # 整数到符号

encode = lambda str1: [stoi[c] for c in str1]  # 编码，将字符串转换为数字串
decode = lambda list1: "".join([itos[i] for i in list1])  # 解码，将数字串转换为字符串

# 训练、验证分组
data = torch.tensor(encode(text), dtype=torch.long)  # 用整数表示字符
n = int(0.9 * len(data))  # 前90%的长度用于训练
train_data = data[:n]  # 训练数据
val_data = data[n:]  # 验证数据

print(f"文件{file_name}读取完成")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # 生成batch_size个随机索引
    x = torch.stack([data[i: i + block_size] for i in ix])  # 输入值batch
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])  # 标签值batch
    x, y = x.to(device), y.to(device)  # 放到GPU上运行
    return x, y


# -------------损失评测-----------------
@torch.no_grad()
def estimate_loss(model):  # 不做梯度计算的decorator,作用域为整个函数
    out = {}
    model.eval()  # 将模型转化为评估模式（默认模式是train）
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)  # 建立一个初始值为0的容器，用于存储loss值
        for k in range(eval_iters):
            X, Y = get_batch(split)  # split是一个字符串，用来控制get_batch函数的行为
            logits, loss = model(X, Y)  # model的输入值一个是index（以每个字符的序号表示序列），一个是targets
            losses[k] = loss.item()
        out[split] = losses.mean()  # out是含有两个元素的字典，一个是train，一个是val，每个元素对应一个loss的平均值
    model.train()  # 将模型转化为训练模式（如果之前没有转为评估模式，则不需要这一步，因为模型建立后默认为训练模式）
    return out


# Head类
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # 线性变换层
        self.query = nn.Linear(n_embd, head_size, bias=False)  # 线性变换层
        self.value = nn.Linear(n_embd, head_size, bias=False)  # 线性变换层

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  # 使用buffer生成不可训练的下三角矩阵，用作掩码矩阵
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # 注意力矩阵 (B, T, T)，乘以k.shape[-1]**-0.5是为了防止值过大
        wei = wei.masked_fill(self.tril == 0, float("-inf"))  # 下三角矩阵,用作掩码填充
        wei = F.softmax(wei, dim=-1)  # 正则化
        wei = self.dropout(wei)  # 随机归零一些值，增加网络的稳定性

        v = self.value(x)
        out = wei @ v  # (B, T, head_size)
        return out


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # 多个注意力头
        self.proj = nn.Linear(head_size * num_heads, n_embd)  # 输出层
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size)  # 自注意力（多头注意力）
        self.ffwd = FeedForward(n_embd)  # 前馈神经网络
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # 残差多头注意力网络
        x = x + self.ffwd(self.ln2(x))  # 残差线性前馈神经网络
        return x


# -------------语言模型-----------------
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # 词嵌入层
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # 位置嵌入层
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])  # 多级残差注意力网络
        self.ln_f = nn.LayerNorm(n_embd)  # 最后的归一化层
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B: batch_size, T: block_size, 数据为token（整数形式）
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T, device=device)
        position_embd = self.position_embedding_table(position_idx)
        x = token_embd + position_embd  # 嵌入向量 (B, T, n_embd)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # 降维，变成B*T行，C列
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, token_sequ, max_new_tokens): # token_sequ是已知的上文，max_new_tokens是续写的长度
        for _ in range(max_new_tokens):
            token_input = token_sequ[:, -block_size:]
            logits, loss = self.forward(token_input)  # logits: [B, T, vocab_size]
            logits = logits[:, -1, :]  # 只取字符串的最后一位（概率分布，向量格式）
            probs = F.softmax(logits, dim=-1)  # 转换为0-1之间的概率分布
            token_next = torch.multinomial(probs, num_samples=1)  # 概率分布向量 --> one-hot向量 --> 整数token
            token_sequ = torch.cat([token_sequ, token_next], dim=1)  # 拼接字符串
        new_tokens = token_sequ[:, -max_new_tokens:]
        return new_tokens


# -------------主函数-----------------
def main():
    print(f"训练内容：{file_name}")
    print(f"训练次数：{max_iters}")
    model = LanguageModel()
    model = model.to(device)  # 将模型放到GPU运行
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")  # 模型参数量

    # 设定一个优化器
    oprimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    for i in range(max_iters):

        # 验证
        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 取样
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        oprimizer.zero_grad(set_to_none=True)  # 优化器梯度清零
        loss.backward()  # 反向传播,计算新的梯度
        oprimizer.step()  # 优化器对梯度进行优化，更新参数

    print("训练结束，下面开始生成内容")

    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens)

    # 上文内容
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)  # (B, T) B = 1, T = block_size
    context[0, :] = val_data[start_idx: start_idx + block_size]
    context_str = decode(context[0].tolist())
    wrapped_context_str = textwrap.fill(context_str, width=wrap_width)

    # 真实下文
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
    real_next_tokens[0, :] = val_data[start_idx + block_size: start_idx + block_size + max_new_tokens]
    context_str = decode(real_next_tokens[0].tolist())
    wrapped_real_next_context_str = textwrap.fill(context_str, width=wrap_width)

    # 生成下文
    generated_tokens = model.generate(context, max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

    print("----------上文内容----------")
    print(wrapped_context_str)
    print("----------真实下文----------")
    print(wrapped_real_next_context_str)
    print("----------生成下文----------")
    print(wrapped_generated_str)


#-----------------运行-----------------
main()