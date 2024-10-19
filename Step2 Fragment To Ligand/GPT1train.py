import torch
from transformers import AutoTokenizer, OpenAIGPTLMHeadModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import AllChem
from tqdm import tqdm


# 步骤1：数据准备
class MoleculeDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        self.pairs = [line.strip().split(',') for line in lines]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        scaffold_smiles, smiles = self.pairs[idx]
        return scaffold_smiles, smiles


# 假设数据文件为'molecules.csv'
dataset = MoleculeDataset('./train_data/zinc_scaffold.csv')

# 步骤2：加载标记器和模型
tokenizer = AutoTokenizer.from_pretrained('DeepChem/SmilesTokenizer_PubChem_1M')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')


# 步骤3：自定义Collate函数
def collate_fn(batch):
    inputs = [tokenizer(scaffold, return_tensors='pt', padding='max_length', max_length=100, truncation=True) for
              scaffold, _ in batch]
    targets = [tokenizer(smiles, return_tensors='pt', padding='max_length', max_length=100, truncation=True) for
               _, smiles in batch]

    input_ids = torch.stack([x['input_ids'].squeeze() for x in inputs])
    attention_mask = torch.stack([x['attention_mask'].squeeze() for x in inputs])
    labels = torch.stack([x['input_ids'].squeeze() for x in targets])

    return {'input_ids': input_ids, 'attention_mask': attention_mask}, {'input_ids': labels}


# 步骤4：训练数据集构建
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 步骤5：模型训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 5

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        inputs, targets = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        targets = targets['input_ids'].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Average Epoch Loss: {avg_epoch_loss:.4f}")

    # 保存检查点
    model.save_pretrained(f"./train_checkpoint_epoch_{epoch + 1}")

# 步骤6：生成和评估
model.eval()


def generate_molecule(scaffold_smiles):
    inputs = tokenizer(scaffold_smiles, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_length=100)
    generated_smiles = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_smiles


def evaluate_generated_molecules(generated_molecules, target_molecules):
    similarities = []
    for gen, tgt in zip(generated_molecules, target_molecules):
        gen_mol = Chem.MolFromSmiles(gen)
        tgt_mol = Chem.MolFromSmiles(tgt)
        if gen_mol and tgt_mol:
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2)
            tgt_fp = AllChem.GetMorganFingerprintAsBitVect(tgt_mol, 2)
            similarity = FingerprintSimilarity(gen_fp, tgt_fp)
            similarities.append(similarity)
    return np.mean(similarities)


# 示例生成和评估
generated_molecules = [generate_molecule(scaffold) for scaffold in [pair[0] for pair in dataset.pairs]]
mean_similarity = evaluate_generated_molecules(generated_molecules, [pair[1] for pair in dataset.pairs])
print(f'Mean Tanimoto Similarity: {mean_similarity}')
