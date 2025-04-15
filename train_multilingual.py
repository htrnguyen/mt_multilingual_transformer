import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
from torch.utils.data import DataLoader, TensorDataset
from multilingual_transformer import create_multilingual_transformer
from language_routing import LanguageDetector, LanguageRouter
import sentencepiece as spm
import pickle

# Set console output encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

class MultilingualTrainer:
    """
    Lớp huấn luyện mô hình dịch máy đa ngôn ngữ
    """
    
    def __init__(self, model, sp_model_path, data_dir, output_dir, device='cpu'):
        """
        Khởi tạo bộ huấn luyện
        
        Args:
            model: Mô hình MultilingualTransformer
            sp_model_path: Đường dẫn đến mô hình SentencePiece
            data_dir: Thư mục chứa dữ liệu đã tiền xử lý
            output_dir: Thư mục lưu mô hình và kết quả
            device: Thiết bị tính toán ('cpu' hoặc 'cuda')
        """
        self.model = model
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Tải mô hình SentencePiece
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        
        # Ánh xạ mã ngôn ngữ sang ID ngôn ngữ
        self.language_ids = {
            'en': 0,
            'fr': 1,
            'es': 2,
            'vi': 3
        }
        
        # Chuyển mô hình sang thiết bị tính toán
        self.model = self.model.to(device)
    
    def load_data(self, language_pair, split='train'):
        """
        Tải dữ liệu từ file
        
        Args:
            language_pair: Cặp ngôn ngữ ('en_fr', 'en_es', 'en_vi')
            split: Phân chia dữ liệu ('train', 'val', 'test')
            
        Returns:
            Tuple (src_data, tgt_data)
        """
        # Đường dẫn đến file dữ liệu
        data_file = os.path.join(self.data_dir, f"{language_pair}_{split}.pkl")
        
        # Tải dữ liệu
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['src'], data['tgt']
    
    def create_dataloader(self, src_data, tgt_data, src_lang, tgt_lang, batch_size=32, shuffle=True):
        """
        Tạo DataLoader cho huấn luyện
        
        Args:
            src_data: Dữ liệu nguồn
            tgt_data: Dữ liệu đích
            src_lang: Mã ngôn ngữ nguồn
            tgt_lang: Mã ngôn ngữ đích
            batch_size: Kích thước batch
            shuffle: Xáo trộn dữ liệu
            
        Returns:
            DataLoader
        """
        # Chuyển đổi thành tensor
        src_tensor = torch.tensor(src_data, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_data, dtype=torch.long)
        
        # Tạo tensor ngôn ngữ
        src_lang_id = self.language_ids[src_lang]
        tgt_lang_id = self.language_ids[tgt_lang]
        
        src_lang_tensor = torch.tensor([src_lang_id] * len(src_data), dtype=torch.long)
        tgt_lang_tensor = torch.tensor([tgt_lang_id] * len(tgt_data), dtype=torch.long)
        
        # Tạo dataset
        dataset = TensorDataset(src_tensor, tgt_tensor, src_lang_tensor, tgt_lang_tensor)
        
        # Tạo dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
    
    def create_masks(self, src, tgt):
        """
        Tạo masks cho transformer
        
        Args:
            src: Tensor, shape [batch_size, src_seq_len]
            tgt: Tensor, shape [batch_size, tgt_seq_len]
            
        Returns:
            Tuple (src_padding_mask, combined_mask)
        """
        # Padding mask cho src (1 cho tokens, 0 cho padding)
        src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
        
        # Padding mask cho tgt (1 cho tokens, 0 cho padding)
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)
        
        # Look-ahead mask cho decoder (1 cho tokens được attend, 0 cho tokens bị mask)
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len)), diagonal=1).eq(0)
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1).to(self.device)  # (1, 1, tgt_seq_len, tgt_seq_len)
        
        # Kết hợp padding mask với look-ahead mask
        combined_mask = torch.logical_and(tgt_padding_mask, look_ahead_mask)
        
        return src_padding_mask.to(self.device), combined_mask.to(self.device)
    
    def train_step(self, src, tgt, src_lang_id, tgt_lang_id, optimizer, criterion):
        """
        Một bước huấn luyện
        
        Args:
            src: Tensor, shape [batch_size, src_seq_len]
            tgt: Tensor, shape [batch_size, tgt_seq_len]
            src_lang_id: Tensor, shape [batch_size]
            tgt_lang_id: Tensor, shape [batch_size]
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Loss
        """
        # Đưa dữ liệu vào thiết bị tính toán
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_lang_id = src_lang_id.to(self.device)
        tgt_lang_id = tgt_lang_id.to(self.device)
        
        # Tạo input và output cho decoder
        tgt_input = tgt[:, :-1]  # Loại bỏ token <end>
        tgt_output = tgt[:, 1:]  # Loại bỏ token <start>
        
        # Tạo masks
        src_padding_mask, combined_mask = self.create_masks(src, tgt_input)
        
        # Xóa gradient
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = self.model(
            src=src,
            tgt=tgt_input,
            src_lang=src_lang_id,
            tgt_lang=tgt_lang_id
        )
        
        # Tính loss
        loss = criterion(logits.contiguous().view(-1, logits.size(-1)), tgt_output.contiguous().view(-1))
        
        # Backward pass
        loss.backward()
        
        # Cập nhật tham số
        optimizer.step()
        
        return loss.item()
    
    def validate(self, dataloader, criterion):
        """
        Đánh giá mô hình trên tập validation
        
        Args:
            dataloader: DataLoader
            criterion: Loss function
            
        Returns:
            Validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt, src_lang_id, tgt_lang_id in dataloader:
                # Đưa dữ liệu vào thiết bị tính toán
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_lang_id = src_lang_id.to(self.device)
                tgt_lang_id = tgt_lang_id.to(self.device)
                
                # Tạo input và output cho decoder
                tgt_input = tgt[:, :-1]  # Loại bỏ token <end>
                tgt_output = tgt[:, 1:]  # Loại bỏ token <start>
                
                # Tạo masks
                src_padding_mask, combined_mask = self.create_masks(src, tgt_input)
                
                # Forward pass
                logits, _ = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_lang=src_lang_id,
                    tgt_lang=tgt_lang_id
                )
                
                # Tính loss
                loss = criterion(logits.contiguous().view(-1, logits.size(-1)), tgt_output.contiguous().view(-1))
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, language_pairs, batch_size=32, epochs=10, lr=0.0001, save_every=1):
        """
        Huấn luyện mô hình
        
        Args:
            language_pairs: Danh sách cặp ngôn ngữ [('en', 'fr'), ('en', 'es'), ('en', 'vi')]
            batch_size: Kích thước batch
            epochs: Số epoch
            lr: Learning rate
            save_every: Lưu mô hình sau mỗi bao nhiêu epoch
            
        Returns:
            History
        """
        # Tạo optimizer và criterion
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Lưu lịch sử huấn luyện
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Tạo dataloaders cho từng cặp ngôn ngữ
        train_dataloaders = []
        val_dataloaders = []
        
        for src_lang, tgt_lang in language_pairs:
            # Tạo tên cặp ngôn ngữ
            lang_pair = f"{src_lang}_{tgt_lang}"
            
            # Tải dữ liệu
            train_src, train_tgt = self.load_data(lang_pair, 'train')
            val_src, val_tgt = self.load_data(lang_pair, 'val')
            
            # Tạo dataloaders
            train_dataloader = self.create_dataloader(train_src, train_tgt, src_lang, tgt_lang, batch_size)
            val_dataloader = self.create_dataloader(val_src, val_tgt, src_lang, tgt_lang, batch_size, shuffle=False)
            
            train_dataloaders.append(train_dataloader)
            val_dataloaders.append(val_dataloader)
        
        # Huấn luyện
        for epoch in range(epochs):
            start_time = time.time()
            
            # Huấn luyện
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            # Huấn luyện trên từng cặp ngôn ngữ
            for i, (src_lang, tgt_lang) in enumerate(language_pairs):
                dataloader = train_dataloaders[i]
                
                for src, tgt, src_lang_id, tgt_lang_id in dataloader:
                    batch_loss = self.train_step(src, tgt, src_lang_id, tgt_lang_id, optimizer, criterion)
                    train_loss += batch_loss
                    train_steps += 1
            
            # Tính loss trung bình
            train_loss /= train_steps
            
            # Đánh giá trên tập validation
            val_loss = 0
            val_steps = 0
            
            # Đánh giá trên từng cặp ngôn ngữ
            for i, (src_lang, tgt_lang) in enumerate(language_pairs):
                dataloader = val_dataloaders[i]
                
                val_loss_i = self.validate(dataloader, criterion)
                val_loss += val_loss_i
                val_steps += 1
            
            # Tính loss trung bình
            val_loss /= val_steps
            
            # Lưu lịch sử
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # In thông tin
            print(f'Epoch {epoch + 1}/{epochs} - {time.time() - start_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Lưu mô hình
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                model_path = os.path.join(self.output_dir, f'model_epoch_{epoch + 1}.pt')
                torch.save(self.model.state_dict(), model_path)
                print(f'Đã lưu mô hình: {model_path}')
        
        return history
    
    def translate(self, text, src_lang=None, tgt_lang=None, max_length=100):
        """
        Dịch văn bản
        
        Args:
            text: Văn bản cần dịch
            src_lang: Mã ngôn ngữ nguồn (tự động phát hiện nếu không cung cấp)
            tgt_lang: Mã ngôn ngữ đích (mặc định là 'en' nếu không cung cấp)
            max_length: Độ dài tối đa của chuỗi token đầu ra
            
        Returns:
            Văn bản đã dịch
        """
        # Tạo bộ phát hiện ngôn ngữ
        detector = LanguageDetector()
        
        # Phát hiện ngôn ngữ nguồn nếu không được cung cấp
        if not src_lang:
            src_lang = detector.detect_language(text)
        
        # Mặc định ngôn ngữ đích là tiếng Anh nếu không được cung cấp
        if not tgt_lang:
            tgt_lang = 'en' if src_lang != 'en' else 'fr'
        
        # Chuẩn hóa văn bản
        text = text.strip()
        
        # Tokenize văn bản
        token_ids = self.sp.encode(text, out_type=int)
        
        # Thêm token <s> và </s>
        token_ids = [2] + token_ids + [3]  # <s> và </s>
        
        # Chuyển đổi thành tensor
        src = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        # Lấy ID ngôn ngữ
        src_lang_id = torch.tensor([self.language_ids[src_lang]], dtype=torch.long).to(self.device)
        tgt_lang_id = torch.tensor([self.language_ids[tgt_lang]], dtype=torch.long).to(self.device)
        
        # Dịch văn bản
        self.model.eval()
        with torch.no_grad():
            output_ids, _ = self.model.translate(
                src=src,
                src_lang=src_lang_id,
                tgt_lang=tgt_lang_id,
                max_length=max_length
            )
        
        # Detokenize kết quả
        output_text = self.sp.decode(output_ids[0].cpu().numpy().tolist())
        
        return output_text

def save_data_for_training(data_dir, output_dir):
    """
    Chuẩn bị và lưu dữ liệu cho huấn luyện
    
    Args:
        data_dir: Thư mục chứa dữ liệu đã tiền xử lý
        output_dir: Thư mục lưu dữ liệu đã chuẩn bị
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tải dữ liệu đã tiền xử lý
    with open(os.path.join(data_dir, 'all_data.pkl'), 'rb') as f:
        all_data = pickle.load(f)
    
    # Lưu dữ liệu cho từng cặp ngôn ngữ
    for lang_pair, data in all_data.items():
        for split in ['train', 'val', 'test']:
            src_data, tgt_data = data[split]
            
            # Tạo dict dữ liệu
            split_data = {
                'src': src_data,
                'tgt': tgt_data
            }
            
            # Lưu dữ liệu
            output_file = os.path.join(output_dir, f"{lang_pair}_{split}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
            
            print(f"Saved data {lang_pair}_{split}: {src_data.shape}")

def main():
    """
    Hàm chính để huấn luyện mô hình
    """
    # Đường dẫn
    data_dir = './processed'
    output_dir = './models'
    sp_model_path = './processed/spm_model.model'
    
    # Chuẩn bị dữ liệu cho huấn luyện
    save_data_for_training(data_dir, output_dir)
    
    # Tải mô hình SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    
    # Lấy kích thước từ vựng
    vocab_size = sp.get_piece_size()
    
    # Tạo mô hình
    model = create_multilingual_transformer(
        vocab_size=vocab_size,
        num_languages=4,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=3,
        max_seq_length=100,
        dropout=0.1
    )
    
    # Tạo bộ huấn luyện
    trainer = MultilingualTrainer(
        model=model,
        sp_model_path=sp_model_path,
        data_dir=output_dir,
        output_dir=output_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train the model
    language_pairs = [('en', 'fr'), ('en', 'es'), ('en', 'vi')]
    history = trainer.train(
        language_pairs=language_pairs,
        batch_size=32,
        epochs=10,
        lr=0.0001,
        save_every=1
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
