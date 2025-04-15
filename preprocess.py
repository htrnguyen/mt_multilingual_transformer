import os
import re
import sys
import unicodedata
import sentencepiece as spm
import numpy as np
from sklearn.model_selection import train_test_split

# Set console output encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

class MultilingualPreprocessor:
    """
    Lớp tiền xử lý đa ngôn ngữ cho mô hình dịch máy transformer
    Hỗ trợ 4 ngôn ngữ: Anh (en), Pháp (fr), Tây Ban Nha (es), Việt (vi)
    """
    
    def __init__(self, data_dir, output_dir, vocab_size=32000, character_coverage=0.9995):
        """
        Khởi tạo bộ tiền xử lý đa ngôn ngữ
        
        Args:
            data_dir: Thư mục chứa dữ liệu gốc
            output_dir: Thư mục lưu dữ liệu đã xử lý và mô hình tokenizer
            vocab_size: Kích thước từ vựng cho SentencePiece
            character_coverage: Độ phủ ký tự cho SentencePiece
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Định nghĩa token đặc biệt cho từng ngôn ngữ
        self.lang_tokens = {
            'en': '<en>',
            'fr': '<fr>',
            'es': '<es>',
            'vi': '<vi>'
        }
        
        # Định nghĩa token đặc biệt khác
        self.special_tokens = {
            'pad': '<pad>',
            'unk': '<unk>',
            'bos': '<s>',
            'eos': '</s>'
        }
        
        # Đường dẫn đến các file dữ liệu
        self.data_files = {
            'fr': os.path.join(data_dir, 'extracted/fra-eng/fra.txt'),
            'es': os.path.join(data_dir, 'extracted/spa-eng/spa.txt'),
            'vi': os.path.join(data_dir, 'extracted/vie-eng/vie.txt')
        }
        
        # Đường dẫn đến các file đã tiền xử lý
        self.processed_files = {
            'fr': os.path.join(output_dir, 'processed_fr.txt'),
            'es': os.path.join(output_dir, 'processed_es.txt'),
            'vi': os.path.join(output_dir, 'processed_vi.txt'),
            'en': os.path.join(output_dir, 'processed_en.txt'),
            'all': os.path.join(output_dir, 'processed_all.txt')
        }
        
        # Đường dẫn đến mô hình SentencePiece
        self.spm_model_prefix = os.path.join(output_dir, 'spm_model')
        self.spm_model_file = self.spm_model_prefix + '.model'
        
    def unicode_to_ascii(self, s):
        """
        Chuyển đổi unicode thành ASCII cho tiếng Anh
        Giữ nguyên dấu cho các ngôn ngữ khác
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    
    def normalize_english(self, s):
        """
        Chuẩn hóa văn bản tiếng Anh
        """
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s
    
    def normalize_french(self, s):
        """
        Chuẩn hóa văn bản tiếng Pháp
        """
        s = s.lower().strip()
        
        # Xử lý dấu câu đặc biệt của tiếng Pháp
        s = re.sub(r"\s+([.!?:;,])", r"\1", s)
        s = re.sub(r"([.!?])", r" \1", s)
        
        # Xử lý elision
        s = re.sub(r"l'(\w)", r"l' \1", s)
        s = re.sub(r"d'(\w)", r"d' \1", s)
        s = re.sub(r"j'(\w)", r"j' \1", s)
        s = re.sub(r"c'(\w)", r"c' \1", s)
        s = re.sub(r"n'(\w)", r"n' \1", s)
        
        # Loại bỏ khoảng trắng thừa
        s = re.sub(r"\s+", r" ", s).strip()
        return s
    
    def normalize_spanish(self, s):
        """
        Chuẩn hóa văn bản tiếng Tây Ban Nha
        """
        s = s.lower().strip()
        
        # Xử lý dấu câu đặc biệt của tiếng Tây Ban Nha
        s = re.sub(r"¿", r" ¿ ", s)
        s = re.sub(r"¡", r" ¡ ", s)
        s = re.sub(r"([.!?])", r" \1", s)
        
        # Loại bỏ khoảng trắng thừa
        s = re.sub(r"\s+", r" ", s).strip()
        return s
    
    def normalize_vietnamese(self, s):
        """
        Chuẩn hóa văn bản tiếng Việt
        """
        s = s.lower().strip()
        
        # Xử lý dấu câu
        s = re.sub(r"([.!?])", r" \1", s)
        
        # Loại bỏ khoảng trắng thừa
        s = re.sub(r"\s+", r" ", s).strip()
        return s
    
    def preprocess_sentence(self, sentence, lang):
        """
        Tiền xử lý câu dựa trên ngôn ngữ
        
        Args:
            sentence: Câu cần tiền xử lý
            lang: Mã ngôn ngữ ('en', 'fr', 'es', 'vi')
            
        Returns:
            Câu đã được tiền xử lý
        """
        if lang == 'en':
            return self.normalize_english(sentence)
        elif lang == 'fr':
            return self.normalize_french(sentence)
        elif lang == 'es':
            return self.normalize_spanish(sentence)
        elif lang == 'vi':
            return self.normalize_vietnamese(sentence)
        else:
            raise ValueError(f"Ngôn ngữ không được hỗ trợ: {lang}")
    
    def extract_sentence_pairs(self, file_path, src_lang='en', tgt_lang=None):
        """
        Trích xuất các cặp câu từ file dữ liệu
        
        Args:
            file_path: Đường dẫn đến file dữ liệu
            src_lang: Ngôn ngữ nguồn (mặc định là 'en')
            tgt_lang: Ngôn ngữ đích
            
        Returns:
            Danh sách các cặp câu đã tiền xử lý
        """
        lines = open(file_path, encoding='UTF-8').read().strip().split('\n')
        pairs = []
        
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2:
                src_text, tgt_text = parts[0], parts[1]
                
                # Tiền xử lý câu
                src_text = self.preprocess_sentence(src_text, src_lang)
                tgt_text = self.preprocess_sentence(tgt_text, tgt_lang)
                
                # Thêm token ngôn ngữ
                src_text = f"{self.lang_tokens[src_lang]} {src_text}"
                tgt_text = f"{self.lang_tokens[tgt_lang]} {tgt_text}"
                
                pairs.append((src_text, tgt_text))
        
        return pairs
    
    def process_all_data(self):
        """
        Xử lý tất cả dữ liệu từ các file nguồn
        Tạo các file văn bản đã tiền xử lý
        """
        all_src_sentences = []
        all_tgt_sentences = []
        
        # Xử lý từng cặp ngôn ngữ
        for lang, file_path in self.data_files.items():
            print(f"Đang xử lý dữ liệu {lang}...")
            pairs = self.extract_sentence_pairs(file_path, 'en', lang)
            
            # Lưu các cặp câu đã xử lý
            with open(self.processed_files[lang], 'w', encoding='UTF-8') as f:
                for src, tgt in pairs:
                    f.write(f"{src}\t{tgt}\n")
            
            # Lưu câu tiếng Anh vào file riêng (chỉ lần đầu)
            if lang == 'fr':
                with open(self.processed_files['en'], 'w', encoding='UTF-8') as f:
                    for src, _ in pairs:
                        f.write(f"{src}\n")
            
            # Thêm vào danh sách tổng hợp
            all_src_sentences.extend([src for src, _ in pairs])
            all_tgt_sentences.extend([tgt for _, tgt in pairs])
        
        # Lưu tất cả câu vào một file để huấn luyện SentencePiece
        with open(self.processed_files['all'], 'w', encoding='UTF-8') as f:
            for sentence in all_src_sentences + all_tgt_sentences:
                f.write(f"{sentence}\n")
        
        print(f"Đã xử lý tổng cộng {len(all_src_sentences)} cặp câu.")
        return all_src_sentences, all_tgt_sentences
    
    def train_sentencepiece(self):
        """
        Huấn luyện mô hình SentencePiece trên tất cả dữ liệu
        """
        print(f"Đang huấn luyện SentencePiece với kích thước từ vựng {self.vocab_size}...")
        
        # Tạo file cấu hình huấn luyện
        spm_command = f'--input={self.processed_files["all"]} '
        spm_command += f'--model_prefix={self.spm_model_prefix} '
        spm_command += f'--vocab_size={self.vocab_size} '
        spm_command += f'--character_coverage={self.character_coverage} '
        spm_command += '--model_type=unigram '
        spm_command += f'--user_defined_symbols={",".join(self.lang_tokens.values())} '
        spm_command += '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
        spm_command += '--input_sentence_size=1000000 '
        spm_command += '--shuffle_input_sentence=true'
        
        # Huấn luyện mô hình
        spm.SentencePieceTrainer.train(spm_command)
        print(f"Đã huấn luyện xong mô hình SentencePiece: {self.spm_model_file}")
    
    def load_sentencepiece(self):
        """
        Tải mô hình SentencePiece đã huấn luyện
        
        Returns:
            Đối tượng SentencePiece processor
        """
        if not os.path.exists(self.spm_model_file):
            raise FileNotFoundError(f"Không tìm thấy mô hình SentencePiece: {self.spm_model_file}")
        
        sp = spm.SentencePieceProcessor()
        sp.load(self.spm_model_file)
        return sp
    
    def prepare_training_data(self, test_size=0.1, val_size=0.1, max_len=100):
        """
        Chuẩn bị dữ liệu huấn luyện, kiểm tra và đánh giá
        
        Args:
            test_size: Tỷ lệ dữ liệu kiểm tra
            val_size: Tỷ lệ dữ liệu đánh giá
            max_len: Độ dài tối đa của câu sau khi tokenize
            
        Returns:
            Dữ liệu đã được tokenize và chia tập
        """
        # Tải mô hình SentencePiece
        sp = self.load_sentencepiece()
        
        # Chuẩn bị dữ liệu cho từng cặp ngôn ngữ
        all_data = {}
        
        for lang in ['fr', 'es', 'vi']:
            print(f"Đang chuẩn bị dữ liệu cho cặp en-{lang}...")
            
            # Đọc các cặp câu đã xử lý
            pairs = []
            with open(self.processed_files[lang], 'r', encoding='UTF-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        pairs.append((parts[0], parts[1]))
            
            # Tokenize các cặp câu
            src_tokens = []
            tgt_tokens = []
            
            for src, tgt in pairs:
                src_ids = sp.encode(src, out_type=int)
                tgt_ids = sp.encode(tgt, out_type=int)
                
                # Thêm token BOS và EOS
                src_ids = [sp.bos_id()] + src_ids + [sp.eos_id()]
                tgt_ids = [sp.bos_id()] + tgt_ids + [sp.eos_id()]
                
                # Cắt bớt nếu quá dài
                if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                    src_tokens.append(src_ids)
                    tgt_tokens.append(tgt_ids)
            
            # Đệm các câu để có cùng độ dài
            max_src_len = max(len(seq) for seq in src_tokens)
            max_tgt_len = max(len(seq) for seq in tgt_tokens)
            
            # Tạo mảng numpy với padding
            src_data = np.zeros((len(src_tokens), max_src_len), dtype=np.int32)
            tgt_data = np.zeros((len(tgt_tokens), max_tgt_len), dtype=np.int32)
            
            # Điền dữ liệu vào mảng
            for i, seq in enumerate(src_tokens):
                src_data[i, :len(seq)] = seq
            
            for i, seq in enumerate(tgt_tokens):
                tgt_data[i, :len(seq)] = seq
            
            # Chia tập dữ liệu
            src_train, src_temp, tgt_train, tgt_temp = train_test_split(
                src_data, tgt_data, test_size=(test_size + val_size), random_state=42
            )
            
            # Chia tập temp thành val và test
            val_ratio = val_size / (test_size + val_size)
            src_val, src_test, tgt_val, tgt_test = train_test_split(
                src_temp, tgt_temp, test_size=(1 - val_ratio), random_state=42
            )
            
            # Lưu dữ liệu
            all_data[f'en_{lang}'] = {
                'train': (src_train, tgt_train),
                'val': (src_val, tgt_val),
                'test': (src_test, tgt_test)
            }
            
            print(f"  Tập huấn luyện: {len(src_train)} mẫu")
            print(f"  Tập kiểm tra: {len(src_val)} mẫu")
            print(f"  Tập đánh giá: {len(src_test)} mẫu")
        
        return all_data, sp
    
    def run_full_pipeline(self):
        """
        Chạy toàn bộ quy trình tiền xử lý
        
        Returns:
            Dữ liệu đã được tokenize và chia tập, cùng với mô hình SentencePiece
        """
        # Bước 1: Xử lý tất cả dữ liệu
        self.process_all_data()
        
        # Bước 2: Huấn luyện SentencePiece
        self.train_sentencepiece()
        
        # Bước 3: Chuẩn bị dữ liệu huấn luyện
        all_data, sp = self.prepare_training_data()
        
        # Lưu thông tin về kích thước từ vựng
        vocab_info = {
            'vocab_size': sp.get_piece_size(),
            'pad_id': sp.pad_id(),
            'unk_id': sp.unk_id(),
            'bos_id': sp.bos_id(),
            'eos_id': sp.eos_id(),
            'lang_tokens': {lang: sp.piece_to_id(token) for lang, token in self.lang_tokens.items()}
        }
        
        print(f"Kích thước từ vựng: {vocab_info['vocab_size']}")
        print(f"Token ngôn ngữ: {vocab_info['lang_tokens']}")
        
        return all_data, sp, vocab_info

# Hàm chính để chạy tiền xử lý
def main():
    import zipfile
    import shutil
    import pickle
    
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    extracted_dir = os.path.join(base_dir, 'extracted')
    output_dir = os.path.join(base_dir, 'processed')
    
    # Create directories if they don't exist
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract zip files if needed
    zip_files = {
        'fra-eng': os.path.join(data_dir, 'fra-eng.zip'),
        'spa-eng': os.path.join(data_dir, 'spa-eng.zip'),
        'vie-eng': os.path.join(data_dir, 'vie-eng.zip')
    }
    
    for lang, zip_path in zip_files.items():
        if os.path.exists(zip_path):
            extract_path = os.path.join(extracted_dir, lang)
            if not os.path.exists(extract_path):
                print(f'Extracting {lang} data...')
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
    
    # Create and run preprocessor
    preprocessor = MultilingualPreprocessor(base_dir, output_dir)
    all_data, sp, vocab_info = preprocessor.run_full_pipeline()
    
    # Save the preprocessed data and vocab info
    with open(os.path.join(output_dir, 'all_data.pkl'), 'wb') as f:
        pickle.dump(all_data, f)
    
    with open(os.path.join(output_dir, 'vocab_info.pkl'), 'wb') as f:
        pickle.dump(vocab_info, f)
    
    print('Preprocessing completed successfully!')

if __name__ == "__main__":
    main()
