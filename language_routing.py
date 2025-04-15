import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import unicodedata
import sentencepiece as spm
from collections import Counter

class LanguageDetector:
    """
    Lớp phát hiện ngôn ngữ dựa trên đặc trưng ngôn ngữ
    Hỗ trợ 4 ngôn ngữ: Anh (en), Pháp (fr), Tây Ban Nha (es), Việt (vi)
    """
    
    def __init__(self):
        # Đặc trưng ngôn ngữ: các từ và ký tự đặc biệt thường xuất hiện trong mỗi ngôn ngữ
        self.language_features = {
            'en': {
                'chars': set("abcdefghijklmnopqrstuvwxyz"),
                'words': set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'it', 'for']),
                'patterns': [r'\bthe\b', r'\band\b', r'\bto\b', r'\bof\b', r'\ba\b', r'\bin\b']
            },
            'fr': {
                'chars': set("abcdefghijklmnopqrstuvwxyzàâæçéèêëîïôœùûüÿ"),
                'words': set(['le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'que', 'pour']),
                'patterns': [r'\ble\b', r'\bla\b', r'\bles\b', r'\bun\b', r'\bune\b', r'\bdes\b', r'\best\b']
            },
            'es': {
                'chars': set("abcdefghijklmnopqrstuvwxyzáéíóúüñ¿¡"),
                'words': set(['el', 'la', 'los', 'las', 'un', 'una', 'y', 'que', 'es', 'en']),
                'patterns': [r'\bel\b', r'\bla\b', r'\blos\b', r'\blas\b', r'\by\b', r'\bque\b', r'\bes\b']
            },
            'vi': {
                'chars': set("abcdefghijklmnopqrstuvwxyzàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"),
                'words': set(['của', 'và', 'là', 'có', 'không', 'trong', 'đã', 'được', 'một', 'người']),
                'patterns': [r'\bcủa\b', r'\bvà\b', r'\blà\b', r'\bcó\b', r'\bkhông\b', r'\btrong\b', r'\bđã\b']
            }
        }
        
        # Trọng số cho các loại đặc trưng
        self.weights = {
            'char_ratio': 0.3,
            'word_match': 0.4,
            'pattern_match': 0.3
        }
    
    def normalize_text(self, text):
        """
        Chuẩn hóa văn bản: chuyển thành chữ thường, loại bỏ dấu câu
        """
        # Chuyển thành chữ thường
        text = text.lower()
        
        # Loại bỏ dấu câu (giữ lại dấu trong các ký tự đặc biệt)
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_char_ratio(self, text, language):
        """
        Tính tỷ lệ ký tự thuộc ngôn ngữ
        """
        if not text:
            return 0
        
        chars = set(text)
        lang_chars = self.language_features[language]['chars']
        
        # Đếm số ký tự thuộc ngôn ngữ
        matched_chars = chars.intersection(lang_chars)
        
        # Tính tỷ lệ
        return len(matched_chars) / len(chars) if chars else 0
    
    def calculate_word_match(self, words, language):
        """
        Tính tỷ lệ từ thuộc ngôn ngữ
        """
        if not words:
            return 0
        
        lang_words = self.language_features[language]['words']
        
        # Đếm số từ thuộc ngôn ngữ
        matched_words = sum(1 for word in words if word in lang_words)
        
        # Tính tỷ lệ
        return matched_words / len(words) if words else 0
    
    def calculate_pattern_match(self, text, language):
        """
        Tính số lượng mẫu ngôn ngữ xuất hiện trong văn bản
        """
        if not text:
            return 0
        
        patterns = self.language_features[language]['patterns']
        
        # Đếm số mẫu xuất hiện
        matched_patterns = sum(1 for pattern in patterns if re.search(pattern, text))
        
        # Tính tỷ lệ
        return matched_patterns / len(patterns) if patterns else 0
    
    def detect_language(self, text):
        """
        Phát hiện ngôn ngữ của văn bản
        
        Args:
            text: Văn bản cần phát hiện ngôn ngữ
            
        Returns:
            Mã ngôn ngữ ('en', 'fr', 'es', 'vi')
        """
        if not text:
            return 'en'  # Mặc định là tiếng Anh nếu không có văn bản
        
        # Chuẩn hóa văn bản
        normalized_text = self.normalize_text(text)
        words = normalized_text.split()
        
        # Tính điểm cho từng ngôn ngữ
        scores = {}
        
        for lang in self.language_features.keys():
            # Tính điểm cho từng loại đặc trưng
            char_ratio = self.calculate_char_ratio(normalized_text, lang)
            word_match = self.calculate_word_match(words, lang)
            pattern_match = self.calculate_pattern_match(normalized_text, lang)
            
            # Tính điểm tổng hợp
            score = (
                self.weights['char_ratio'] * char_ratio +
                self.weights['word_match'] * word_match +
                self.weights['pattern_match'] * pattern_match
            )
            
            scores[lang] = score
        
        # Trả về ngôn ngữ có điểm cao nhất
        return max(scores.items(), key=lambda x: x[1])[0]

class LanguageRouter:
    """
    Lớp định tuyến ngôn ngữ cho mô hình dịch máy đa ngôn ngữ
    """
    
    def __init__(self, language_detector=None, sp_model_path=None):
        """
        Khởi tạo bộ định tuyến ngôn ngữ
        
        Args:
            language_detector: Đối tượng LanguageDetector
            sp_model_path: Đường dẫn đến mô hình SentencePiece
        """
        self.language_detector = language_detector or LanguageDetector()
        
        # Tải mô hình SentencePiece nếu được cung cấp
        self.sp = None
        if sp_model_path:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(sp_model_path)
        
        # Ánh xạ mã ngôn ngữ sang token ngôn ngữ
        self.language_tokens = {
            'en': '<en>',
            'fr': '<fr>',
            'es': '<es>',
            'vi': '<vi>'
        }
        
        # Ánh xạ mã ngôn ngữ sang ID ngôn ngữ
        self.language_ids = {
            'en': 0,
            'fr': 1,
            'es': 2,
            'vi': 3
        }
    
    def detect_language(self, text):
        """
        Phát hiện ngôn ngữ của văn bản
        
        Args:
            text: Văn bản cần phát hiện ngôn ngữ
            
        Returns:
            Mã ngôn ngữ ('en', 'fr', 'es', 'vi')
        """
        return self.language_detector.detect_language(text)
    
    def get_language_token(self, language):
        """
        Lấy token ngôn ngữ từ mã ngôn ngữ
        
        Args:
            language: Mã ngôn ngữ ('en', 'fr', 'es', 'vi')
            
        Returns:
            Token ngôn ngữ
        """
        return self.language_tokens.get(language, '<en>')
    
    def get_language_id(self, language):
        """
        Lấy ID ngôn ngữ từ mã ngôn ngữ
        
        Args:
            language: Mã ngôn ngữ ('en', 'fr', 'es', 'vi')
            
        Returns:
            ID ngôn ngữ
        """
        return self.language_ids.get(language, 0)
    
    def preprocess_for_translation(self, text, src_lang=None, tgt_lang=None):
        """
        Tiền xử lý văn bản cho dịch thuật
        
        Args:
            text: Văn bản cần dịch
            src_lang: Mã ngôn ngữ nguồn (tự động phát hiện nếu không cung cấp)
            tgt_lang: Mã ngôn ngữ đích (mặc định là 'en' nếu không cung cấp)
            
        Returns:
            Tuple (văn bản đã xử lý, mã ngôn ngữ nguồn, mã ngôn ngữ đích)
        """
        # Phát hiện ngôn ngữ nguồn nếu không được cung cấp
        if not src_lang:
            src_lang = self.detect_language(text)
        
        # Mặc định ngôn ngữ đích là tiếng Anh nếu không được cung cấp
        if not tgt_lang:
            tgt_lang = 'en' if src_lang != 'en' else 'fr'
        
        # Chuẩn hóa văn bản
        text = text.strip()
        
        # Thêm token ngôn ngữ nếu cần
        if self.sp and not text.startswith(self.get_language_token(src_lang)):
            text = f"{self.get_language_token(src_lang)} {text}"
        
        return text, src_lang, tgt_lang
    
    def tokenize(self, text):
        """
        Tokenize văn bản sử dụng SentencePiece
        
        Args:
            text: Văn bản cần tokenize
            
        Returns:
            Danh sách ID token
        """
        if not self.sp:
            raise ValueError("SentencePiece model not loaded")
        
        return self.sp.encode(text, out_type=int)
    
    def detokenize(self, ids):
        """
        Detokenize từ danh sách ID token
        
        Args:
            ids: Danh sách ID token
            
        Returns:
            Văn bản
        """
        if not self.sp:
            raise ValueError("SentencePiece model not loaded")
        
        return self.sp.decode(ids)
    
    def prepare_batch(self, texts, src_lang=None, tgt_lang=None, max_length=100):
        """
        Chuẩn bị batch dữ liệu cho mô hình dịch thuật
        
        Args:
            texts: Danh sách văn bản cần dịch
            src_lang: Mã ngôn ngữ nguồn (tự động phát hiện nếu không cung cấp)
            tgt_lang: Mã ngôn ngữ đích (mặc định là 'en' nếu không cung cấp)
            max_length: Độ dài tối đa của chuỗi token
            
        Returns:
            Tuple (tensor đầu vào, mã ngôn ngữ nguồn, mã ngôn ngữ đích)
        """
        if not self.sp:
            raise ValueError("SentencePiece model not loaded")
        
        processed_texts = []
        src_langs = []
        tgt_langs = []
        
        # Xử lý từng văn bản
        for text in texts:
            # Tiền xử lý văn bản
            processed_text, src_lang_i, tgt_lang_i = self.preprocess_for_translation(text, src_lang, tgt_lang)
            
            # Tokenize văn bản
            token_ids = self.tokenize(processed_text)
            
            # Cắt bớt nếu quá dài
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            processed_texts.append(token_ids)
            src_langs.append(src_lang_i)
            tgt_langs.append(tgt_lang_i)
        
        # Đệm các chuỗi token để có cùng độ dài
        max_len = max(len(ids) for ids in processed_texts)
        padded_texts = []
        
        for ids in processed_texts:
            padded_ids = ids + [0] * (max_len - len(ids))
            padded_texts.append(padded_ids)
        
        # Chuyển đổi thành tensor
        input_tensor = torch.tensor(padded_texts)
        
        # Lấy ID ngôn ngữ nguồn và đích
        src_lang_ids = [self.get_language_id(lang) for lang in src_langs]
        tgt_lang_ids = [self.get_language_id(lang) for lang in tgt_langs]
        
        return input_tensor, src_lang_ids, tgt_lang_ids

class MultilingualTranslationPipeline:
    """
    Pipeline dịch thuật đa ngôn ngữ
    """
    
    def __init__(self, model, router, device='cpu'):
        """
        Khởi tạo pipeline dịch thuật
        
        Args:
            model: Mô hình MultilingualTransformer
            router: Đối tượng LanguageRouter
            device: Thiết bị tính toán ('cpu' hoặc 'cuda')
        """
        self.model = model
        self.router = router
        self.device = device
        
        # Chuyển mô hình sang thiết bị tính toán
        self.model = self.model.to(device)
    
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
        # Chuẩn bị dữ liệu đầu vào
        input_tensor, src_lang_ids, tgt_lang_ids = self.router.prepare_batch([text], src_lang, tgt_lang)
        input_tensor = input_tensor.to(self.device)
        
        # Dịch văn bản
        with torch.no_grad():
            output_ids, _ = self.model.translate(
                input_tensor, 
                src_lang=src_lang_ids[0], 
                tgt_lang=tgt_lang_ids[0],
                max_length=max_length
            )
        
        # Detokenize kết quả
        output_text = self.router.detokenize(output_ids[0].cpu().numpy().tolist())
        
        # Loại bỏ token ngôn ngữ đích nếu có
        tgt_token = self.router.get_language_token(tgt_lang_ids[0])
        if output_text.startswith(tgt_token):
            output_text = output_text[len(tgt_token):].strip()
        
        return output_text
    
    def batch_translate(self, texts, src_lang=None, tgt_lang=None, max_length=100):
        """
        Dịch một batch văn bản
        
        Args:
            texts: Danh sách văn bản cần dịch
            src_lang: Mã ngôn ngữ nguồn (tự động phát hiện nếu không cung cấp)
            tgt_lang: Mã ngôn ngữ đích (mặc định là 'en' nếu không cung cấp)
            max_length: Độ dài tối đa của chuỗi token đầu ra
            
        Returns:
            Danh sách văn bản đã dịch
        """
        # Chuẩn bị dữ liệu đầu vào
        input_tensor, src_lang_ids, tgt_lang_ids = self.router.prepare_batch(texts, src_lang, tgt_lang)
        input_tensor = input_tensor.to(self.device)
        
        # Dịch văn bản
        with torch.no_grad():
            output_ids, _ = self.model.translate(
                input_tensor, 
                src_lang=src_lang_ids, 
                tgt_lang=tgt_lang_ids,
                max_length=max_length
            )
        
        # Detokenize kết quả
        output_texts = []
        for i, ids in enumerate(output_ids):
            output_text = self.router.detokenize(ids.cpu().numpy().tolist())
            
            # Loại bỏ token ngôn ngữ đích nếu có
            tgt_token = self.router.get_language_token(tgt_lang_ids[i])
            if output_text.startswith(tgt_token):
                output_text = output_text[len(tgt_token):].strip()
            
            output_texts.append(output_text)
        
        return output_texts

# Hàm tiện ích để tạo pipeline dịch thuật đa ngôn ngữ
def create_translation_pipeline(model_path, sp_model_path, device='cpu'):
    """
    Tạo pipeline dịch thuật đa ngôn ngữ
    
    Args:
        model_path: Đường dẫn đến mô hình MultilingualTransformer
        sp_model_path: Đường dẫn đến mô hình SentencePiece
        device: Thiết bị tính toán ('cpu' hoặc 'cuda')
        
    Returns:
        Đối tượng MultilingualTranslationPipeline
    """
    # Tải mô hình SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    
    # Tạo bộ phát hiện và định tuyến ngôn ngữ
    detector = LanguageDetector()
    router = LanguageRouter(detector, sp_model_path)
    
    # Tạo mô hình MultilingualTransformer
    from multilingual_transformer import create_multilingual_transformer
    
    # Lấy kích thước từ vựng từ mô hình SentencePiece
    vocab_size = sp.get_piece_size()
    
    # Tạo mô hình với kích thước nhỏ hơn để tiết kiệm bộ nhớ
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
    
    # Tải trọng số mô hình nếu tồn tại
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Tạo pipeline dịch thuật
    pipeline = MultilingualTranslationPipeline(model, router, device)
    
    return pipeline
