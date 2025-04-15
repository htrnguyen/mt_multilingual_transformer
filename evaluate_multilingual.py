import os
import torch
import numpy as np
import sentencepiece as spm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from sacrebleu import corpus_bleu as sacrebleu_corpus_bleu
from multilingual_transformer import create_multilingual_transformer
from language_routing import LanguageDetector, LanguageRouter, MultilingualTranslationPipeline
import pickle
import pandas as pd
from tabulate import tabulate

class MultilingualEvaluator:
    """
    Lớp đánh giá mô hình dịch máy đa ngôn ngữ
    """
    
    def __init__(self, model, sp_model_path, data_dir, output_dir, device='cpu'):
        """
        Khởi tạo bộ đánh giá
        
        Args:
            model: Mô hình MultilingualTransformer
            sp_model_path: Đường dẫn đến mô hình SentencePiece
            data_dir: Thư mục chứa dữ liệu đã tiền xử lý
            output_dir: Thư mục lưu kết quả đánh giá
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
        
        # Tạo bộ phát hiện và định tuyến ngôn ngữ
        self.detector = LanguageDetector()
        self.router = LanguageRouter(self.detector, sp_model_path)
        
        # Tạo pipeline dịch thuật
        self.pipeline = MultilingualTranslationPipeline(model, self.router, device)
        
        # Ánh xạ mã ngôn ngữ sang tên ngôn ngữ
        self.language_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'vi': 'Vietnamese'
        }
        
        # Chuyển mô hình sang thiết bị tính toán
        self.model = self.model.to(device)
    
    def load_test_data(self, language_pair):
        """
        Tải dữ liệu kiểm tra
        
        Args:
            language_pair: Cặp ngôn ngữ ('en_fr', 'en_es', 'en_vi')
            
        Returns:
            Tuple (src_data, tgt_data)
        """
        # Đường dẫn đến file dữ liệu
        data_file = os.path.join(self.data_dir, f"{language_pair}_test.pkl")
        
        # Tải dữ liệu
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['src'], data['tgt']
    
    def decode_batch(self, batch_ids):
        """
        Giải mã batch token IDs thành văn bản
        
        Args:
            batch_ids: Batch token IDs
            
        Returns:
            Danh sách văn bản
        """
        texts = []
        
        for ids in batch_ids:
            # Loại bỏ padding và token đặc biệt
            valid_ids = [id for id in ids if id > 3]  # Bỏ qua <pad>, <unk>, <s>, </s>
            
            # Giải mã thành văn bản
            text = self.sp.decode(valid_ids)
            texts.append(text)
        
        return texts
    
    def translate_batch(self, src_texts, src_lang, tgt_lang, max_length=100):
        """
        Dịch một batch văn bản
        
        Args:
            src_texts: Danh sách văn bản nguồn
            src_lang: Mã ngôn ngữ nguồn
            tgt_lang: Mã ngôn ngữ đích
            max_length: Độ dài tối đa của chuỗi token đầu ra
            
        Returns:
            Danh sách văn bản đã dịch
        """
        return self.pipeline.batch_translate(src_texts, src_lang, tgt_lang, max_length)
    
    def calculate_bleu(self, references, hypotheses):
        """
        Tính điểm BLEU
        
        Args:
            references: Danh sách các câu tham chiếu
            hypotheses: Danh sách các câu dự đoán
            
        Returns:
            Điểm BLEU
        """
        # Chuẩn bị dữ liệu cho NLTK BLEU
        tokenized_refs = [[ref.split()] for ref in references]
        tokenized_hyps = [hyp.split() for hyp in hypotheses]
        
        # Tính điểm BLEU
        bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps)
        
        # Tính điểm BLEU với sacrebleu
        sacrebleu_score = sacrebleu_corpus_bleu(hypotheses, [references]).score
        
        return bleu_score, sacrebleu_score
    
    def evaluate_language_pair(self, src_lang, tgt_lang, num_samples=1000):
        """
        Đánh giá mô hình trên một cặp ngôn ngữ
        
        Args:
            src_lang: Mã ngôn ngữ nguồn
            tgt_lang: Mã ngôn ngữ đích
            num_samples: Số lượng mẫu để đánh giá
            
        Returns:
            Dict kết quả đánh giá
        """
        # Tạo tên cặp ngôn ngữ
        lang_pair = f"{src_lang}_{tgt_lang}"
        
        # Tải dữ liệu kiểm tra
        src_data, tgt_data = self.load_test_data(lang_pair)
        
        # Giới hạn số lượng mẫu
        num_samples = min(num_samples, len(src_data))
        src_data = src_data[:num_samples]
        tgt_data = tgt_data[:num_samples]
        
        # Giải mã dữ liệu nguồn và đích
        src_texts = self.decode_batch(src_data)
        ref_texts = self.decode_batch(tgt_data)
        
        # Dịch văn bản
        hyp_texts = self.translate_batch(src_texts, src_lang, tgt_lang)
        
        # Tính điểm BLEU
        bleu_score, sacrebleu_score = self.calculate_bleu(ref_texts, hyp_texts)
        
        # Lưu kết quả
        results = {
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'num_samples': num_samples,
            'bleu': bleu_score,
            'sacrebleu': sacrebleu_score,
            'examples': []
        }
        
        # Lưu một số ví dụ
        num_examples = min(10, num_samples)
        for i in range(num_examples):
            results['examples'].append({
                'src': src_texts[i],
                'ref': ref_texts[i],
                'hyp': hyp_texts[i]
            })
        
        return results
    
    def evaluate_all_language_pairs(self, language_pairs, num_samples=1000):
        """
        Đánh giá mô hình trên tất cả các cặp ngôn ngữ
        
        Args:
            language_pairs: Danh sách cặp ngôn ngữ [('en', 'fr'), ('en', 'es'), ...]
            num_samples: Số lượng mẫu để đánh giá
            
        Returns:
            Dict kết quả đánh giá
        """
        self.model.eval()
        all_results = {}
        
        for src_lang, tgt_lang in language_pairs:
            print(f"Đánh giá cặp ngôn ngữ: {src_lang} -> {tgt_lang}")
            
            # Đánh giá cặp ngôn ngữ
            results = self.evaluate_language_pair(src_lang, tgt_lang, num_samples)
            
            # Lưu kết quả
            pair_name = f"{src_lang}_{tgt_lang}"
            all_results[pair_name] = results
            
            # In kết quả
            print(f"BLEU: {results['bleu']:.4f}, sacreBLEU: {results['sacrebleu']:.4f}")
        
        return all_results
    
    def save_results(self, results, filename='evaluation_results.pkl'):
        """
        Lưu kết quả đánh giá
        
        Args:
            results: Dict kết quả đánh giá
            filename: Tên file để lưu
        """
        # Đường dẫn đến file kết quả
        output_file = os.path.join(self.output_dir, filename)
        
        # Lưu kết quả
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Đã lưu kết quả đánh giá: {output_file}")
    
    def generate_report(self, results, filename='evaluation_report.md'):
        """
        Tạo báo cáo đánh giá
        
        Args:
            results: Dict kết quả đánh giá
            filename: Tên file báo cáo
        """
        # Đường dẫn đến file báo cáo
        output_file = os.path.join(self.output_dir, filename)
        
        # Tạo báo cáo
        report = "# Báo cáo đánh giá mô hình dịch máy đa ngôn ngữ\n\n"
        
        # Bảng tổng hợp kết quả
        report += "## Kết quả tổng hợp\n\n"
        
        # Tạo bảng kết quả
        table_data = []
        for pair_name, pair_results in results.items():
            src_lang = pair_results['src_lang']
            tgt_lang = pair_results['tgt_lang']
            
            table_data.append([
                f"{self.language_names[src_lang]} -> {self.language_names[tgt_lang]}",
                pair_results['num_samples'],
                f"{pair_results['bleu']:.4f}",
                f"{pair_results['sacrebleu']:.4f}"
            ])
        
        # Tạo bảng Markdown
        headers = ["Cặp ngôn ngữ", "Số mẫu", "BLEU", "sacreBLEU"]
        table = tabulate(table_data, headers=headers, tablefmt="pipe")
        
        report += table + "\n\n"
        
        # Ví dụ dịch thuật
        report += "## Ví dụ dịch thuật\n\n"
        
        for pair_name, pair_results in results.items():
            src_lang = pair_results['src_lang']
            tgt_lang = pair_results['tgt_lang']
            
            report += f"### {self.language_names[src_lang]} -> {self.language_names[tgt_lang]}\n\n"
            
            for i, example in enumerate(pair_results['examples']):
                report += f"**Ví dụ {i+1}:**\n\n"
                report += f"- **Nguồn ({self.language_names[src_lang]}):** {example['src']}\n"
                report += f"- **Tham chiếu ({self.language_names[tgt_lang]}):** {example['ref']}\n"
                report += f"- **Dự đoán ({self.language_names[tgt_lang]}):** {example['hyp']}\n\n"
        
        # Lưu báo cáo
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Đã tạo báo cáo đánh giá: {output_file}")
    
    def plot_bleu_scores(self, results, filename='bleu_scores.png'):
        """
        Vẽ biểu đồ điểm BLEU
        
        Args:
            results: Dict kết quả đánh giá
            filename: Tên file biểu đồ
        """
        # Đường dẫn đến file biểu đồ
        output_file = os.path.join(self.output_dir, filename)
        
        # Chuẩn bị dữ liệu
        pair_names = []
        bleu_scores = []
        sacrebleu_scores = []
        
        for pair_name, pair_results in results.items():
            src_lang = pair_results['src_lang']
            tgt_lang = pair_results['tgt_lang']
            
            pair_names.append(f"{src_lang} -> {tgt_lang}")
            bleu_scores.append(pair_results['bleu'])
            sacrebleu_scores.append(pair_results['sacrebleu'])
        
        # Tạo biểu đồ
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(pair_names))
        width = 0.35
        
        plt.bar(x - width/2, bleu_scores, width, label='BLEU')
        plt.bar(x + width/2, sacrebleu_scores, width, label='sacreBLEU')
        
        plt.xlabel('Cặp ngôn ngữ')
        plt.ylabel('Điểm BLEU')
        plt.title('Điểm BLEU cho các cặp ngôn ngữ')
        plt.xticks(x, pair_names, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_file)
        
        print(f"Đã tạo biểu đồ điểm BLEU: {output_file}")
    
    def create_cross_lingual_matrix(self, results, filename='cross_lingual_matrix.png'):
        """
        Tạo ma trận hiệu suất dịch chéo giữa các ngôn ngữ
        
        Args:
            results: Dict kết quả đánh giá
            filename: Tên file ma trận
        """
        # Đường dẫn đến file ma trận
        output_file = os.path.join(self.output_dir, filename)
        
        # Danh sách ngôn ngữ
        languages = list(self.language_names.keys())
        
        # Tạo ma trận
        matrix = np.zeros((len(languages), len(languages)))
        
        # Điền dữ liệu vào ma trận
        for i, src_lang in enumerate(languages):
            for j, tgt_lang in enumerate(languages):
                if src_lang != tgt_lang:
                    pair_name = f"{src_lang}_{tgt_lang}"
                    if pair_name in results:
                        matrix[i, j] = results[pair_name]['sacrebleu']
        
        # Tạo biểu đồ
        plt.figure(figsize=(10, 8))
        
        plt.imshow(matrix, cmap='viridis')
        plt.colorbar(label='sacreBLEU')
        
        plt.xticks(np.arange(len(languages)), [self.language_names[lang] for lang in languages])
        plt.yticks(np.arange(len(languages)), [self.language_names[lang] for lang in languages])
        
        plt.xlabel('Ngôn ngữ đích')
        plt.ylabel('Ngôn ngữ nguồn')
        plt.title('Ma trận hiệu suất dịch chéo giữa các ngôn ngữ')
        
        # Thêm giá trị vào ô
        for i in range(len(languages)):
            for j in range(len(languages)):
                if matrix[i, j] > 0:
                    plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w")
        
        plt.tight_layout()
        plt.savefig(output_file)
        
        print(f"Đã tạo ma trận hiệu suất dịch chéo: {output_file}")

def main():
    """
    Hàm chính để đánh giá mô hình
    """
    # Đường dẫn
    data_dir = './models'
    output_dir = './evaluation'
    sp_model_path = './processed/spm_model.model'
    model_path = './models/model_epoch_10.pt'
    
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
    
    # Tải trọng số mô hình
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Tạo bộ đánh giá
    evaluator = MultilingualEvaluator(
        model=model,
        sp_model_path=sp_model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device
    )
    
    # Đánh giá mô hình
    language_pairs = [
        ('en', 'fr'), ('en', 'es'), ('en', 'vi'),
        ('fr', 'en'), ('fr', 'es'), ('fr', 'vi'),
        ('es', 'en'), ('es', 'fr'), ('es', 'vi'),
        ('vi', 'en'), ('vi', 'fr'), ('vi', 'es')
    ]
    
    results = evaluator.evaluate_all_language_pairs(language_pairs, num_samples=100)
    
    # Lưu kết quả
    evaluator.save_results(results)
    
    # Tạo báo cáo
    evaluator.generate_report(results)
    
    # Vẽ biểu đồ
    evaluator.plot_bleu_scores(results)
    evaluator.create_cross_lingual_matrix(results)
    
    print("Đánh giá hoàn tất!")

if __name__ == "__main__":
    main()
