import os
import torch
import gradio as gr
import sentencepiece as spm
from multilingual_transformer import create_multilingual_transformer
from language_routing import LanguageDetector, LanguageRouter, MultilingualTranslationPipeline

class MultilingualTranslationDemo:
    """
    Demo dịch thuật đa ngôn ngữ sử dụng Gradio
    """
    
    def __init__(self, model_path, sp_model_path, device='cpu'):
        """
        Khởi tạo demo dịch thuật
        
        Args:
            model_path: Đường dẫn đến mô hình MultilingualTransformer
            sp_model_path: Đường dẫn đến mô hình SentencePiece
            device: Thiết bị tính toán ('cpu' hoặc 'cuda')
        """
        self.model_path = model_path
        self.sp_model_path = sp_model_path
        self.device = device
        
        # Tải mô hình SentencePiece
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        
        # Lấy kích thước từ vựng
        self.vocab_size = self.sp.get_piece_size()
        
        # Tạo mô hình
        self.model = self.create_model()
        
        # Tạo pipeline dịch thuật
        self.pipeline = self.create_pipeline()
        
        # Ánh xạ mã ngôn ngữ sang tên ngôn ngữ
        self.language_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'vi': 'Vietnamese'
        }
        
        # Ánh xạ tên ngôn ngữ sang mã ngôn ngữ
        self.language_codes = {v: k for k, v in self.language_names.items()}
    
    def create_model(self):
        """
        Tạo mô hình MultilingualTransformer
        
        Returns:
            Mô hình MultilingualTransformer
        """
        # Tạo mô hình
        model = create_multilingual_transformer(
            vocab_size=self.vocab_size,
            num_languages=4,
            d_model=256,
            num_heads=8,
            d_ff=512,
            num_layers=3,
            max_seq_length=100,
            dropout=0.1
        )
        
        # Tải trọng số mô hình nếu tồn tại
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # Chuyển mô hình sang thiết bị tính toán
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def create_pipeline(self):
        """
        Tạo pipeline dịch thuật
        
        Returns:
            Pipeline dịch thuật
        """
        # Tạo bộ phát hiện ngôn ngữ
        detector = LanguageDetector()
        
        # Tạo bộ định tuyến ngôn ngữ
        router = LanguageRouter(detector, self.sp_model_path)
        
        # Tạo pipeline dịch thuật
        pipeline = MultilingualTranslationPipeline(self.model, router, self.device)
        
        return pipeline
    
    def translate(self, text, src_lang, tgt_lang, auto_detect=False):
        """
        Dịch văn bản
        
        Args:
            text: Văn bản cần dịch
            src_lang: Tên ngôn ngữ nguồn
            tgt_lang: Tên ngôn ngữ đích
            auto_detect: Tự động phát hiện ngôn ngữ nguồn
            
        Returns:
            Văn bản đã dịch
        """
        if not text:
            return ""
        
        # Chuyển đổi tên ngôn ngữ sang mã ngôn ngữ
        src_code = self.language_codes.get(src_lang)
        tgt_code = self.language_codes.get(tgt_lang)
        
        # Tự động phát hiện ngôn ngữ nguồn nếu cần
        if auto_detect:
            src_code = None
        
        # Dịch văn bản
        translated_text = self.pipeline.translate(text, src_code, tgt_code)
        
        return translated_text
    
    def detect_language(self, text):
        """
        Phát hiện ngôn ngữ của văn bản
        
        Args:
            text: Văn bản cần phát hiện ngôn ngữ
            
        Returns:
            Tên ngôn ngữ
        """
        if not text:
            return "English"
        
        # Phát hiện ngôn ngữ
        lang_code = self.pipeline.router.detect_language(text)
        
        # Chuyển đổi mã ngôn ngữ sang tên ngôn ngữ
        lang_name = self.language_names.get(lang_code, "English")
        
        return lang_name
    
    def create_demo(self):
        """
        Tạo demo Gradio
        
        Returns:
            Demo Gradio
        """
        # Danh sách ngôn ngữ
        languages = list(self.language_names.values())
        
        # Tạo giao diện
        with gr.Blocks(title="Multilingual Translation Demo") as demo:
            gr.Markdown("# Multilingual Translation Demo")
            gr.Markdown("Translate between English, French, Spanish, and Vietnamese")
            
            with gr.Row():
                with gr.Column():
                    src_text = gr.Textbox(label="Source Text", lines=5, placeholder="Enter text to translate...")
                    
                    with gr.Row():
                        auto_detect = gr.Checkbox(label="Auto-detect language", value=True)
                        detected_lang = gr.Textbox(label="Detected Language", interactive=False)
                    
                    with gr.Row():
                        src_lang = gr.Dropdown(choices=languages, value="English", label="Source Language", interactive=True)
                        tgt_lang = gr.Dropdown(choices=languages, value="French", label="Target Language")
                    
                    translate_btn = gr.Button("Translate")
                
                with gr.Column():
                    tgt_text = gr.Textbox(label="Translated Text", lines=5)
            
            # Cập nhật ngôn ngữ phát hiện khi văn bản thay đổi
            src_text.change(
                fn=self.detect_language,
                inputs=[src_text],
                outputs=[detected_lang]
            )
            
            # Cập nhật trạng thái tương tác của dropdown ngôn ngữ nguồn
            auto_detect.change(
                fn=lambda x: gr.update(interactive=not x),
                inputs=[auto_detect],
                outputs=[src_lang]
            )
            
            # Dịch văn bản khi nhấn nút
            translate_btn.click(
                fn=self.translate,
                inputs=[src_text, src_lang, tgt_lang, auto_detect],
                outputs=[tgt_text]
            )
            
            # Ví dụ
            gr.Examples(
                [
                    ["Hello, how are you today?", "English", "French", True],
                    ["Bonjour, comment allez-vous aujourd'hui?", "French", "English", True],
                    ["Hola, ¿cómo estás hoy?", "Spanish", "Vietnamese", True],
                    ["Xin chào, hôm nay bạn khỏe không?", "Vietnamese", "Spanish", True]
                ],
                fn=self.translate,
                inputs=[src_text, src_lang, tgt_lang, auto_detect],
                outputs=[tgt_text]
            )
        
        return demo
    
    def launch(self, share=False):
        """
        Khởi chạy demo
        
        Args:
            share: Chia sẻ demo công khai
        """
        demo = self.create_demo()
        demo.launch(share=share)

def main():
    """
    Hàm chính để chạy demo
    """
    # Đường dẫn
    model_path = './models/model_epoch_10.pt'
    sp_model_path = './processed/spm_model.model'
    
    # Kiểm tra xem mô hình có tồn tại không
    if not os.path.exists(model_path):
        print(f"Mô hình không tồn tại: {model_path}")
        print("Sử dụng mô hình mới tạo (chưa huấn luyện)")
        model_path = None
    
    # Tạo demo
    demo = MultilingualTranslationDemo(
        model_path=model_path,
        sp_model_path=sp_model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Khởi chạy demo
    demo.launch(share=True)

if __name__ == "__main__":
    main()
