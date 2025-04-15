# Hướng dẫn sử dụng hệ thống dịch máy đa ngôn ngữ

## 1. Giới thiệu

Hệ thống dịch máy đa ngôn ngữ là một giải pháp dịch thuật toàn diện hỗ trợ bốn ngôn ngữ: Anh (English), Pháp (French), Tây Ban Nha (Spanish) và Việt (Vietnamese). Hệ thống sử dụng kiến trúc Transformer được mở rộng để hỗ trợ đa ngôn ngữ, với khả năng dịch thuật hai chiều giữa bất kỳ cặp ngôn ngữ nào trong số bốn ngôn ngữ được hỗ trợ.

## 2. Cài đặt

### 2.1. Yêu cầu hệ thống

- Python 3.8 trở lên
- PyTorch 1.8.0 trở lên
- CUDA (tùy chọn, để tăng tốc độ xử lý)

### 2.2. Cài đặt thư viện

```bash
# Cài đặt các thư viện cần thiết
pip install torch numpy matplotlib scikit-learn sentencepiece gradio nltk sacrebleu pandas tabulate
```

### 2.3. Tải mã nguồn

Tải mã nguồn từ repository hoặc giải nén từ file zip đính kèm:

```bash
# Tạo thư mục cho dự án
mkdir multilingual_transformer
cd multilingual_transformer

# Sao chép các file mã nguồn
# (hoặc giải nén từ file zip)
```

## 3. Cấu trúc thư mục

```
multilingual_transformer/
├── data/                      # Thư mục chứa dữ liệu gốc
│   ├── fra-eng.zip            # Dữ liệu Anh-Pháp
│   ├── spa-eng.zip            # Dữ liệu Anh-Tây Ban Nha
│   └── vie-eng.zip            # Dữ liệu Anh-Việt
├── extracted/                 # Thư mục chứa dữ liệu đã giải nén
├── processed/                 # Thư mục chứa dữ liệu đã tiền xử lý
├── models/                    # Thư mục chứa mô hình đã huấn luyện
├── evaluation/                # Thư mục chứa kết quả đánh giá
├── preprocess.py              # Mã nguồn tiền xử lý dữ liệu
├── multilingual_transformer.py # Mã nguồn kiến trúc Transformer đa ngôn ngữ
├── language_routing.py        # Mã nguồn phát hiện và định tuyến ngôn ngữ
├── train_multilingual.py      # Mã nguồn huấn luyện mô hình
├── evaluate_multilingual.py   # Mã nguồn đánh giá hiệu suất
├── demo.py                    # Mã nguồn demo dịch thuật
├── tokenization_design.md     # Tài liệu thiết kế tokenization
├── architecture.md            # Tài liệu kiến trúc hệ thống
└── README.md                  # Hướng dẫn sử dụng
```

## 4. Tiền xử lý dữ liệu

### 4.1. Tải dữ liệu

Tải dữ liệu từ manythings.org/anki:

```bash
# Tạo thư mục data nếu chưa tồn tại
mkdir -p data

# Tải dữ liệu
wget -O data/fra-eng.zip https://www.manythings.org/anki/fra-eng.zip
wget -O data/spa-eng.zip https://www.manythings.org/anki/spa-eng.zip
wget -O data/vie-eng.zip https://www.manythings.org/anki/vie-eng.zip
```

### 4.2. Tiền xử lý dữ liệu

Chạy script tiền xử lý để chuẩn bị dữ liệu cho huấn luyện:

```bash
# Tiền xử lý dữ liệu
python preprocess.py
```

Script này sẽ thực hiện các bước sau:
- Giải nén dữ liệu
- Chuẩn hóa văn bản theo đặc thù của từng ngôn ngữ
- Huấn luyện mô hình SentencePiece
- Chia dữ liệu thành các tập huấn luyện, kiểm tra và đánh giá

## 5. Huấn luyện mô hình

### 5.1. Huấn luyện mô hình đa ngôn ngữ

Chạy script huấn luyện để huấn luyện mô hình đa ngôn ngữ:

```bash
# Huấn luyện mô hình
python train_multilingual.py
```

Script này sẽ thực hiện các bước sau:
- Tạo mô hình Transformer đa ngôn ngữ
- Huấn luyện mô hình trên tất cả các cặp ngôn ngữ
- Lưu mô hình sau mỗi epoch

### 5.2. Tham số huấn luyện

Bạn có thể điều chỉnh các tham số huấn luyện trong file `train_multilingual.py`:

```python
# Tham số huấn luyện
batch_size = 32
epochs = 10
lr = 0.0001
save_every = 1
```

## 6. Đánh giá mô hình

### 6.1. Đánh giá hiệu suất dịch chéo

Chạy script đánh giá để đánh giá hiệu suất dịch chéo giữa các ngôn ngữ:

```bash
# Đánh giá mô hình
python evaluate_multilingual.py
```

Script này sẽ thực hiện các bước sau:
- Đánh giá mô hình trên tất cả các cặp ngôn ngữ
- Tính điểm BLEU và sacreBLEU
- Tạo báo cáo đánh giá và biểu đồ kết quả

### 6.2. Xem kết quả đánh giá

Kết quả đánh giá được lưu trong thư mục `evaluation`:

```
evaluation/
├── evaluation_results.pkl    # Kết quả đánh giá dạng pickle
├── evaluation_report.md      # Báo cáo đánh giá dạng Markdown
├── bleu_scores.png           # Biểu đồ điểm BLEU
└── cross_lingual_matrix.png  # Ma trận hiệu suất dịch chéo
```

## 7. Sử dụng demo dịch thuật

### 7.1. Khởi chạy demo

Chạy script demo để khởi chạy giao diện dịch thuật:

```bash
# Khởi chạy demo
python demo.py
```

Script này sẽ khởi chạy một giao diện web sử dụng Gradio, cho phép bạn dịch văn bản giữa bất kỳ cặp ngôn ngữ nào trong số bốn ngôn ngữ được hỗ trợ.

### 7.2. Sử dụng demo

1. Nhập văn bản cần dịch vào ô "Source Text"
2. Chọn ngôn ngữ nguồn và đích từ dropdown menu
3. Hoặc bật tùy chọn "Auto-detect language" để tự động phát hiện ngôn ngữ nguồn
4. Nhấn nút "Translate" để dịch văn bản
5. Kết quả dịch sẽ hiển thị trong ô "Translated Text"

## 8. Sử dụng API dịch thuật

### 8.1. Tạo pipeline dịch thuật

```python
from language_routing import create_translation_pipeline

# Tạo pipeline dịch thuật
pipeline = create_translation_pipeline(
    model_path='models/model_epoch_10.pt',
    sp_model_path='processed/spm_model.model',
    device='cuda'  # hoặc 'cpu'
)
```

### 8.2. Dịch văn bản

```python
# Dịch văn bản
translated_text = pipeline.translate(
    text="Hello, how are you today?",
    src_lang="en",  # hoặc None để tự động phát hiện
    tgt_lang="fr"
)

print(translated_text)
```

### 8.3. Dịch batch văn bản

```python
# Dịch batch văn bản
texts = [
    "Hello, how are you today?",
    "I love learning new languages.",
    "Machine translation is fascinating."
]

translated_texts = pipeline.batch_translate(
    texts=texts,
    src_lang="en",
    tgt_lang="fr"
)

for text in translated_texts:
    print(text)
```

## 9. Mở rộng hệ thống

### 9.1. Thêm ngôn ngữ mới

Để thêm một ngôn ngữ mới vào hệ thống:

1. Tải dữ liệu song ngữ cho ngôn ngữ mới
2. Thêm xử lý đặc thù cho ngôn ngữ mới trong `preprocess.py`
3. Thêm đặc trưng ngôn ngữ mới trong `language_routing.py`
4. Cập nhật ánh xạ ngôn ngữ trong các file liên quan
5. Huấn luyện lại mô hình với dữ liệu mới

### 9.2. Tinh chỉnh mô hình

Để tinh chỉnh mô hình cho một lĩnh vực cụ thể:

1. Chuẩn bị dữ liệu song ngữ trong lĩnh vực cụ thể
2. Tải mô hình đã huấn luyện
3. Tinh chỉnh mô hình trên dữ liệu mới với learning rate thấp hơn
4. Đánh giá mô hình trên dữ liệu kiểm tra trong lĩnh vực cụ thể

## 10. Xử lý sự cố

### 10.1. Lỗi thiếu thư viện

Nếu gặp lỗi thiếu thư viện, hãy cài đặt thư viện đó:

```bash
pip install <tên_thư_viện>
```

### 10.2. Lỗi CUDA

Nếu gặp lỗi CUDA, hãy thử chuyển sang CPU:

```python
device = 'cpu'
```

### 10.3. Lỗi bộ nhớ

Nếu gặp lỗi bộ nhớ, hãy giảm kích thước batch hoặc kích thước mô hình:

```python
batch_size = 16  # Giảm kích thước batch

# Hoặc giảm kích thước mô hình
model = create_multilingual_transformer(
    vocab_size=vocab_size,
    num_languages=4,
    d_model=128,  # Giảm từ 256
    num_heads=4,  # Giảm từ 8
    d_ff=256,     # Giảm từ 512
    num_layers=2  # Giảm từ 3
)
```

## 11. Tài liệu tham khảo

- [Kiến trúc hệ thống](architecture.md): Mô tả chi tiết về kiến trúc hệ thống dịch máy đa ngôn ngữ
- [Thiết kế tokenization](tokenization_design.md): Mô tả chi tiết về phương pháp tokenization đa ngôn ngữ
- [Trang web manythings.org/anki](https://www.manythings.org/anki/): Nguồn dữ liệu song ngữ
- [Tài liệu SentencePiece](https://github.com/google/sentencepiece): Thư viện tokenization
- [Tài liệu PyTorch](https://pytorch.org/docs/stable/index.html): Thư viện deep learning
- [Tài liệu Gradio](https://gradio.app/docs/): Thư viện tạo giao diện web
