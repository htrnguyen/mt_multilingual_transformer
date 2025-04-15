# Thiết kế phương pháp Tokenization đa ngôn ngữ

## 1. Phân tích đặc điểm của từng ngôn ngữ

### Tiếng Anh (English)
- Ngôn ngữ sử dụng bảng chữ cái Latin không dấu
- Phân tách từ bằng khoảng trắng
- Cấu trúc từ đơn giản, ít biến đổi hình thái học
- Dấu câu chuẩn (.,!?;:)

### Tiếng Pháp (French)
- Ngôn ngữ sử dụng bảng chữ cái Latin với nhiều dấu
- Có các ký tự đặc biệt: é, è, ê, à, ç, ù, â, î, ô, û, ë, ï, ÿ
- Dấu câu đặc biệt: dấu chấm than và chấm hỏi thường có khoảng trắng trước (! và ?)
- Nhiều biến đổi hình thái học (giống, số, thì)
- Hiện tượng elision: l'homme, j'ai, c'est

### Tiếng Tây Ban Nha (Spanish)
- Ngôn ngữ sử dụng bảng chữ cái Latin với dấu
- Có các ký tự đặc biệt: á, é, í, ó, ú, ü, ñ
- Dấu câu đặc biệt: dấu chấm hỏi và chấm than đảo ngược ở đầu câu (¿ và ¡)
- Nhiều biến đổi hình thái học (giống, số, thì)
- Từ vựng phong phú với nhiều dạng biến đổi của động từ

### Tiếng Việt (Vietnamese)
- Ngôn ngữ sử dụng bảng chữ cái Latin với nhiều dấu thanh
- 6 dấu thanh: không dấu, huyền, sắc, hỏi, ngã, nặng (a, à, á, ả, ã, ạ)
- Nguyên âm có thể có dấu: ă, â, ê, ô, ơ, ư, đ
- Từ đa âm tiết được phân tách bằng khoảng trắng nhưng có ý nghĩa liên kết
- Không có biến đổi hình thái học (không chia giống, số, thì)

## 2. Thách thức trong tokenization đa ngôn ngữ

1. **Xử lý ký tự đặc biệt**: Mỗi ngôn ngữ có các ký tự đặc biệt riêng cần được xử lý đúng cách
2. **Phân đoạn từ**: Tiếng Việt có cấu trúc từ đa âm tiết phức tạp hơn so với các ngôn ngữ Latin
3. **Biến đổi hình thái học**: Tiếng Pháp và Tây Ban Nha có nhiều biến đổi hình thái học
4. **Kích thước từ vựng**: Từ điển chung cho 4 ngôn ngữ sẽ rất lớn
5. **Hiệu quả học tập**: Mô hình cần học được các mẫu chung giữa các ngôn ngữ

## 3. Phương pháp tokenization đề xuất

### 3.1 Sử dụng SentencePiece với mô hình Unigram

SentencePiece là một thuật toán tokenization không phụ thuộc vào ngôn ngữ, phù hợp cho xử lý đa ngôn ngữ:

- **Ưu điểm**:
  - Xử lý văn bản thô mà không cần tiền xử lý đặc thù cho từng ngôn ngữ
  - Tự động học các subword units từ dữ liệu
  - Xử lý tốt các ngôn ngữ không có ranh giới từ rõ ràng
  - Hỗ trợ mô hình Unigram phù hợp cho ngôn ngữ có cấu trúc phức tạp

- **Tham số**:
  - Kích thước từ vựng: 32,000 tokens (đủ lớn cho 4 ngôn ngữ)
  - Mô hình: Unigram (linh hoạt hơn BPE cho đa ngôn ngữ)
  - Character coverage: 0.9995 (đảm bảo bao phủ hầu hết các ký tự đặc biệt)

### 3.2 Tokenization đặc thù cho từng ngôn ngữ

Ngoài SentencePiece chung, chúng ta sẽ triển khai xử lý đặc thù cho từng ngôn ngữ:

#### Tiếng Anh:
- Chuẩn hóa: chuyển thành chữ thường, xử lý dấu câu
- Xử lý các trường hợp đặc biệt: từ viết tắt, số, đơn vị đo lường

#### Tiếng Pháp:
- Chuẩn hóa: chuyển thành chữ thường, xử lý dấu câu đặc biệt
- Xử lý elision: l', d', qu', j', n', m', t', s', c'
- Bảo toàn các dấu trên nguyên âm

#### Tiếng Tây Ban Nha:
- Chuẩn hóa: chuyển thành chữ thường, xử lý dấu câu đặc biệt (¿, ¡)
- Bảo toàn các dấu trọng âm và ký tự đặc biệt (ñ)

#### Tiếng Việt:
- Chuẩn hóa: giữ nguyên dấu thanh và dấu nguyên âm
- Xử lý từ ghép: cân nhắc giữa tokenization theo từ hoặc âm tiết

### 3.3 Chiến lược học chung và riêng

1. **Từ vựng chung**: Xây dựng từ vựng chung cho tất cả các ngôn ngữ
2. **Embedding ngôn ngữ**: Thêm embedding đánh dấu ngôn ngữ để mô hình phân biệt ngôn ngữ
3. **Học chuyển tiếp**: Huấn luyện mô hình trên tất cả dữ liệu, sau đó tinh chỉnh cho từng cặp ngôn ngữ

## 4. Quy trình tokenization

1. **Tiền xử lý**:
   - Chuẩn hóa Unicode (NFKC)
   - Xử lý dấu câu đặc thù cho từng ngôn ngữ
   - Chuẩn hóa chữ hoa/thường

2. **Huấn luyện tokenizer**:
   - Kết hợp dữ liệu từ tất cả các ngôn ngữ
   - Huấn luyện SentencePiece với mô hình Unigram
   - Lưu mô hình tokenizer

3. **Áp dụng tokenization**:
   - Thêm token đánh dấu ngôn ngữ (<en>, <fr>, <es>, <vi>)
   - Tokenize văn bản đầu vào
   - Chuyển đổi thành ID token

4. **Xử lý đặc biệt**:
   - Xử lý các trường hợp ngoại lệ cho từng ngôn ngữ
   - Bảo toàn thông tin ngôn ngữ trong quá trình tokenization

## 5. Đánh giá và điều chỉnh

- Đánh giá độ phủ từ vựng trên mỗi ngôn ngữ
- So sánh hiệu suất với các phương pháp tokenization riêng biệt
- Điều chỉnh kích thước từ vựng dựa trên kết quả thực nghiệm
- Phân tích lỗi tokenization trên các trường hợp đặc biệt
