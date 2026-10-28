# Multi-Image Visual Search — Review Web

Web kiểm tra và sửa sample, tách độc lập từ `multi_image_visual_search`.
Giữ giao diện review GT, sửa câu hỏi/thuộc tính, thay ảnh ngoài GT, reset và export JSON.
Chạy bằng Python 3.10+, không cần GPU, model, API key hay pipeline sinh dataset.

## 1. Clone và chạy thử ngay

Clone repo:

```bash
git clone https://github.com/fonzi22/multi_image_verify_web.git review-web
cd review-web
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python review_web.py
```

Trên Windows, kích hoạt môi trường bằng `.venv\Scripts\Activate.ps1` trong PowerShell.
Mở **http://127.0.0.1:8002**. Trên Windows có thể dùng `python` thay `python3`.

Repo chứa một sample **demo tổng hợp**, 32 ảnh JPEG nhỏ và crop để thử đầy đủ
luồng xem ảnh, phóng to, lưu, thay ảnh và reset. Đây không phải ảnh benchmark thật.
Mặc định chỉ đọc `demo/`, kết quả ghi vào `reviews/`; hai thư mục tách biệt.

Sau khi bấm Save, kết quả nằm trên **máy chạy backend**, không phải máy chỉ mở trình duyệt:

```text
reviews/
  samples_reviewed.json       # sample sau chỉnh sửa
  sample_review_state.json    # trạng thái và lịch sử review theo sample_id
```

Chỉnh sửa trên giao diện chỉ được lưu khi bấm nút lưu; thao tác thay ảnh và reset
lưu ngay. Khởi động lại web sẽ đọc kết quả đã lưu. Nút Export mở `/api/export`;
endpoint này trả JSON, không tự tạo thêm file export trên server. Theo hành vi gốc,
export chứa cả sample chưa review và loại bỏ sample `no_valid_gt`.

## 2. Chạy với dataset thật có sẵn

Không đưa dataset lớn vào repo. Trỏ web đến thư mục `data` có sẵn trên server
hoặc ổ dùng chung được mount trên máy chạy web. Không cần sao chép ảnh vào repo.

```bash
cp config.server.example.yaml config.local.yaml
export REVIEW_CONFIG="$PWD/config.local.yaml"
export REVIEW_DATA_ROOT=/absolute/path/to/multi_image_visual_search/data
export REVIEW_OUTPUT_DIR="$PWD/reviews-real"
python review_web.py --host 127.0.0.1 --port 8002
```

Nếu cổng 8002 đang chạy web cũ, dùng `--port 8003` để chạy bản mới song song.
Ví dụ PowerShell: `$env:REVIEW_DATA_ROOT = 'D:\datasets\data'`.

Sửa các tên file trong `config.local.yaml` cho khớp dataset. Các đường dẫn dạng
`data/...` được tính từ `REVIEW_DATA_ROOT` sau khi bỏ tiền tố `data/`.
Đường dẫn tuyệt đối được hỗ trợ cho file metadata; `image_dirs` và `crops_dir`
phải là thư mục tương đối bên trong dataset, có tiền tố `data/`.

Ví dụ cấu trúc tương thích với dataset hiện tại:

```text
<REVIEW_DATA_ROOT>/
  output/samples.json
  intermediate/enriched_attributes.json
  intermediate/filtered_bboxes_20000.json
  intermediate/image_urls_20000.json
  pilot_5000_images/<image_id>.jpg
  pilot_10000_images/<image_id>.jpg
  crops/<bbox_id>.jpg
```

`samples.json` là bắt buộc. Các metadata còn lại nên được cung cấp đầy đủ để
đề xuất câu hỏi và thay ảnh đúng với dataset. Giữ cấu trúc `ground_truth`, `query`,
`answer`, `pools` như ví dụ trong `demo/output/samples.json`; đường dẫn ảnh/crop
trong JSON dùng dạng `data/...`, không dùng đường dẫn riêng của máy tác giả.

`REVIEW_OUTPUT_DIR` nên riêng cho từng dataset, nằm ngoài thư mục ảnh.
Web không ghi vào dataset nguồn. Nếu muốn tiếp tục kết quả từ web cũ, trước khi
khởi động hãy copy `pilot_10_samples_reviewed.json` thành
`<REVIEW_OUTPUT_DIR>/samples_reviewed.json` và copy `sample_review_state.json`
vào cùng thư mục đó. Không trộn kết quả giữa các dataset khác nhau.

Các biến `REVIEW_*` được đọc khi khởi động. `.env.example` chỉ là tài liệu mẫu;
web không tự động đọc file `.env`.

Kiểm tra các ảnh/crop mà sample tham chiếu trước khi mở web:

```bash
python check_dataset.py
```

Lệnh này chỉ đọc dữ liệu, báo đường dẫn bị thiếu và trả exit code 1 nếu thiếu
ảnh/crop. Clone code không khôi phục được ảnh đang thiếu trong dataset gốc.

## 3. Người review không cần tải dataset hoặc SSH

Clone repo chỉ cung cấp code và demo; **không cung cấp quyền truy cập ảnh trên server**.
Để review dataset thật mà không tải ảnh, chạy một backend tập trung trên server,
rồi cho mọi người mở URL của backend đó. Người review không cần clone hay cài Python.

Nếu ở cùng mạng và có quyền truy cập cổng server:

```bash
python review_web.py --host 0.0.0.0 --port 8002
```

Mở `http://<server-ip>:8002` từ máy khác. Cách này cần mạng/firewall cho phép.
Đối với truy cập Internet, đặt web sau reverse proxy HTTPS hoặc tunnel có kiểm soát
truy cập. Repo này chưa tích hợp tài khoản/đăng nhập; tránh công khai endpoint ghi
dữ liệu cho người không có quyền review. Mọi người dùng chung một trạng thái review.

Để web chạy lâu dài, dùng service manager của server cho lệnh trên với đúng
working directory và các biến môi trường `REVIEW_*`.

## 4. Lưu trữ và nhiều người review

- Chỉ chạy **một tiến trình backend** cho một thư mục kết quả. Không dùng nhiều
  Uvicorn workers hoặc hai backend cùng ghi vào một `REVIEW_OUTPUT_DIR`.
- Các request truy cập trạng thái được khóa trong tiến trình; từng file JSON
  được ghi qua file tạm rồi thay thế để tránh file bị ghi dở.
- Chưa có khóa sample hoặc phát hiện chỉnh sửa từ giao diện cũ: nếu hai người
  cùng sửa một sample, lần lưu sau có thể ghi đè lần trước. Nên chia sample cho
  từng người. Chưa có nhật ký danh tính người review.
- Hai file kết quả được cập nhật lần lượt, chưa phải một transaction chung.
  Sao lưu thư mục kết quả định kỳ nếu dùng cho công việc thật.
- Chỉ các thư mục ảnh/crop trong config được phục vụ qua HTTP; không mount toàn
  bộ dataset metadata hay thư mục kết quả.

## 5. Đưa thư mục này lên GitHub

Chạy các lệnh sau **bên trong thư mục repo này**:

```bash
git init
git add .
git status --short
git commit -m "Package standalone sample review web"
git branch -M main
git remote add origin https://github.com/fonzi22/multi_image_verify_web.git
git push -u origin main
```

Remote GitHub: https://github.com/fonzi22/multi_image_verify_web.
`.gitignore` loại `data/`, `reviews/`, `reviews-*/`, config cá nhân, `.env`, log
và môi trường Python. Demo nhỏ được đưa lên Git để clone là chạy thử được.
Nếu bạn tự đặt thư mục dữ liệu/kết quả tên khác trong repo, thêm tên đó vào
`.gitignore` trước khi `git add`.

## 6. Kiểm tra

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Các kiểm tra dùng bản demo và thư mục tạm, không chạm kết quả review thật.
