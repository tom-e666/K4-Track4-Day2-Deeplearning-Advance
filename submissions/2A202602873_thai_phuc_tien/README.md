# DeepWeeds — Thái Phúc Tiến — 2A202602873

Bài hoàn thành bằng dữ liệu thật và GPU Tesla T4 ngày 04/10/2026. Final: ConvNeXt-Tiny + focal loss + horizontal-flip probability TTA. Accuracy test 97,78% ± 0,29 pp; macro-F1 0,9709 ± 0,0045, chưa cải thiện baseline 0,9714 ± 0,0030. Xem report để đọc giới hạn và kết quả đầy đủ.

## Notebook và phiên chạy

- [Job chính, toàn bộ thí nghiệm](https://www.kaggle.com/code/tom298/track4-day2-deepweeds-complete-20261004)
- [AMP validation bổ sung](https://www.kaggle.com/code/tom298/track4-day2-deepweeds-amp-validation-20261004)

Hai notebook hiện private, cần đăng nhập tài khoản được cấp quyền. Bản source notebook của phiên thực thi được lưu trong code; runtime/logs và kết quả đi kèm là bằng chứng độc lập với quyền truy cập URL. Không tự động công khai dữ liệu/tài khoản. Job chính Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, timm 1.0.24; seed 0/1/2, epoch 10, batch 32. `runtime.json` có thời điểm UTC chính xác. Phiên chạy notebook được nhận diện bằng URL có ngày 20261004 và bản source/log đi kèm; không khẳng định số version UI chưa xác minh.

## Sản phẩm

1. results.xlsx: bảy sheet theo yêu cầu, các thống kê final là công thức.
2. report.md: báo cáo kết quả, phân tích và giới hạn.
3. curves/: EDA, augmentation, 18 đường học, confusion/errors và inference tradeoff.
4. code/: dataset/model/loss/train/inference/benchmark, runner và notebook có thể chạy.
5. README.md: hướng dẫn này.
6. predictions/: 24 CSV validation, 6 CSV test, đầy đủ xác suất chín lớp.

data_labels/ chứa bốn CSV fold 0 chính thức; runs/ chứa config/history/metrics/logits; eval_out/ chứa kết quả evaluator; evidence/ giữ bảng GPU gốc. Không chứa ảnh dataset, API key hoặc checkpoint. Checkpoint có trong Output của job Kaggle chính. SHA-256 của từng file đóng gói nằm trong MANIFEST.json.

## Chạy lại GPU

Import code/kaggle_complete.ipynb vào Kaggle, attach dataset `imsparsh/deepweeds`, bật GPU và Internet, chọn Save & Run All. Notebook nhúng các module và runner nên tự dựng đường dẫn trong /kaggle/working; tải CSV chính thức, kiểm tra split, chạy ablations, khóa lựa chọn trên val rồi mới đánh giá test. Cần môi trường phù hợp và khoảng 4 giờ trên T4; thời gian có thể thay đổi. Kết quả tái chạy có thể sai khác do runtime/GPU và phép toán không xác định.

Notebook AMP attach dataset trên và Output của job chính như kernel source; nó tìm T05 seed 0 best_model.pt, chỉ đánh giá validation và latency, không đụng test. `lab_day2.ipynb` là bản tương tác; hai notebook kaggle_* là source dùng cho cloud job.

## Đánh giá lại không cần GPU

Từ thư mục bài đã giải nén, cài numpy, pandas, rồi chạy:

```powershell
python code/eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv data_labels/test_subset0.csv --labels data_labels/labels.csv --tag F01 --out eval_out
python code/eval.py score --pred "predictions/T00_seed*_test.csv" --test-csv data_labels/test_subset0.csv --labels data_labels/labels.csv --tag T00 --out eval_out
python code/eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --test-csv data_labels/test_subset0.csv --val-csv data_labels/val_subset0.csv --labels data_labels/labels.csv --latency-p95-ms 12.16125365 --out eval_out
```

Không sửa evaluator. Không có --uncal vì final chưa áp dụng temperature scaling. Điểm phần I có dữ liệu là 14/19, một điểm calibration chưa đánh giá; giảng viên quyết định điểm chính thức.

## Lưu ý diễn giải

I08 gộp BN là no-op trên ConvNeXt và không tính trong số phương pháp thực sự. I09 AMP chỉ bổ sung validation sau khóa final. Temperature scaling tốt hơn trên validation một view không chứng minh final test được hiệu chuẩn. Latency chỉ bao gồm model/TTA trên T4, chưa gồm I/O/preprocessing và hệ thống robot. Lab này không có vòng active learning.
