from pathlib import Path
import json, csv, hashlib, zipfile, shutil

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / 'submissions/2A202602873_thai_phuc_tien'
t = json.loads((S/'tables.json').read_text(encoding='utf-8'))
f = json.loads((S/'eval_out/F01_summary.json').read_text(encoding='utf-8'))
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
b = table(['ID','Backbone / pretrained tag','Params M','GMAC','Val macro-F1','Train/epoch s'], [[r['exp_id'],r['backbone']+' / '+r['tag'],f"{r['params_m']:.3f}",f"{r['gmacs']:.3f}",f"{r['val_macro_f1']:.4f}",f"{r['time_per_epoch_s']:.1f}"] for r in t['Backbones']])
tr = table(['ID','Thay đổi riêng so với T00','Val macro-F1','Delta'], [[r['exp_id'],json.dumps(r['changes']),f"{r['val_macro_f1']:.4f}",f"{r['delta_macro_f1']:+.4f}"] for r in t['Training']])
inf = table(['ID','Phương pháp','Val macro-F1','ECE','p50 / p95 / p99 ms'], [[r['exp_id'],r['method'],f"{r['val_macro_f1']:.4f}",f"{r['ece']:.4f}",f"{r['p50']:.3f} / {r['p95']:.3f} / {r['p99']:.3f}"] for r in t['Inference']])
pc = table(['Lớp','Precision mean ± std','Recall mean ± std','F1 mean ± std'], [[name]+[f"{f[k]['mean'][i]:.4f} ± {f[k]['std'][i]:.4f}" for k in ['precision','recall','f1']] for i,name in enumerate(f['classes'])])
report = f'''# Báo cáo Lab Day 2 — DeepWeeds

**Học viên:** Thái Phúc Tiến — 2A202602873. **Ngày:** 04/10/2026.

## 1. Kết quả và lựa chọn

Cấu hình được khóa bằng validation trước khi đọc test: ConvNeXt-Tiny, focal loss (T05), horizontal-flip TTA trung bình xác suất (I01). Ba seed 0, 1, 2 đạt test accuracy **97,78% ± 0,29 điểm phần trăm**, macro-F1 **0,9709 ± 0,0045**. Baseline cùng backbone, cross-entropy và một view đạt macro-F1 **0,9714 ± 0,0030**. Chênh lệch −0,00046 không cho thấy cải thiện; không đổi lựa chọn sau khi xem test.

Đây là lab so sánh backbone, recipe và inference. Không triển khai vòng active learning chọn mẫu–gán nhãn–huấn luyện lại trong thí nghiệm này.

## 2. Dữ liệu, kiểm tra pipeline và môi trường

Dùng toàn bộ 17.509 ảnh DeepWeeds, 9 lớp, fold 0 chính thức: train 10.501, validation 3.501, test 3.507. Không có Filename trùng giữa các tập. CSV và SHA-256 được lưu trong `data_labels/` và `split_checks.json`; không chia ngẫu nhiên lại. Negative chiếm 9.106 ảnh, khoảng 52%, nên macro-F1 là tiêu chí chính bên cạnh accuracy. CSV hiện có Chinee apple 1.126 và Lantana 1.063 ảnh, lệch một ảnh so với số mô tả 1.125/1.064; giữ nguyên dữ liệu chính thức.

Kiểm tra trên 9 ảnh thật (mỗi lớp một ảnh): loss ban đầu 2,17677, gần ln(9)=2,19722; overfit cùng vòng huấn luyện xuống 0,04775 sau 13 bước. Đây là kiểm tra pipeline, không phải kết quả tổng quát hóa. Có hình phân bố lớp, mẫu thật và augmentation.

Job chính chạy Tesla T4 trên Kaggle, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, timm 1.0.24. Bắt đầu 12:32:23, kết thúc 16:44:30 giờ Việt Nam ngày 04/10/2026 (4 giờ 12 phút 7 giây). Tổng 19 lần train × 10 epoch: 5 backbone, 8 recipe và 6 lần baseline/final; T00 seed 0 được chạy lại xác định. Có 18 bộ history riêng vì lần chạy T00 seed 0 cùng tên được thay thế. Log, cấu hình và history đều được lưu.

Recipe chung: input 224; train RandomResizedCrop và horizontal flip; validation/test resize 256, center crop 224, chuẩn hóa ImageNet. Batch 32, 10 epoch, AdamW backbone LR 1e-4, head LR 1e-3, weight decay 0,05; bias/norm không decay. Warmup 1 epoch rồi cosine; AMP khi train. Chọn checkpoint macro-F1 validation tốt nhất (giữ checkpoint đầu khi bằng điểm). Ablation dùng seed 0; baseline và final dùng seed 0/1/2. Test chỉ được dự đoán một lần cho mỗi cấu hình/seed sau khóa lựa chọn.

## 3. So sánh backbone

{b}

ConvNeXt có macro-F1 cao nhất trong điều kiện này; Swin đứng kế tiếp. Số GMAC bao gồm attention, không chỉ Conv/Linear. Backbone nhỏ giảm tham số và FLOPs nhưng không đảm bảo latency thấp hơn trên T4. **Giới hạn so sánh:** ConvNeXt dùng pretrained ImageNet-12k rồi fine-tune ImageNet-1k; các model khác dùng tag ImageNet-1k. Vì vậy không quy toàn bộ chênh lệch cho kiến trúc. Mười epoch và recipe chung có thể chưa tối ưu cho từng backbone.

## 4. Ablation công thức huấn luyện

{tr}

Ba trục có nhiều mức: khởi tạo (pretrained fine-tune/frozen/scratch), augmentation (cơ bản/ColorJitter/CutMix), loss (CE/focal/weighted CE), thêm tổ hợp ColorJitter+focal. Mỗi hàng đổi yếu tố được ghi so với T00; T07 kiểm tra kết hợp.

Scratch giảm mạnh trong ngân sách 10 epoch. ColorJitter và focal đều tăng khoảng 0,0057 trên validation seed 0; T05 hơn T03 chỉ 0,000039, chưa đủ bằng chứng phân biệt ổn định. CutMix tăng ít hơn; weighted CE và tổ hợp không cải thiện. Kết quả cho thấy lợi ích không cộng tuyến tính. Lựa chọn T05 theo thứ hạng validation, không tuyên bố focal luôn tốt hơn CE.

## 5. Suy luận, hiệu chuẩn và độ trễ

{inf}

I01 và I03 cùng hai view nhưng khác không gian tổng hợp; I01 có validation macro-F1 cao nhất. I07 fit temperature T=0,689924 trên validation, giảm ECE một view từ 0,04153 xuống 0,00971, giữ nguyên argmax/F1. Đây là chẩn đoán validation, không áp dụng TS vào final I01 và không khẳng định hiệu chuẩn test.

I08 là **no-op**: ConvNeXt-Tiny không có BatchNorm để gộp, nên không tính là phương pháp inference bổ sung có ý nghĩa. I09 là AMP FP16 thực sự, chạy toàn bộ validation trong job phụ sau khóa lựa chọn; không train thêm, không đọc hoặc đánh giá lại test, không thay đổi final. Bốn phương pháp bổ sung có ý nghĩa là I01, I03, I07, I09.

Đo với 10 warmup, 100 iteration, CUDA synchronize, batch 1 và batch 32; TTA đo cả tạo view, forward và tổng hợp. I01 p95=12,161 ms, khoảng hai lần chi phí một view. AMP batch 32 đạt p50=46,768 ms, 684,23 ảnh/s; batch 1 không nhanh hơn FP32 trong đo này. I09 thuộc session T4 khác nên chênh lệch có thể chịu tác động clock/runtime. I07 latency chỉ đo forward, chưa cộng TS/softmax. Các phép đo loại trừ đọc ảnh, CPU preprocessing, truyền dữ liệu và chu kỳ robot; đạt ngân sách model 100 ms không chứng minh robot thời gian thực end-to-end. Latency final đại diện checkpoint T05 seed 0, không đo riêng cả ba seed.

## 6. Final qua ba seed

| Cấu hình | Test accuracy | Test macro-F1 | Test ECE | Val macro-F1 |
|---|---|---|---|---|
| T00 + I00 | 97,73% ± 0,26 pp | 0,9714 ± 0,0030 | 0,0092 ± 0,0029 | 0,9650 ± 0,0034 |
| F01: T05 + I01 | 97,78% ± 0,29 pp | 0,9709 ± 0,0045 | 0,0383 ± 0,0046 | 0,9703 ± 0,0011 |

Std là sample std (ddof=1), không phải khoảng tin cậy. Accuracy tăng khoảng 0,05 điểm phần trăm nhưng macro-F1 không tăng; cả hai chênh lệch nhỏ so với biến thiên seed. ECE final kém hơn baseline. Chênh macro-F1 val/test final chỉ 0,00061, dưới ngưỡng 0,02. Báo cáo giữ nguyên các kết quả này, không chọn lại bằng test.

Tự chấm bằng `eval.py grade`: I1=7, I2=0, I3=4, I4b=1, I5=2; **14/19 điểm có đủ dữ liệu**, phần I tối đa 20. I4a chưa chấm vì không có cặp final calibrated/uncalibrated test. Đây là gợi ý từ evaluator, giảng viên xác nhận; không suy ra điểm tổng 100.

## 7. Theo lớp và phân tích lỗi

{pc}

Recall Chinee apple 93,51%, Snake weed 96,57%. F1 hai lớp này khoảng 0,951, thấp hơn nhiều lớp còn lại. Confusion matrix seed 0 có Chinee apple→Snake weed 10 ảnh (chiều ngược 4), Negative→Prickly acacia 8, Chinee apple→Negative 6; còn Parthenium→Prickly acacia 5. Các hình lỗi giúp xem ảnh thật, nhưng 12 ảnh được lấy theo thứ tự Filename, không phải mẫu đại diện thống kê và không đủ chứng minh nguyên nhân sinh học.

![Confusion matrix final seed 0](curves/F01_seed0_confusion.png)

![Ví dụ lỗi thật final seed 0](curves/F01_seed0_errors.png)

## 8. Kết luận và hướng tiếp theo

Trong ngân sách này, pretrained ConvNeXt là lựa chọn tốt trên validation; recipe và TTA có tác động đo được trên một seed nhưng lợi ích không được xác nhận trên test ba seed. Nếu ưu tiên chi phí và độ tin cậy xác suất, baseline CE một view là ứng viên hợp lý cho một thí nghiệm tiếp theo. Cần nghiên cứu với nhiều seed ở vòng chọn recipe và hiệu chuẩn trên tập calibration riêng; không sửa kết quả hoặc tái tối ưu trên test đã dùng. Chưa đo triển khai thực tế, phân bố ngoài miền hoặc active learning.

## 9. Bằng chứng và tái lập

`results.xlsx` gồm Summary, Final, Backbones, Training, Inference, PerClass, Latency; Final dùng AVERAGE/STDEV.S từ hàng seed. `tables.json` là bảng hợp nhất; `evidence/tables_raw_gpu.json` giữ bảng gốc trước bổ sung AMP/no-op annotations. CSV bảng được đồng bộ với JSON. 24 CSV validation và 6 CSV test được đối chiếu Filename/Label với fold 0; metric recompute khớp bảng. `eval.py` giữ nguyên bản repo.

Xem `README.md` để mở hai notebook Kaggle, tái chạy và đánh giá. Checkpoint giữ trong private Kaggle Output, không đưa ảnh gốc hoặc weights vào ZIP. Log job phụ và `amp_validation.json` giải thích I09; không dùng kết quả bổ sung để sửa lựa chọn đã khóa.
'''
(S/'report.md').write_text(report,encoding='utf-8')
readme='''# DeepWeeds — Thái Phúc Tiến — 2A202602873

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
'''
(S/'README.md').write_text(readme,encoding='utf-8')
# Keep flat CSV views consistent with the enriched workbook data.
for name,rows in t.items():
    if not rows: continue
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (S/f'{name}.csv').open('w',encoding='utf-8',newline='') as out:
        w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
out=ROOT/'artifacts/final_submission';out.mkdir(parents=True,exist_ok=True)
files=[]
allowed_dirs={'curves','data_labels','eval_out','evidence','predictions','runs'}
for p in S.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(S)
    if len(rel.parts)==1:
        keep=p.suffix in {'.json','.csv','.log','.md','.xlsx','.txt'} and p.name not in {'workbook_checks.txt'}
    elif rel.parts[0]=='code':
        keep=len(rel.parts)==2 and p.suffix in {'.py','.ipynb'}
    else:keep=rel.parts[0] in allowed_dirs and p.suffix in {'.json','.csv','.png','.npy','.md','.txt','.log'}
    if keep:files.append(p)
manifest={str(p.relative_to(S)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
(S/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
files.append(S/'MANIFEST.json')
target=out/(S.name+'.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,Path(S.name)/p.relative_to(S))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert not any('/data/' in n or n.endswith(('.pt','.pth','.jpg','.pyc')) for n in z.namelist())
    for req in ['README.md','report.md','results.xlsx','code/kaggle_complete.ipynb','predictions/F01_seed2_test.csv']:
        assert S.name+'/'+req in z.namelist()
print(json.dumps({'zip':str(target),'files':len(files),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
