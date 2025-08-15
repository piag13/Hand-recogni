# Cách dùng và cài đặt dự án

## Tải git:

### Cho linux:
```bash
apt-get install git
```
### Cho windows: 
https://git-scm.com/downloads/win

## Cách sử dụng dự án:

### Bước 1: Clone dự án
```bash
git clone https://github.com/piag13/Hand-recogni
cd Hand-recogni
```
### Bước 2: Tạo dataset
Chạy file Create_dataset.py
```bash
python dataset/Create_dataset.py
```
### Bước 3: Train Model
Chạy file Train.py để train model (khuyến khích dùng GPU)
```bash
python src/Train.py
```
### Bước 4: Chạy chương trình
```bash
python src/main.py
```
