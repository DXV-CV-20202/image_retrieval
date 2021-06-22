# Các phần thay đổi :
- __init__.py trong feature_extractor  
- __init__.py trong ir_utils  
- extract_feature.py để tách từng loại feature cho từng loại dataset riêng  
- retrieve.py cũng tương tự  

# Kết quả thử nghiệm
- cifar10 : HOG (48 - 43 - 41 - 38)
- coil100 : HuMoments (70 - 60 - 54 - 45), HOG (97 - 92 - 86 - 76), HSV Hist (99 - 98 - 96 - 93)
- caltech101 : HuMoments (16 - 14 - 13 - 12), HOG (56 - 51 - 48 - 45), HSV Hist (26 - 23 - 21 - 19), HOG_HSV (44 - 41 - 39 - 36)