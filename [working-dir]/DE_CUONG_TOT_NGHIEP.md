# ĐỀ CƯƠNG ĐỒ ÁN TỐT NGHIỆP

---

## THÔNG TIN NHÓM

**Nhóm số:** 07

| STT | Họ và tên | MSSV | Lớp |
|-----|-----------|------|-----|
| 1 | Nguyễn Ngọc Dương | B22DCVT112 | D22VTHI02 |
| 2 | Hà Minh Hải | B22DCVT173 | D22VTMD03 |
| 3 | Tạ Trung Kiên | B22DCVT265 | D22VTMD01 |

**Khoá:** 2022 – 2026  
**Ngành:** Kỹ thuật Điện tử Viễn thông | **Hệ:** Chính quy

---

## 1. TÊN ĐỀ TÀI

**Nghiên cứu, xây dựng hệ thống phát hiện tấn công mạng IoT dựa trên kỹ thuật XAI**

---

## 2. LÝ DO CHỌN ĐỀ TÀI

Sự bùng nổ của các thiết bị Internet vạn vật (IoT) trong những năm gần đây — từ nhà thông minh, y tế, giao thông đến công nghiệp — đã kéo theo sự gia tăng nhanh chóng về số lượng và mức độ tinh vi của các cuộc tấn công mạng nhắm vào hệ thống IoT (DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, Mirai...). Đặc điểm tài nguyên hạn chế, thiếu cơ chế bảo mật nội tại và tính không đồng nhất của các thiết bị IoT khiến việc phát hiện tấn công trở nên khó khăn hơn so với mạng máy tính truyền thống.

Các hệ thống phát hiện xâm nhập (IDS) dựa trên Machine Learning/Deep Learning đã cho thấy hiệu quả cao trong việc phân loại lưu lượng độc hại, tuy nhiên phần lớn hoạt động theo cơ chế "hộp đen" (black-box), khiến người quản trị không thể hiểu được vì sao mô hình đưa ra một quyết định cụ thể. Điều này gây khó khăn trong việc kiểm chứng độ tin cậy, phát hiện sai lệch (bias), gỡ lỗi mô hình, cũng như đáp ứng các yêu cầu về tính minh bạch trong an ninh mạng.

Kỹ thuật AI giải thích được (Explainable AI - XAI), với các phương pháp tiêu biểu như SHAP, LIME, giúp diễn giải đóng góp của từng đặc trưng (feature) vào quyết định của mô hình, qua đó tăng độ tin cậy, hỗ trợ chuyên gia an ninh mạng đưa ra quyết định ứng phó nhanh và chính xác hơn. Việc kết hợp XAI vào hệ thống phát hiện tấn công IoT không chỉ nâng cao hiệu năng phát hiện mà còn giải quyết bài toán minh bạch hóa mô hình — một yêu cầu cấp thiết trong bối cảnh an ninh mạng hiện nay.

---

## 3. NỘI DUNG VÀ NHIỆM VỤ NGHIÊN CỨU

**Chương I: Giới thiệu chung**
1.1. Tổng quan về IoT, đặc điểm mạng IoT và bề mặt tấn công
1.2. Tổng quan về hệ thống phát hiện xâm nhập (IDS) trong môi trường IoT
1.3. Giới thiệu về Explainable AI (XAI) và vai trò của XAI trong an ninh mạng
1.4. Bài toán phát hiện tấn công mạng IoT
1.5. Tổng kết chương I

**Chương II: Bài toán phát hiện tấn công mạng IoT dựa trên kỹ thuật XAI**
2.1. Các loại tấn công phổ biến trên mạng IoT
   2.1.1. Tấn công từ chối dịch vụ (DoS/DDoS)
   2.1.2. Tấn công Reconnaissance (Recon)
   2.1.3. Tấn công Brute Force
   2.1.4. Tấn công Spoofing
   2.1.5. Malware và Botnet (Mirai)
2.2. Các mô hình Machine Learning/Deep Learning trong IDS
   2.2.1. Mô hình Tree-based (Decision Tree, Random Forest)
   2.2.2. Mô hình Gradient Boosting (XGBoost, LightGBM)
   2.2.3. Mô hình Deep Learning (MLP, CNN, RNN/LSTM)
   2.2.4. Đánh giá hiệu năng các mô hình
2.3. Giới thiệu về Explainable AI (XAI)
   2.3.1. Khái niệm và sự cần thiết của XAI
   2.3.2. Phân loại các phương pháp XAI
   2.3.3. SHAP (SHapley Additive exPlanations)
   2.3.4. LIME (Local Interpretable Model-agnostic Explanations)
2.4. Áp dụng XAI trong phát hiện tấn công mạng IoT
   2.4.1. Vai trò của XAI trong an ninh mạng IoT
   2.4.2. Các nghiên cứu liên quan về phát hiện tấn công IoT có giải thích
   2.4.3. Một số hướng tiếp cận hiện có
2.5. Phân tích dataset CICIoT2023
   2.5.1. Giới thiệu dataset
   2.5.2. Cấu trúc dữ liệu và các đặc trưng
   2.5.3. Các loại tấn công trong dataset
2.6. Tổng kết chương II

**Chương III: Xây dựng hệ thống**
3.1. Kiến trúc hệ thống phát hiện tấn công IoT dựa trên XAI
3.2. Tiền xử lý dữ liệu và trích chọn đặc trưng
3.3. Huấn luyện mô hình phát hiện tấn công (Random Forest, XGBoost, LightGBM)
3.4. Áp dụng kỹ thuật XAI (SHAP, LIME) để giải thích dự đoán
3.5. Tổng kết chương III

**Chương IV: Thực nghiệm và đánh giá kết quả**
4.1. Môi trường thực nghiệm
4.2. Kết quả huấn luyện mô hình và so sánh
4.3. Kết quả phân tích XAI và đánh giá hệ thống
4.4. Tổng kết chương IV

---

## 4. TÀI LIỆU THAM KHẐO

[1] Shafiq, M., et al. *"CICIoT2023: A Real-time Network Traffic Dataset for IoT Intrusion Detection"*, IEEE Access, 2023.

[2] Lundberg, S. M., & Lee, S. I. *"A Unified Approach to Interpreting Model Predictions"*, Advances in Neural Information Processing Systems (NeurIPS), 2017.

[3] Ribeiro, M. T., Singh, S., & Guestrin, C. *""Why Should I Trust You?": Explaining the Predictions of Any Classifier"*, ACM SIGKDD, 2016.

[4] Chen, T., & Guestrin, C. *"XGBoost: A Scalable Tree Boosting System"*, ACM SIGKDD, 2016.

[5] Ke, G., et al. *"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"*, Advances in Neural Information Processing Systems (NeurIPS), 2017.

[6] Breiman, L. *"Random Forests"*, Machine Learning, Vol. 45, pp. 5-32, 2001.

[7] Molnar, C. *"Interpretable Machine Learning: A Guide for Making Black Box Models Explainable"*, 2nd Edition, 2022.

[8] Arrieta, A. B., et al. *"Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges"*, Information Fusion, Vol. 58, pp. 82-115, 2020.

[9] Ferrag, M. A., et al. *"Deep Learning for Cyber Security Intrusion Detection: A Survey"*, IEEE Communications Surveys & Tutorials, Vol. 22, No. 1, 2020.

[10] Khrais, L. T., et al. *"Network Intrusion Detection Based on Machine Learning: A Comparative Study"*, International Journal of Advanced Computer Science and Applications, Vol. 11, 2020.

[11] Singh, J., et al. *"Explainable AI for Network Intrusion Detection: A Systematic Review"*, IEEE Access, 2024.

[12] Vinayakumar, R., et al. *"Deep Learning Approach for Intelligent Intrusion Detection System"*, IEEE Access, Vol. 7, 2019.

---

**Ngày giao đề cương:** 25/9/2026  
**Ngày nộp quyển:** 06/12/2026

---

| | |
|---|---|
| **CÁN BỘ, GIẢNG VIÊN HƯỚNG DẪN** | **TRƯỞNG NHÓM SINH VIÊN** |
| *(Ký, ghi rõ họ tên)* | *(Ký, ghi rõ họ tên)* |
| | Nguyễn Ngọc Dương |
| | Hà Minh Hải |
| | Tạ Trung Kiên |

| |
|---|
| **TRƯỞNG BỘ MÔN** |
| *(Ký, ghi rõ họ tên)* |

---

## PHỤ LỤC: BẢNG PHÂN CÔNG CÔNG VIỆC

**Nhóm số 07**

| STT | Nội dung chi tiết | SV thực hiện | Ghi chú |
|-----|-------------------|--------------|---------|
| **Chương I** | Giới thiệu chung | | |
| 1.1 | Tổng quan về IoT và bề mặt tấn công | Nguyễn Ngọc Dương | |
| 1.2 | Hệ thống phát hiện xâm nhập (IDS) | Hà Minh Hải | |
| 1.3 | Giới thiệu XAI và vai trò trong an ninh mạng | Tạ Trung Kiên | |
| 1.4-1.5 | Bài toán và tổng kết | Toàn nhóm | |
| **Chương II** | Bài toán phát hiện tấn công mạng IoT dựa trên kỹ thuật XAI | | |
| 2.1 | Các loại tấn công phổ biến trên mạng IoT | Nguyễn Ngọc Dương | |
| 2.2 | ML/DL trong IDS | Hà Minh Hải | |
| 2.3.1-2.3.2 | Khái niệm, phân loại XAI | Tạ Trung Kiên | |
| 2.3.3-2.3.4 | SHAP, LIME | Tạ Trung Kiên | |
| 2.4 | XAI trong phát hiện tấn công IoT | Hà Minh Hải | |
| 2.5 | Phân tích dataset CICIoT2023 | Nguyễn Ngọc Dương | |
| 2.6 | Tổng kết | Toàn nhóm | |
| **Chương III** | Xây dựng hệ thống | | |
| 3.1 | Kiến trúc hệ thống | Nguyễn Ngọc Dương | |
| 3.2 | Tiền xử lý dữ liệu | Hà Minh Hải | |
| 3.3 | Huấn luyện mô hình ML | Tạ Trung Kiên | |
| 3.4 | Áp dụng SHAP, LIME | Toàn nhóm | |
| 3.5 | Tổng kết | Toàn nhóm | |
| **Chương IV** | Thực nghiệm và đánh giá | | |
| 4.1 | Môi trường thực nghiệm | Nguyễn Ngọc Dương | |
| 4.2 | Kết quả huấn luyện mô hình | Hà Minh Hải | |
| 4.3 | Kết quả XAI và đánh giá | Tạ Trung Kiên | |
| 4.4 | Tổng kết | Toàn nhóm | |

---

| | |
|---|---|
| **GIẢNG VIÊN HƯỚNG DẪN** | **ĐẠI DIỆN NHÓM SINH VIÊN** |
| *(Ký và ghi rõ họ tên)* | Nguyễn Ngọc Dương |
