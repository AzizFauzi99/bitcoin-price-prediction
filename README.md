# Laporan Proyek Machine Learning - Aziz Fatih Fauzi
## "Prediksi Harga Bitcoin Menggunakan LSTM"

## 1. Domain Proyek
### Latar Belakang
Cryptocurrency, khususnya Bitcoin, telah menjadi subjek yang menarik dalam beberapa tahun terakhir. Harganya yang sangat fluktuatif membuatnya menarik bagi para investor dan trader. Analisis teknis dan prediksi harga menjadi kunci dalam pengambilan keputusan investasi yang cerdas. Salah satu pendekatan yang digunakan adalah penggunaan jaringan saraf LSTM (Long Short-Term Memory) untuk memprediksi harga Bitcoin di masa depan.
### Masalah dan Solusinya
Masalahnya adalah fluktuasi harga Bitcoin yang sulit diprediksi menggunakan metode analisis tradisional. Hal ini dapat diselesaikan dengan menggunakan pendekatan yang lebih canggih seperti LSTM, yang dapat menangkap pola kompleks dalam data historis harga Bitcoin dan memprediksi pergerakan harga di masa depan.
### Referensi Riset
[Price Prediction of Bitcoin Using LSTM Neural Network](https://www.researchgate.net/publication/369425973_Price_Prediction_of_Bitcoin_Using_LSTM_Neural_Network) 
[Prediksi Mata Uang Bitcoin Menggunakan LSTM dan Analisis Sentimen pada Sosial Media](https://ejournal.jak-stik.ac.id/files/journals/1/articles/Vol19No4Des2020/370/submission/proof/370-1-1492-1-10-20210226.pdf)

## 2. Business Understanding
### Problem Statement
1. Volatilitas Harga Bitcoin: Harga Bitcoin sangat fluktuatif, sehingga sulit bagi investor dan trader untuk membuat keputusan investasi yang tepat. Prediksi harga yang akurat menjadi penting untuk mengurangi risiko dan meningkatkan potensi keuntungan.
2. Keterbatasan Metode Analisis Tradisional: Metode analisis tradisional sering kali tidak cukup efektif dalam memprediksi pergerakan harga Bitcoin yang kompleks. Diperlukan pendekatan yang lebih canggih dan adaptif untuk mengatasi tantangan ini.
### Goals
Mengembangkan model prediksi harga Bitcoin menggunakan jaringan saraf LSTM untuk memberikan prediksi harga yang lebih akurat dan membantu para investor dalam pengambilan keputusan investasi.
### Solutions Statement
1. Menggunakan SGD (Stochastic Gradient Descent): Menggunakan optimizer SGD dengan hyperparameter yang dioptimalkan untuk melatih model LSTM. SGD adalah optimizer yang sederhana dan efisien, yang dapat bekerja dengan baik untuk meminimalkan fungsi kerugian dalam model.
2. Menggunakan Adam Optimizer: Menggunakan optimizer Adam dengan hyperparameter yang dioptimalkan. Adam merupakan optimizer yang canggih dan sering digunakan dalam pelatihan model deep learning karena kemampuannya dalam menyesuaikan laju pembelajaran secara adaptif untuk setiap parameter.
3. Peningkatan Baseline Model dengan Hyperparameter Tuning: Melakukan penyetelan hyperparameter pada model LSTM untuk meningkatkan performa. Misalnya, mengubah jumlah unit LSTM dan panjang time steps untuk mencari konfigurasi yang optimal.
4. Metrik Evaluasi: Untuk membandingkan performa kedua optimizer ini, akan digunakan metrik evaluasi yang sama, yaitu Mean Absolute Error (MAE) . Performa kedua optimizer akan dievaluasi berdasarkan nilai-nilai MAE yang dihasilkan dalam memprediksi harga Bitcoin.

## 3. Data Understanding
### Informasi Umum tentang Data
Dataset Bitcoin dari 2017 hingga 2022 dengan 1886 baris mencakup data harga dan volume perdagangan Bitcoin selama periode tersebut. Setiap baris mewakili data harian, yang mencakup informasi seperti harga pembukaan, tertinggi, terendah, dan penutupan, serta volume perdagangan dalam Bitcoin dan dalam mata uang dasar atau konversi (misalnya, dalam USDT untuk pasangan perdagangan BTC/USDT).
### Sumber Data
[Sumber Data](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd)
### Fitur pada Data
Unix Timestamp - Ini adalah timestamp unix atau yang juga dikenal sebagai "Epoch Time".
Date - Timestamp ini adalah UTC Timezone
Symbol - Simbol untuk data time series yang bersangkutan
Open - harga pembukaan periode waktu
High - harga tertinggi dari periode waktu tersebut
Low - harga terendah dari periode waktu tersebut
Close - harga penutupan dari periode waktu tersebut
Volume (Crypto) - volume dalam mata uang yang ditransaksikan. Misalnya, untuk BTC/USDT, ini dalam jumlah BTC
Volume Base Ccy - volume dalam mata uang dasar/konversi. Misalnya, untuk BTC/USDT, ini dalam jumlah USDT
### Exploratory Data Analysis
- Deskripsi variabel: Memahami karakteristik data pada setiap variabel, seperti mean, median, dan distribusi.
![image](assets\1.JPG)
- Cek Missing value
- Analisis univariat pada fitur numerik: Menganalisis distribusi dan statistik deskriptif dari setiap fitur numerik.
- Analisis multivariat pada fitur numerik: Mengeksplorasi hubungan antara beberapa fitur numerik.
- Melihat Matrik Korelasi
- Melihat tren close price setiap harinya

## 4. Data Preparation
1. Drop Data Sebelum 2017
-- Melakukan filter untuk menghapus data sebelum tahun 2017.
2. Splitting Data
-- Memisahkan data menjadi set pelatihan dan set validasi. Set pelatihan digunakan untuk melatih model, sementara set validasi digunakan untuk mengevaluasi performa model.
-- Pembagian ini umumnya dilakukan dengan proporsi tertentu, misalnya 80% data untuk pelatihan dan 20% untuk validasi.
3. Scaling Data
-- Melakukan penskalaan fitur menggunakan MinMaxScaler untuk mengubah nilai fitur ke rentang antara 0 dan 1. Hal ini penting karena LSTM sensitif terhadap skala data.
4. Membuat Dataset untuk Model LSTM
-- Membuat dataset dalam bentuk time series yang sesuai untuk model LSTM. Ini melibatkan pembuatan pasangan data input-output dengan jendela waktu (time steps) tertentu.
