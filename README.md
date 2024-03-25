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
- Deskripsi variabel: Memahami karakteristik data pada setiap variabel, seperti mean, median, dan distribusi.<br>
![image](https://github.com/AzizFauzi99/bitcoin-price-prediction/assets/92005833/408dfa43-3139-4c42-92c6-a09e29514d8c) <br>
- Cek Missing value <br>
![image](https://github.com/AzizFauzi99/bitcoin-price-prediction/assets/92005833/62e9e191-d603-40f6-94cc-cc56f799c8dd) <br>
   Tidak ada missing value di data ini <br>
- Analisis univariat pada fitur numerik: Menganalisis distribusi dan statistik deskriptif dari setiap fitur numerik.<br> 
![image](https://github.com/AzizFauzi99/bitcoin-price-prediction/assets/92005833/1a413ccc-5f1a-4a30-a5ea-2fe7ce9ce075) <br>
- Analisis multivariat pada fitur numerik: Mengeksplorasi hubungan antara beberapa fitur numerik. <br>
![image](https://github.com/AzizFauzi99/bitcoin-price-prediction/assets/92005833/c684b0f8-6026-43e8-b567-2b93314e82a5) <br>
    Pola Persebaran data tersebut memiliki korelasi positif. Hal ini ditandai dengan meningkatnya variabel pada sumbu y saat terjadi peningkatan variabel pada sumbu x <br>
- Melihat Matrik Korelasi <br>
![image](https://github.com/AzizFauzi99/bitcoin-price-prediction/assets/92005833/51974620-f725-4e57-a39d-a63fe9e3b764) <br>
    Matriks di atas menunjukkan Volume USD dan Volume BTC tidak terlalu berkorelasi terhadap close price <br>
- Melihat tren close price setiap harinya<br>
![image](https://github.com/AzizFauzi99/bitcoin-price-prediction/assets/92005833/f4866a44-4fd9-4e47-bae0-42e6f4699f93) <br>
   Trend cenderung fluktuatif tetapi mengarah pada kenaikan <br>
  
## 4. Data Preparation
1. Drop Data Sebelum 2017 <br>
- Melakukan filter untuk menghapus data sebelum tahun 2017, untuk fokus pada data terbaru yang lebih relevan. <br>
2. Splitting Data <br>
- Memisahkan data menjadi set pelatihan dan set validasi. Set pelatihan digunakan untuk melatih model, sementara set validasi digunakan untuk mengevaluasi performa model. <br>
- Pembagian ini umumnya dilakukan dengan proporsi tertentu, misalnya 80% data untuk pelatihan dan 20% untuk validasi. <br>
3. Scaling Data <br>
- Melakukan penskalaan fitur menggunakan MinMaxScaler untuk mengubah nilai fitur ke rentang antara 0 dan 1. Hal ini penting karena LSTM sensitif terhadap skala data. <br>
- Scalling dilakukan setelah spilitting agar tidak terjadi data leakage<br>
4. Membuat Dataset untuk Model LSTM <br>
- Membuat dataset dalam bentuk time series yang sesuai untuk model LSTM. Ini melibatkan pembuatan pasangan data input-output dengan jendela waktu (time steps) tertentu.<br>

## 5. Modeling
### Tahapan dan Parameter Pemodelan
1. Model Pertama:
- Menggunakan model Sequential dengan dua lapisan LSTM dan dua lapisan Dense.
- Optimizer: SGD dengan momentum 0.9.
- Loss function: Huber loss.
- Metrik evaluasi: Mean Absolute Error (MAE).
- Callback: EarlyStopping untuk mencegah overfitting.
2. Model Perbaikan:
- Sama seperti model pertama, namun menggunakan optimizer Adam.
### Kelebihan dan Kekurangan Algoritma (Optimizer)
#### SGD:
- Kelebihan: Cocok untuk masalah yang memiliki banyak data dan dapat mencapai minimum global.
- Kekurangan: Kurang efisien dalam masalah kompleks dan membutuhkan penyetelan hyperparameter yang lebih teliti.
#### Adam:
- Kelebihan: Lebih efisien dalam menemukan minimum global, cocok untuk masalah dengan data yang besar atau kompleks.
- Kekurangan: Dapat overfit jika tidak diatur dengan benar, dan dapat lebih lambat dalam mencapai konvergensi pada beberapa kasus.

### Proses Improvement dengan Hyperparameter Tuning
Pada model kedua, terdapat peningkatan performa dengan mengubah optimizer dari SGD menjadi Adam. Adam memberikan kemampuan adaptasi yang lebih baik terhadap laju pembelajaran untuk setiap parameter, sehingga membantu model mencapai konvergensi yang lebih baik. Hal ini menghasilkan penurunan MAE dari 0.0463 menjadi 0.0241, menunjukkan bahwa model kedua lebih baik dalam memprediksi harga Bitcoin.

## 6. Evaluation
1. Metrik Evaluasi: Mean Absolute Error (MAE).

2. Hasil Proyek Berdasarkan Metrik Evaluasi:
- Model pertama memiliki MAE sebesar 0.0463.
![image](https://github.com/AzizFauzi99/bitcoin-price-prediction/assets/92005833/fa56ffbf-5b01-40e7-905e-3907f43a6cf8)
- Setelah melakukan perbaikan dengan menggunakan optimizer Adam, MAE turun menjadi 0.0241.
![image](https://github.com/AzizFauzi99/bitcoin-price-prediction/assets/92005833/7e74780c-207d-4b9e-81aa-acc34aab584e)
- Penurunan MAE menunjukkan bahwa model kedua lebih baik dalam memprediksi harga Bitcoin dibandingkan dengan model pertama.

3. Penjelasan Metrik Evaluasi (MAE):
- MAE mengukur rata-rata absolut dari selisih antara nilai prediksi dan nilai observasi aktual.
- Formula MAE = (1/n) Σ|yi - ŷi|
- MAE menghasilkan skor absolut yang mewakili kesalahan rata-rata dari model dalam memprediksi nilai sebenarnya.
- Semakin rendah nilai MAE, semakin baik model dalam memprediksi nilai sebenarnya.
