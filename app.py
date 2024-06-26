import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from streamlit_option_menu import option_menu

data = pd.read_csv('datagastritis.csv', encoding='latin1')
# Mapping for binary conversion
binary_mapping = {'Ya': 1, 'Tidak': 0, 'Gastritis': 1, 'Non Gastritis': 0}

# Apply the mapping to convert categorical data to binary
df_binary = data.applymap(lambda x: binary_mapping.get(x, x))
#Menghapus Kolom Ynag tidak dipakai
df = df_binary.drop(columns=['NO', 'Nama', 'ICD-10','Jenis Kelamin','Usia', 'TB (CM)', 'BB (KG)'])
# Membagi data menjadi data feature(X) dan data target(Y)
X = df.drop(columns=['Hasil'])
y = df['Hasil']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Menampilkan aplikasi Streamlit
def main():
    st.title("Klasifikasi Penyakit Gastritis")
    st.subheader("Kelompok 7")
    st.write("1. AKH. RAIHAN GIMNASTIAR RAKHMAN 210411100232")
    st.write("2. A. Makmun Alji 210411100241")
    page = option_menu(None, ["Data","Analisa Data","Preprocessing", 'Modeling', "Implementasi"],
                    icons=['grid', 'wrench', 'chart-line', 'play'],
                    menu_icon="cast", default_index=0, orientation="horizontal")
    page
    if page == "Data":
        st.header("Tentang Data")
        st.subheader("Data Awal")
        st.write(data)

    elif page == "Analisa Data":
        st.header("Analisa Data")
        st.subheader("Analisa Fungsi Agregat")
        average_usia = df_binary['Usia'].mean()
        average_tb = df_binary['TB (CM)'].mean()
        average_bb = df_binary['BB (KG)'].mean()
        max_usia = df_binary['Usia'].max()
        max_tb = df_binary['TB (CM)'].max()
        max_bb = df_binary['BB (KG)'].max()
        agregat = {
            'Average Usia': [average_usia],
            'Average TB': [average_tb],
            'Average BB': [average_bb],
            'Max Usia': [max_usia],
            'Max TB': [max_tb],
            'Max BB': [max_bb]
        }
        st.write(agregat)
        st.subheader("Analisa Tren")
        # Plot untuk kolom 'Usia'
        st.write("#### Tren Usia")
        fig_usia = plt.figure(figsize=(10, 6))
        plt.plot(df_binary['Usia'], marker='o', linestyle='-', color='b')
        plt.xlabel('Index')
        plt.ylabel('Usia')
        plt.grid(True)
        st.pyplot(fig_usia)

        # Plot untuk kolom 'TB (CM)'
        st.write("#### Tren Tinggi Badan (TB)")
        fig_tb = plt.figure(figsize=(10, 6))
        plt.plot(df_binary['TB (CM)'], marker='o', linestyle='-', color='g')
        plt.xlabel('Index')
        plt.ylabel('TB (CM)')
        plt.grid(True)
        st.pyplot(fig_tb)

        # Plot untuk kolom 'BB (KG)'
        st.write("#### Tren Berat Badan (BB)")
        fig_bb = plt.figure(figsize=(10, 6))
        plt.plot(df_binary['BB (KG)'], marker='o', linestyle='-', color='r')
        plt.xlabel('Index')
        plt.ylabel('BB (KG)')
        plt.grid(True)
        st.pyplot(fig_bb)

        st.subheader("Analisa Korelasi")
        # Hitung matriks korelasi
        correlation_matrix = df_binary[['Usia', 'TB (CM)', 'BB (KG)', 'Nyeri', 'Mual', 'Muntah', 'Keram', 'Hasil']].corr()

        # Plot matriks korelasi menggunakan seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Matriks Korelasi')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # Menampilkan plot di Streamlit dengan objek fig
        fig, ax = plt.subplots()
        ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('Matriks Korelasi')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        
    # Preprocessing menu
    elif page == "Preprocessing":
        st.header("Preprocessing")
        st.subheader("Data Awal")
        st.write(df)
        st.subheader("Split Data Dan Normalisasi Data")
        # Membuat dua kolom
        col1, col2 = st.columns(2)

        # Menampilkan "Data Training" di kolom pertama
        with col1:
            st.write("Data Training")
            st.write(X_train)
            st.write(X_train.shape)

        # Menampilkan "Data Testing" di kolom kedua
        with col2:
            st.write("Data Testing")
            st.write(X_test)
            st.write(X_test.shape)

    elif page == "Modeling":
        st.header("Model Yang digunakan adalah KNN")
        st.subheader("Laporan Akurasi")
        # Hitung dan tampilkan akurasi
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi model : {accuracy}")
        # Membuat DataFrame dari laporan akurasi
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        # Menampilkan DataFrame sebagai tabel
        st.table(df_report[['precision', 'recall', 'f1-score', 'support']])
        
    elif page == "Implementasi":
        st.title("Implementasi")
        # Pilihan untuk nyeri, mual, muntah, keram
        st.subheader("Gejala:")
        new_nyeri = st.selectbox("Nyeri:", ["Tidak", "Iya"])
        new_mual = st.selectbox("Mual:", ["Tidak", "Iya"])
        new_muntah = st.selectbox("Muntah:", ["Tidak", "Iya"])
        new_keram = st.selectbox("Keram:", ["Tidak", "Iya"])
        
        # Mengubah pilihan menjadi nilai biner
        new_nyeri = 1 if new_nyeri == "Iya" else 0
        new_mual = 1 if new_mual == "Iya" else 0
        new_muntah = 1 if new_muntah == "Iya" else 0
        new_keram = 1 if new_keram == "Iya" else 0
        
        # Prediksi menggunakan model yang telah dilatih
        if st.button("Prediksi"):
            new_data = scaler.transform([[new_nyeri, new_mual, new_muntah, new_keram]])
            predicted_class = knn.predict(new_data)
            # Konversi nilai prediksi menjadi label yang lebih deskriptif
            if predicted_class[0] == 1:
                prediksi_label = "Gastritis"
            else:
                prediksi_label = "Non Gastritis"
            
            st.write(f"Prediksi: {prediksi_label}")
if __name__ == "__main__":
    main()
