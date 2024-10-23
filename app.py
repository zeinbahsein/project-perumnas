import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Project", page_icon="🛠️")

# Judul Aplikasi
st.title("Faktor Yang Mempengaruhi Tingkat Keberhasilan Customer Membeli Rumah")

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca data CSV yang diunggah
    zein = pd.read_csv(uploaded_file, delimiter=';')

    # Preprocessing Data
    variabel_kategorik = ['Jenis Kelamin', 'Sumber Informasi', 'Status Pernikahan', 'Pekerjaan']
    label_encoders = {}

    # Encode categorical variables and store the encoders
    for var in variabel_kategorik:
        le = LabelEncoder()
        zein[var] = le.fit_transform(zein[var])
        label_encoders[var] = le

    # Ubah tipe data menjadi numerik
    zein['Nominal Pendapatan'] = zein['Nominal Pendapatan'].str.replace('.', '').str.replace(' ', '').astype(float)
    zein['Dana Yang Tersedia'] = zein['Dana Yang Tersedia'].str.replace('.', '').str.replace(' ', '').astype(float)

    # Klasifikasikan Pendapatan ke dalam kategori
    bins = [0, 1000000, 3000000, 7000000, 10000000, 15000000, float('inf')]
    labels = ["0 - 1 Juta", "1 - 3 Juta", "3 - 7 Juta", "7 - 10 Juta", "10 - 15 Juta", "Diatas 15 Juta"]
    zein['Pendapatan Customer'] = pd.cut(zein['Nominal Pendapatan'], bins=bins, labels=labels, right=False)

    # Gabungkan proyek sesuai dengan permintaan
    proyek_mapping = {
        'BANDUNG I': 'BANDUNG',
        'BANDUNG II': 'BANDUNG',
        'JATIM 1': 'JATIM JATENG',
        'JATIM 2': 'JATIM JATENG',
        'SOLO JOGJA': 'JATIM JATENG',
        'SEMARANG': 'JATIM JATENG',
        'SULSEL': 'Sulawesi',
        'SULUT': 'Sulawesi',
        'SUTRA': 'Sulawesi',
        'SUMUT': 'SUMATERA',
        'LAMPUNG': 'SUMATERA',
        'KEPRI': 'SUMATERA',
        'SUMSEL': 'SUMATERA',
        'MAHATA MARGONDA': 'JABODETABEK',
        'MAHATA TANJUNG BARAT': 'JABODETABEK',
        'MAHATA SERPONG': 'JABODETABEK',
        'PARUNG PANJANG': 'JABODETABEK',
        'CENGKARENG': 'JABODETABEK',
        'DRAMAGA': 'JABODETABEK',
        'EAST POINT': 'JABODETABEK'
    }

    # Menerapkan penggantian proyek
    zein['Proyek'] = zein['Proyek'].replace(proyek_mapping)

    # Menambahkan opsi NASIONAL ke dalam proyek
    proyek_options = zein['Proyek'].unique().tolist()  # Mendapatkan opsi proyek yang unik
    proyek_options.append("SELURUH DAERAH")  # Menambahkan opsi NASIONAL

    # Tambahkan filter untuk proyek
    selected_proyek = st.selectbox("Pilih Daerah", proyek_options)

    # Filter data berdasarkan proyek yang dipilih
    if selected_proyek == "SELURUH DAERAH":
        zein_filtered = zein  # Jika NASIONAL, gunakan semua data
    else:
        zein_filtered = zein[zein['Proyek'] == selected_proyek]

    # Menyiapkan data untuk model
    X = zein_filtered[['Jenis Kelamin', 'Sumber Informasi', 'Dana Yang Tersedia', 'Status Pernikahan', 'Pekerjaan', 'Nominal Pendapatan']]
    y = zein_filtered['Keputusan Akhir']

    # Pisahkan data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardisasi data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Regresi Logistik
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Prediksi
    y_pred = model.predict(X_test_scaled)


   # Hitung jumlah nilai dari setiap kategori Klasifikasi Pendapatan
    st.subheader("Jumlah Pendapatan Di Setiap Daerah")

    # Mendapatkan jumlah nilai untuk setiap kategori di Klasifikasi Pendapatan
    pendapatan_counts = zein_filtered['Pendapatan Customer'].value_counts()

    # Mengurutkan nilai dari yang tertinggi
    sorted_pendapatan = pendapatan_counts.sort_values(ascending=False)

    # Identifikasi dua nilai tertinggi
    top_two = sorted_pendapatan.index[:1]

   # Membuat DataFrame untuk menampilkan pendapatan dalam tabel
    data_pendapatan = {
        'Pendapatan Customer': ['1 - 3 Juta', '3 - 7 Juta', '7 - 10 Juta', '10 - 15 Juta', 'Diatas 15 Juta'],
        'Jumlah Customer': [
            pendapatan_counts.get('1 - 3 Juta', 0),
            pendapatan_counts.get('3 - 7 Juta', 0),
            pendapatan_counts.get('7 - 10 Juta', 0),
            pendapatan_counts.get('10 - 15 Juta', 0),
            pendapatan_counts.get('Diatas 15 Juta', 0)
        ]
    }

    # Konversi menjadi DataFrame
    df_pendapatan = pd.DataFrame(data_pendapatan)

    # Beri warna merah pada dua nilai tertinggi
    def highlight_top_two(row):
        if row['Pendapatan Customer'] in top_two:
            return ['','background-color: green; color: white']
        else:
            return ['', '']

    # Tampilkan tabel dengan highlight pada dua nilai tertinggi
    st.dataframe(df_pendapatan.style.apply(highlight_top_two, axis=1))
    
        # Visualisasi koefisien fitur
    koefisien = model.coef_[0]
    fitur = X.columns
    df_penting = pd.DataFrame({'Fitur': fitur, 'Skala Koefisien': koefisien})

    # Pisahkan koefisien positif dan negatif
    df_positif = df_penting[df_penting['Skala Koefisien'] > 0]
    df_negatif = df_penting[df_penting['Skala Koefisien'] < 0]

    # Mendapatkan fitur yang paling berpengaruh untuk positif dan negatif
    fitur_tertinggi_positif = df_positif.loc[df_positif['Skala Koefisien'].idxmax(), 'Fitur']
    fitur_terendah_negatif = df_negatif.loc[df_negatif['Skala Koefisien'].idxmin(), 'Fitur']

    # Plot koefisien positif dengan altair (horizontal bar chart)
    st.subheader("Variabel Yang Paling Berpengaruh Terhadap Tingkat Keberhasilan Customer Membeli Rumah")
    chart_positif = alt.Chart(df_positif).mark_bar(color='steelblue').encode(
        x=alt.X('Skala Koefisien:Q', title='Skala Koefisien'),
        y=alt.Y('Fitur:N', sort='-x', title='Fitur'),
        tooltip=[alt.Tooltip('Fitur:N', title='Fitur'), alt.Tooltip('Skala Koefisien:Q', title='Skala Koefisien')]
    ).properties(
        title='Variabel Yang Paling Berpengaruh Terhadap Tingkat Keberhasilan Customer Membeli Rumah'
    )

    # Menambahkan label untuk koefisien positif
    label_positif = chart_positif.mark_text(
        align='left',
        baseline='middle',
        dx=3,  # Jarak dari batang
        color='black'  # Warna label hitam
    ).encode(
        text=alt.Text('Skala Koefisien:Q', format='.2f')
    )

    # Gabungkan chart batang dengan label
    st.altair_chart(chart_positif + label_positif, use_container_width=True)

    

    # Tampilkan kalimat fitur paling berpengaruh positif
    st.write(f"Variabel Yang Paling Berpengaruh Terhadap Tingkat Keberhasilan Customer Membeli Rumah adalah Variabel **{fitur_tertinggi_positif}**.")

    # Plot koefisien negatif dengan altair (horizontal bar chart)
    st.subheader("Variabel Yang Paling Tidak Berpengaruh Terhadap Tingkat Keberhasilan Customer Membeli Rumah")
    chart_negatif = alt.Chart(df_negatif).mark_bar(color='salmon').encode(
        x=alt.X('Skala Koefisien:Q', title='Skala Koefisien'),
        y=alt.Y('Fitur:N', sort='x', title='Fitur'),
        tooltip=[alt.Tooltip('Fitur:N', title='Fitur'), alt.Tooltip('Skala Koefisien:Q', title='Skala Koefisien')]
    ).properties(
        title='Variabel Yang Paling Tidak Berpengaruh Terhadap Tingkat Keberhasilan Customer Membeli Rumah'
    )

    # Menambahkan label untuk koefisien negatif
    label_negatif = chart_negatif.mark_text(
        align='right',
        baseline='middle',
        dx=-3,  # Jarak dari batang (ke kiri)
        color='black'  # Warna label hitam
    ).encode(
        text=alt.Text('Skala Koefisien:Q', format='.2f')
    )

    # Gabungkan chart batang dengan label
    st.altair_chart(chart_negatif + label_negatif, use_container_width=True)

    # Tampilkan kalimat fitur paling tidak berpengaruh negatif
    st.write(f"Variabel yang paling tidak berpengaruh terhadap Tingkat Keberhasilan Customer Membeli Rumah adalah Variabel **{fitur_terendah_negatif}**.")

    if st.button("Tampilkan Detail"):

        # Visualisasi jumlah setiap nilai untuk variabel
        st.subheader("Jumlah Setiap Nilai untuk Variabel")
        for var in variabel_kategorik + ['Pendapatan Customer', 'Dana Yang Tersedia']:
            if var in variabel_kategorik:
                # Gunakan label encoder untuk mendapatkan original categories
                original_values = label_encoders[var].classes_
                zein_filtered[var] = zein_filtered[var].apply(lambda x: original_values[x])
    
            color = 'steelblue' if var in df_positif['Fitur'].values else 'salmon'  # Warna sesuai koefisien
            chart = alt.Chart(zein_filtered).mark_bar(color=color).encode(
                x=alt.X('count():Q', title='Jumlah'),
                y=alt.Y(f'{var}:N', sort='-x', title=var),  # Pastikan y-axis menggunakan variabel kategorikal
                tooltip=[alt.Tooltip(f'{var}:N', title=var), alt.Tooltip('count():Q', title='Jumlah')]
            ).properties(
                title=f'Jumlah Setiap Nilai untuk Variabel {var}'
            )
        
            # Menambahkan label jumlah di atas setiap batang
            label = chart.mark_text(
                align='left',
                baseline='middle',
                dx=3,  # Jarak dari batang
                color='black'  # Warna label hitam
            ).encode(
                text=alt.Text('count():Q', format='.0f')  # Menampilkan jumlah sebagai label
            )
        
            # Gabungkan chart batang dengan label
            st.altair_chart(chart + label, use_container_width=True)



   
