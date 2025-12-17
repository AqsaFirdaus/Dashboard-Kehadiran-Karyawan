import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@st.cache_resource
def get_engine():
    username = "magang"
    password = "magang"
    host = "110.239.64.117"
    port = "3306"
    database = "db_permata"

    return create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")

@st.cache_data(ttl=21600) # 6 jam
def load_data_bersih():
    query = """
    SELECT tk.nik, tk.nama, tp.nama as plant, tds.department,
        tr.tgl as tanggal, tr.jammasuk, tr.jampulang, tr.real_istirahat
    FROM tbl_karyawan tk
    JOIN tbl_department_staff tds
      ON tds.id = tk.iddepartemen
    JOIN tbl_rekapabsen tr
      ON tr.nik = tk.nik
      AND tr.tgl >= '2025-08-01'
      AND tr.tgl <  '2025-12-01'
    JOIN tbl_plant tp
      ON tp.idplant = tr.idplant
    WHERE tk.f_aktif = 1
    """

    engine = get_engine()
    df_bersih = pd.read_sql(query, con=engine)

    # Data cleaning
    df_bersih = df_bersih.drop_duplicates(subset=['nik', 'tanggal'], keep='last')
    df_bersih['tanggal'] = pd.to_datetime(df_bersih['tanggal'], errors='coerce')
    df_bersih['department'] = df_bersih['department'].fillna("Unknown")
    df_bersih = df_bersih.dropna(subset=['jammasuk'])
    df_bersih['jampulang'] = df_bersih['jampulang'].fillna(df_bersih['tanggal'] + pd.Timedelta(hours=17))
    df_bersih['lama_kerja'] = (df_bersih['jampulang'] - df_bersih['jammasuk']).dt.total_seconds() / 3600
    df_bersih['terlambat'] = df_bersih['jammasuk'].dt.time > pd.to_datetime("08:00:00").time()
    df_bersih['pulang_cepat'] = df_bersih['jampulang'].dt.time < pd.to_datetime("17:00:00").time()

    day_map = {
        'Monday': 'Senin',
        'Tuesday': 'Selasa',
        'Wednesday': 'Rabu',
        'Thursday': 'Kamis',
        'Friday': 'Jumat',
        'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }

    df_bersih['hari'] = df_bersih['tanggal'].dt.day_name().map(day_map)
    df_bersih['minggu_ke'] = df_bersih['tanggal'].dt.isocalendar().week
    df_bersih['bulan'] = df_bersih['tanggal'].dt.month
    df_bersih['bulan_nama'] = df_bersih['tanggal'].dt.month_name()
    df_bersih['tahun'] = df_bersih['tanggal'].dt.year
    df_bersih = df_bersih[df_bersih['tahun'] == 2025]
    return df_bersih

@st.cache_data
def create_visual_data(df_filter):
    # Persentase terlambat per departemen
    agregasi_terlambat = df_filter.groupby(['department', 'bulan_nama'])['terlambat'].mean().reset_index()
    heatmap = (agregasi_terlambat.pivot(index='department', columns='bulan_nama', values='terlambat') * 100).round(1).fillna(0)

    # Kehadiran mingguan
    # tren_mingguan = (df_filter[(df_filter['minggu_ke'] >= 32) & (df_filter['minggu_ke'] < 45)].groupby(['minggu_ke'])['nik'].count().reset_index().rename(columns={'nik': 'jumlah_hadir'}))
    tren_mingguan = (df_filter[df_filter['minggu_ke'] >= 32].groupby(['minggu_ke'])['nik'].count().reset_index().rename(columns={'nik': 'jumlah_hadir'}))

    # Top 10 Keterlambatan Karyawan
    top_terlambat = df_filter.groupby(['nik', 'nama'])['terlambat'].sum().reset_index()
    top_terlambat = top_terlambat.rename(columns={'terlambat': 'jumlah_terlambat'}).sort_values(by='jumlah_terlambat', ascending=False).head(10)

    # Rata-rata lama kerja per departemen
    lama_kerja = df_filter.groupby('department')['lama_kerja'].mean().reset_index().round(2)
    lama_kerja = lama_kerja.sort_values(by='lama_kerja', ascending=False).head(15)

    # Terlambat vs tidak terlambat
    total_terlambat = df_filter['terlambat'].sum()
    total_tidak_terlambat = len(df_filter) - total_terlambat

    return heatmap, tren_mingguan, top_terlambat, lama_kerja, total_terlambat, total_tidak_terlambat

def persentase_terlambat_per_karyawan(df):
    df['terlambat_int'] = df['terlambat'].astype(int)
    agg = df.groupby(['nik', 'nama', 'bulan_nama']).agg(
        jumlah_hadir=('nik', 'count'),
        jumlah_terlambat=('terlambat_int', 'sum')
    ).reset_index()
    agg['persentase_terlambat'] = (agg['jumlah_terlambat'] / agg['jumlah_hadir'] * 100).round(1)
    hasil = agg[['nama', 'bulan_nama', 'jumlah_hadir', 'jumlah_terlambat', 'persentase_terlambat']]

    return hasil

def get_scaler():
    return StandardScaler()

def get_kmeans():
    return KMeans(n_clusters=3, random_state=42, n_init=10)

def get_pca():
    return PCA(n_components=2, random_state=42)

st.set_page_config(layout="wide", page_title="Dashboard Kehadiran")
st.title("📊 Dashboard Rekap Absen Karyawan")

df_bersih = load_data_bersih()

# Membuat beberapa filter
filter1, filter2, filter3, filter4 = st.columns(4)

with filter1:
    tanggal_awal = st.date_input("📅 Tanggal Awal", value=df_bersih['tanggal'].min(), min_value=df_bersih['tanggal'].min(), max_value=df_bersih['tanggal'].max())

with filter2:
    tanggal_akhir = st.date_input("📅 Tanggal Akhir", value=df_bersih['tanggal'].max(), min_value=df_bersih['tanggal'].min(), max_value=df_bersih['tanggal'].max())

df_filter = df_bersih[(df_bersih['tanggal'] >= pd.to_datetime(tanggal_awal)) &(df_bersih['tanggal'] <= pd.to_datetime(tanggal_akhir))]

with filter3:
    plant_list = ['Semua'] + sorted(df_filter['plant'].dropna().unique().tolist())
    selected_plant = st.selectbox("🏭 Plant", plant_list)

    if selected_plant != 'Semua':
        df_filter = df_filter[df_filter['plant'] == selected_plant]

with filter4:
    dept_list = ['Semua'] + sorted(df_filter['department'].dropna().unique().tolist())
    selected_departemen = st.selectbox("🏢 Departemen", dept_list)

    if selected_departemen != 'Semua':
        df_filter = df_filter[df_filter['department'] == selected_departemen]

heatmap, tren_mingguan, top_terlambat, lama_kerja, total_terlambat, total_tidak_terlambat = create_visual_data(df_filter)

# Perhitungan metrik
total_karyawan = df_filter['nik'].nunique()
persentase_terlambat = df_filter['terlambat'].mean() * 100
persentase_pulang_cepat = df_filter['pulang_cepat'].mean() * 100
rata_jam_kerja = df_filter['lama_kerja'].mean()
rata_istirahat = df_filter['real_istirahat'].mean()

# Membuat beberapa metrik
metrik1, metrik2, metrik3, metrik4, metrik5 = st.columns(5)
metrik1.metric("👥 Total Karyawan", f"{total_karyawan}")
metrik2.metric("⏰ Persentase Terlambat", f"{persentase_terlambat:.1f}%")
metrik3.metric("🏃‍♂️ Persentase Pulang Cepat", f"{persentase_pulang_cepat:.1f}%")
metrik4.metric("💼 Rata-rata Jam Kerja", f"{rata_jam_kerja:.1f} Jam")
metrik5.metric("☕ Rata-rata Istirahat", f"{rata_istirahat:.1f} Jam")

# Clustering
df_cluster = df_filter.copy()
df_cluster = df_cluster[(df_cluster['lama_kerja'] >= 0) & (df_cluster['lama_kerja'] <= 24)]
df_cluster['terlambat'] = df_cluster['terlambat'].astype(int)
df_cluster['pulang_cepat'] = df_cluster['pulang_cepat'].astype(int)
df_cluster = df_cluster.sort_values(by=['nik']).reset_index(drop=True)

data_cluster = df_cluster.groupby(['nik', 'nama']).agg({
    'terlambat': 'mean',
    'pulang_cepat': 'mean',
    'lama_kerja': 'mean',
    'real_istirahat': 'mean'
}).reset_index().round(2).rename(columns={'real_istirahat':'istirahat'})

fitur = data_cluster[['terlambat', 'pulang_cepat', 'lama_kerja', 'istirahat']]

# Standarisasi
scaler = get_scaler()
X_scaled = scaler.fit_transform(fitur)

# KMeans
kmeans = get_kmeans()
data_cluster['cluster'] = kmeans.fit_predict(X_scaled)

rata_cluster = data_cluster.groupby('cluster')[['terlambat','pulang_cepat','lama_kerja','istirahat']].mean()
faktor_cluster = rata_cluster['terlambat'] # Urutkan cluster berdasarkan keterlambatan
urutan_cluster = faktor_cluster.sort_values().index.tolist()

# Mapping kategori
kategori_mapping = {
    urutan_cluster[0]: "Disiplin",
    urutan_cluster[1]: "Normal",
    urutan_cluster[2]: "Kurang Disiplin"
}

data_cluster['kategori'] = data_cluster['cluster'].map(kategori_mapping)

# PCA untuk visualisasi
pca = get_pca()
X_pca = pca.fit_transform(X_scaled)

data_cluster['PCA1'] = X_pca[:,0]
data_cluster['PCA2'] = X_pca[:,1]

# Rata-rata tiap cluster
kategori_cluster = data_cluster.groupby('kategori')[['terlambat','pulang_cepat','lama_kerja','istirahat']].mean().reset_index()

# Visualisasi
barchart1, barchart2 = st.columns(2)

with barchart1:
    barchart_terlambat = px.bar(
        top_terlambat,
        x='jumlah_terlambat',
        y='nama',
        orientation='h',
        title='Top 10 Karyawan Paling Sering Terlambat',
        color='jumlah_terlambat',
        color_continuous_scale=['#ffb3b3', '#d00000'],
        labels={"jumlah_terlambat": "Jumlah Terlambat", "nama": "Nama Karyawan"})
    barchart_terlambat.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(barchart_terlambat, use_container_width=True)

with barchart2:
    barchart_rata_kerja = px.bar(
        lama_kerja,
        x='lama_kerja',
        y='department',
        orientation='h',
        title='Rata-rata Jam Kerja per Departemen',
        color='lama_kerja',
        color_continuous_scale=['#90e0ef', '#0077b6'],
        labels={"lama_kerja": "Jam Kerja","department": " Departemen"})
    barchart_rata_kerja.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(barchart_rata_kerja, use_container_width=True)

layout_kiri, layout_kanan = st.columns([1.2, 2])

with layout_kiri:
    pie_df = pd.DataFrame({
        'Status': ['Terlambat', 'Tidak Terlambat'],
        'Jumlah': [total_terlambat, total_tidak_terlambat]
    })

    pie_chart = px.pie(
        pie_df,
        names='Status',
        values='Jumlah',
        title='Proporsi Karyawan Terlambat vs Tidak Terlambat',
        color='Status',
        color_discrete_map={
            'Terlambat': '#e63946',
            'Tidak Terlambat': '#0077b6'},
        height=400)
    pie_chart.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(pie_chart, use_container_width=True)

    urutan_bulan = [
        "January","February","March","April","May","June","July",
        "August","September","October","November","December"
    ]

    bulan_tersedia = [m for m in urutan_bulan if m in heatmap.columns]
    heatmap = heatmap[bulan_tersedia]
    heatmap_persentase = go.Figure(data=go.Heatmap(
        z=heatmap.values,
        x=heatmap.columns.tolist(),
        y=heatmap.index.tolist(),
        colorscale='Reds',
        colorbar=dict(title='Persentase (%)'),
        text=heatmap.values,
        texttemplate="%{text:.1f}%"
    ))

    heatmap_persentase.update_layout(
        title='Persentase Keterlambatan per Departemen',
        xaxis_title='Bulan',
        yaxis_title='Departemen',
        xaxis=dict(tickangle=-90),
        height=800
    )

    st.plotly_chart(heatmap_persentase, use_container_width=True)

with layout_kanan:
    linechart = px.line(
        tren_mingguan,
        x='minggu_ke',
        y='jumlah_hadir',
        markers=True,
        title='Tren Kehadiran Mingguan')
    linechart.update_layout(yaxis_title="Jumlah Kehadiran", xaxis_title="Minggu Ke-", height=400)
    st.plotly_chart(linechart, use_container_width=True)

    scatterplot = px.scatter(
        data_cluster,
        x='PCA1',
        y='PCA2',
        color='kategori',
        title='Hasil Clustering Karyawan',
        hover_data=['nik', 'nama'])
    scatterplot.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2", height=470)
    st.plotly_chart(scatterplot, use_container_width=True)

    st.subheader("Karakteristik Setiap Cluster")
    kategori_cluster = kategori_cluster.rename(columns={
        'kategori': 'Kategori',
        'terlambat': 'Terlambat',
        'pulang_cepat': 'Pulang Cepat',
        'lama_kerja': 'Jam Kerja',
        'istirahat': 'Istirahat'})

    st.dataframe(kategori_cluster.style.format({
    'Terlambat': '{:.0%}',
    'Pulang Cepat': '{:.0%}',
    'Jam Kerja': '{:.1f} jam',
    'Istirahat': '{:.1f} jam'}),
    use_container_width=True, hide_index=True)

if selected_plant != "Semua" or selected_departemen != "Semua":
    df_persen = persentase_terlambat_per_karyawan(df_filter)
    df_persen['bulan_nama'] = pd.Categorical(df_persen['bulan_nama'], categories=urutan_bulan, ordered=True)
    df_persen = df_persen.sort_values(['nama', 'bulan_nama'])
    df_persen = df_persen.merge(data_cluster[['nama', 'kategori']],on=['nama'],how='left')

    styled_df = df_persen.style.format({
    'persentase_terlambat': '{:.1f}%'
    })

    st.dataframe(styled_df, use_container_width=True)

else:
    if selected_plant == "Semua" and selected_departemen == "Semua":
        with st.expander("💡 Insight"):
            st.markdown("""
              ### ⏰ 1. Karyawan Paling Sering Terlambat
              - Nama seperti Zaenal Ariadi, Eko Andrianto, dan Muhammad Ravvi memiliki frekuensi terlambat tertinggi (mendekati ±100 kali).
              - Masalah keterlambatan tidak merata, tetapi terkonsentrasi pada individu tertentu.

              ---

              ### 💼 2. Rata-rata Jam Kerja per Departemen
              - Jam kerja tiap departemen relatif stabil, namun terdapat perbedaan kecil antar departemen.
              - Beberapa departemen seperti HRD, General Affair, dan Kesehatan memiliki jam kerja lebih tinggi dibanding yang lain, menandakan adanya perbedaan beban kerja antar departemen.

              ---

              ### ⚖️ 3. Proporsi Terlambat vs Tidak Terlambat
              - Proporsinya sangat seimbang (±50–50).
              - Meskipun banyak karyawan terlambat, separuh karyawan tetap disiplin.
              - Masalah kedisiplinan tidak menyeluruh, hanya pada sebagian populasi.

              ---

              ### 📈 4. Tren Kehadiran Mingguan
              - Tren cenderung meningkat dari minggu ke-32 hingga minggu ke-45.
              - Ada sedikit penurunan di minggu ke-36, kemungkinan karena libur besar atau cuti bersama.

              ---

              ### 🔴 5. Persentase Keterlambatan per Departemen (Heatmap)
              - Beberapa departemen memiliki tingkat keterlambatan sangat tinggi, misalnya SATPAM, SCM, PPIC, EXIM, HR & GA, EDP (50–75%).
              - Ada departemen sangat disiplin seperti R&D, Internal Audit, Kendaraan.
              - Pola menunjukkan keterlambatan terstruktur di beberapa departemen.

              ---

              ### 🧩 6. Hasil Clustering Karyawan
              - Terdapat 3 kelompok karyawan berdasarkan perilaku kehadiran: Disiplin, Kurang Disiplin, dan Normal.
              - Cluster Kurang Disiplin sangat jelas terpisah.
              - Cluster Disiplin memiliki jam kerja lebih panjang dan tingkat pulang cepat rendah.
              - Cluster Normal memiliki variabilitas cukup besar.

              ---

              ### 📋 7. Karakteristik Tiap Cluster
              - Cluster kurang disiplin memiliki persentase terlambat tertinggi.
              - Cluster normal berada di tengah dengan variasi perilaku yang cukup besar.
              - Jam kerja bukan indikator kedisiplinan utama; masalah terbesar ada pada pola kedatangan, bukan total jam kerja.
              """)