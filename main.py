import streamlit as st
st.set_page_config(page_title="Data Analysis and Visualization", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Custom CSS untuk styling dark mode keseluruhan
st.markdown("""
    <style>
    body, .main, .block-container, .sidebar .sidebar-content {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    .stButton>button {
        text-color: white;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stSidebar {
        background-color: #252526 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Data Analysis and Visualization")

# Sidebar dengan logo dan menu navigasi yang diperbarui
svg_logo = """
    <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
        <circle cx="100" cy="100" r="80" fill="white" />
        <text x="50%" y="50%" font-size="25" fill="black" text-anchor="middle" dy=".3em">KOSAN ACONG</text>
    </svg>
"""
st.sidebar.markdown(svg_logo, unsafe_allow_html=True)

st.sidebar.markdown("## 📌 Navigation Menu")
menu = st.sidebar.selectbox("Pilih Menu ", 
                            ["Upload Data", "Visualisasi Data", "Distribusi Data", "Analisis Korelasi", "Prediksi Model", "Clustering", "Data Mining"],
                            format_func=lambda x: f"⚡ {x}")

# Menu: Upload Data
if menu == "Upload Data":
    st.header("📂 Upload Data")
    file1 = st.file_uploader("Pilih file CSV pertama", type=["csv"], key="file1")
    file2 = st.file_uploader("Pilih file CSV kedua", type=["csv"], key="file2")
    if file1 is not None and file2 is not None:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        st.subheader("Dataset 1 Preview")
        st.dataframe(df1.head())
        st.subheader("Dataset 2 Preview")
        st.dataframe(df2.head())
        st.session_state["data1"] = df1
        st.session_state["data2"] = df2



# Menu: Visualisasi Data (Histogram & Descriptive Statistics)
elif menu == "Visualisasi Data":
    st.header("📊 Data Visualization")
    if "data1" in st.session_state and "data2" in st.session_state:
        dataset_option = st.selectbox("Select Dataset", ["Dataset 1", "Dataset 2"])
        df = st.session_state["data1"] if dataset_option=="Dataset 1" else st.session_state["data2"]
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        st.subheader("Histogram")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select Column", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, color="#4B79A1", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for histogram.")
    else:
        st.warning("Please upload both datasets in 'Upload Data'.")

# Menu: Distribusi Data (Pairplot)
elif menu == "Distribusi Data":
    st.header("📦 Data Distribution")
    if "data1" in st.session_state and "data2" in st.session_state:
        dataset_option = st.selectbox("Select Dataset", ["Dataset 1", "Dataset 2"])
        df = st.session_state["data1"] if dataset_option=="Dataset 1" else st.session_state["data2"]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            cols_to_plot = st.multiselect("Select Columns for Pairplot", numeric_cols, default=numeric_cols)
            if cols_to_plot:
                st.write("Generating pairplot for columns:", cols_to_plot)
                pair_grid = sns.pairplot(df[cols_to_plot])
                st.pyplot(pair_grid.fig)
            else:
                st.warning("Select at least one column.")
        else:
            st.warning("Not enough numeric columns for pairplot.")
    else:
        st.warning("Please upload both datasets in 'Upload Data'.")

# Menu: Analisis Korelasi (Heatmap)
elif menu == "Analisis Korelasi":
    st.header("📈 Correlation Analysis")
    if "data1" in st.session_state and "data2" in st.session_state:
        dataset_option = st.selectbox("Select Dataset", ["Dataset 1", "Dataset 2"])
        df = st.session_state["data1"] if dataset_option=="Dataset 1" else st.session_state["data2"]
        df_numeric = df.select_dtypes(include=np.number)
        st.subheader("Correlation Heatmap")
        if not df_numeric.empty:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns for correlation analysis.")
    else:
        st.warning("Please upload both datasets in 'Upload Data'.")

# Menu: Prediksi Model (Linear Regression)
elif menu == "Prediksi Model":
    st.header("🤖 Linear Regression Model")
    if "data1" in st.session_state and "data2" in st.session_state:
        dataset_option = st.selectbox("Select Dataset for Prediction", ["Dataset 1", "Dataset 2"])
        df = st.session_state["data1"] if dataset_option=="Dataset 1" else st.session_state["data2"]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            target = st.selectbox("Select Target Column", numeric_cols)
            features = st.multiselect("Select Feature Columns", numeric_cols, default=[col for col in numeric_cols if col != target])
            if st.button("Train Model"):
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("Model Evaluation")
                st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
                st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
                st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, color="green", alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for prediction.")
    else:
        st.warning("Please upload both datasets in 'Upload Data'.")

# Menu: Clustering (K-Means)
elif menu == "Clustering":
    st.header("🧩 Clustering with K-Means")
    if "data1" in st.session_state and "data2" in st.session_state:
        dataset_option = st.selectbox("Select Dataset for Clustering", ["Dataset 1", "Dataset 2"])
        df = st.session_state["data1"] if dataset_option=="Dataset 1" else st.session_state["data2"]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Dataset must have at least 2 numeric columns for clustering.")
        else:
            st.markdown("### Select features for clustering")
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Feature 1", numeric_cols, key="clust_feat1")
            with col2:
                feature2 = st.selectbox("Feature 2", numeric_cols, key="clust_feat2")
            
            k = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
            
            if feature1 == feature2:
                st.warning("Please select two distinct features.")
            else:
                X = df[[feature1, feature2]].dropna().copy()
                if X.empty:
                    st.error("No data available for the selected features.")
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    clusters = kmeans.fit_predict(X)
                    X["Cluster"] = clusters.astype(str)
                    
                    st.markdown("### Clustering Visualization")
                    fig, ax = plt.subplots(figsize=(8,6))
                    sns.scatterplot(data=X, x=feature1, y=feature2, hue="Cluster", palette="viridis", s=100, ax=ax)
                    centers = kmeans.cluster_centers_
                    ax.scatter(centers[:,0], centers[:,1], c="red", s=200, marker="X", label="Centroids")
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.markdown("#### Centroid Coordinates")
                    centroids_df = pd.DataFrame(centers, columns=[feature1, feature2])
                    st.dataframe(centroids_df)
    else:
        st.warning("Please upload both datasets in 'Upload Data'.")

# Menu: Data Mining (Decision Tree Classification)
elif menu == "Data Mining":
    st.header("🧠 Data Mining: Decision Tree Classification")
    if "data1" in st.session_state and "data2" in st.session_state:
        dataset_option = st.selectbox("Select Dataset for Data Mining", ["Dataset 1", "Dataset 2"])
        df = st.session_state["data1"] if dataset_option=="Dataset 1" else st.session_state["data2"]
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        target = st.selectbox("Select Target Column (Categorical)", df.columns)
        df[target] = df[target].astype(str)
        features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target],
                                  default=[col for col in df.columns if col != target])
        if st.button("Train Decision Tree Model"):
            from sklearn.tree import DecisionTreeClassifier, plot_tree
            from sklearn.metrics import accuracy_score, classification_report
            X = df[features]
            y = df[target]
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                st.info(f"Non-numeric features detected: {non_numeric}. Applying one-hot encoding.")
                X = pd.get_dummies(X, columns=non_numeric)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.subheader("Model Evaluation")
            st.write(f"Accuracy: {acc:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.subheader("Decision Tree Visualization")
            fig, ax = plt.subplots(figsize=(12,8))
            plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, ax=ax)
            st.pyplot(fig)


# Menu: Analisis Sentimen
elif menu == "Analisis Sentimen":
    st.header("🔍 Analisis Sentimen Kata")
    
    if "data1" in st.session_state or "data2" in st.session_state:
        dataset_option = st.selectbox("Pilih Dataset untuk Analisis Sentimen", ["Dataset 1", "Dataset 2"])
        df = st.session_state["data1"] if dataset_option == "Dataset 1" else st.session_state["data2"]
        
        # Pilih kolom teks untuk analisis sentimen
        text_columns = df.select_dtypes(include="object").columns
        text_column = st.selectbox("Pilih kolom teks", text_columns)
        
        # Fungsi untuk menganalisis sentimen
        def analyze_sentiment(text):
            # Membuat objek TextBlob dari teks
            blob = TextBlob(text)
            # Mengambil polaritas sentimen
            polarity = blob.sentiment.polarity
            # Menentukan sentimen berdasarkan polaritas
            if polarity > 0:
                return "Positif"
            elif polarity < 0:
                return "Negatif"
            else:
                return "Netral"
        
        # Menambahkan kolom sentimen ke DataFrame
        df['Sentimen'] = df[text_column].apply(analyze_sentiment)
        
        # Menampilkan hasil analisis sentimen
        st.subheader("Hasil Analisis Sentimen")
        st.dataframe(df[['Sentimen', text_column]].head())
        
        # Menampilkan jumlah sentimen positif, negatif, dan netral
        sentiment_counts = df['Sentimen'].value_counts()
        st.subheader("Jumlah Sentimen")
        st.write(sentiment_counts)
        
        # Menampilkan pie chart
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=["#4CAF50", "#F44336", "#FFC107"])
        ax.set_ylabel('')
        st.pyplot(fig)


        
    else:
        st.warning("Silakan upload dataset terlebih dahulu di menu 'Upload Data'.")

