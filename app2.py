import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Theme & Styling ------------------
def set_gradient_background():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #ffffff, #e6f0ff);
            color: #111111 !important;           /* Set dark font color */
        }
        .css-18e3th9, .css-1d391kg {             
            color: #111111 !important;           /* Sidebar + Main content */
        }
        .css-1cpxqw2, .css-1v0mbdj, .css-hxt7ib {
            background-color: rgba(255, 255, 255, 0.7);
            color: #111111 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ------------------ Theme & Styling ------------------
# Call it after set_page_config
st.set_page_config(page_title="Product Recommendation", layout="wide")

st.title("🎯 Product-Based Recommendation Dashboard")

import base64

def set_background_image_local(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: #111111;
        }}
        .css-18e3th9, .css-1d391kg {{
            color: #111111;
        }}
        .css-1cpxqw2, .css-1v0mbdj {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            padding: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local("background.jpg")

import os
st.write("Current directory:", os.getcwd())


# ------------------ Sidebar ------------------
st.sidebar.header("📥 Upload your Ratings CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# ------------------ Cached Functions ------------------

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, header=None)
    df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
    df.drop(['timestamp'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df['rating'] = df['rating'].astype(int)
    return df

@st.cache_data
def filter_top_users_products(df, max_users=1000, max_products=500):
    top_users = df['user_id'].value_counts().head(max_users).index
    top_products = df['prod_id'].value_counts().head(max_products).index
    df_small = df[df['user_id'].isin(top_users) & df['prod_id'].isin(top_products)]
    return df_small

@st.cache_resource
def compute_similarity(df_small):
    item_user_matrix = df_small.pivot_table(index='prod_id', columns='user_id', values='rating').fillna(0)
    similarity_matrix = cosine_similarity(item_user_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=item_user_matrix.index, columns=item_user_matrix.index)
    return item_user_matrix, similarity_df

# ------------------ Main App Logic ------------------
if uploaded_file is not None:
    df = load_data(uploaded_file)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div style="background-color: #2e2e2e;
                        padding: 20px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                <div style='font-size: 24px; color: white;'>👤 Total Users</div>
                <div style='font-size: 40px; color: white; font-weight: bold;'>{df['user_id'].nunique()}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #2e2e2e;
                        padding: 20px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                <div style='font-size: 24px; color: white;'>📦 Total Products</div>
                <div style='font-size: 40px; color: white; font-weight: bold;'>{df['prod_id'].nunique()}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_rating = round(df['rating'].mean(), 2)
        st.markdown(f"""
            <div style="background-color: #2e2e2e;
                        padding: 20px; border-radius: 15px; text-align: center;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                <div style='font-size: 24px; color: white;'>⭐ Average Rating</div>
                <div style='font-size: 40px; color: white; font-weight: bold;'>{avg_rating}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🔍 Recommendations", "🏆 Top Products"])

    # ---------- Tab 1: Visualizations ----------
    # ---------- Tab 1: Visualizations ----------
    with tab1:
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

        st.subheader("📊 Ratings Distribution")
        rating_counts = df['rating'].value_counts().sort_index()

        colA, colB = st.columns([1, 1])

        # Improved Bar Chart
        with colA:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            bars = ax_bar.bar(rating_counts.index.astype(str), rating_counts.values, color="#4c8bf5", edgecolor='black')
            ax_bar.set_title("Rating Count per Score", fontsize=14, fontweight='bold', color='white')
            ax_bar.set_xlabel("Rating", fontsize=12, color='white')
            ax_bar.set_ylabel("Count", fontsize=12, color='white')
            ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

            # Display data labels with white color
            for bar in bars:
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom',
                            fontsize=10, color='white')

            # Apply styling: Dark grey background and shadow
            for spine in ax_bar.spines.values():
                spine.set_color('grey')
                spine.set_linewidth(2)

            # Set dark grey background for the plot
            fig_bar.patch.set_facecolor('#2b2b2b')
            ax_bar.set_facecolor('#2b2b2b')

            # Shadow effect on bars (outline)
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

            # Adding shadow effect to the chart
            st.markdown(
                """
                <style>
                .stChart {
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.4);
                }
                </style>
                """, unsafe_allow_html=True
            )

            st.pyplot(fig_bar)

        # Improved Pie Chart
        with colB:
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            colors = sns.color_palette("pastel")[0:len(rating_counts)]
            wedges, texts, autotexts = ax_pie.pie(
                rating_counts, labels=rating_counts.index.astype(str), autopct='%1.1f%%',
                startangle=90, colors=colors, textprops={'color': 'white', 'fontsize': 10}
            )

            # Display white data labels
            for autotext in autotexts:
                autotext.set_color('white')

            ax_pie.set_title("Rating Distribution (%)", fontsize=14, fontweight='bold', color='white')
            ax_pie.axis('equal')

            # Apply styling: Dark grey background and shadow
            fig_pie.patch.set_facecolor('#2b2b2b')
            ax_pie.set_facecolor('#2b2b2b')

            # Adding shadow effect to the chart
            st.markdown(
                """
                <style>
                .stChart {
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.4);
                }
                </style>
                """, unsafe_allow_html=True
            )
            st.pyplot(fig_pie)

    # ---------- Preprocessing for Recs ----------
    df_small = filter_top_users_products(df)
    item_user_matrix, similarity_df = compute_similarity(df_small)

    # ---------- Tab 2: Recommendations ----------
    with tab2:
        st.subheader("🔍 Product Recommendation Engine")
        selected_product = st.selectbox("Select a Product ID", item_user_matrix.index)

        if st.button("Show Recommendations"):
            similar_products = similarity_df[selected_product].sort_values(ascending=False)[1:11]
            st.write(f"🧠 Top 10 products similar to **{selected_product}**:")

            styled_df = similar_products.reset_index()
            styled_df.columns = ['Product ID', 'Similarity']
            st.dataframe(styled_df.style.format({'Similarity': '{:.2f}'}).background_gradient(cmap='Blues'))

            st.subheader("📈 Similarity Heatmap")
            top_sim_df = similarity_df.loc[similar_products.index, similar_products.index]
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(top_sim_df, annot=True, cmap='viridis', fmt=".2f")
            st.pyplot(fig2)

    # ---------- Tab 3: Top Products ----------
    with tab3:
        st.subheader("🏆 Most Purchased Products Overall")
        most_purchased = df['prod_id'].value_counts().head(10)
        st.dataframe(most_purchased.reset_index().rename(columns={"index": "Product ID", "prod_id": "Purchase Count"}))

else:
    st.info("📥 Upload a CSV file from the sidebar to get started.")


