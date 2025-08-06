import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ================== تنظیمات اولیه ==================
st.set_page_config(
    page_title="داشبورد آنالیز روغن", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 داشبورد آنالیز روغن ترانسفورماتور")
st.markdown("---")

# ================== کلاس‌های کمکی ==================
class DataProcessor:
    """کلاس پردازش داده‌ها"""
    
    @staticmethod
    def load_data(uploaded_file):
        """بارگذاری فایل CSV"""
        if uploaded_file:
            try:
                return pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file)
        return None
    
    @staticmethod
    def find_post_column(df):
        """پیدا کردن ستون نام پست"""
        candidates = [c for c in df.columns if 'pos111' in c.lower() or 'pos11' in c.lower() or 'pos_111' in c.lower()]
        if candidates:
            return candidates[0]
        return None
    
    @staticmethod
    def prepare_post_data(df, post_column):
        """آماده‌سازی داده‌های پست"""
        if post_column:
            df['post_name'] = df[post_column].astype(str).str.strip()
            df['post_name_normalized'] = df['post_name'].str.lower().str.replace(r'\s+', ' ', regex=True)
        return df
    
    @staticmethod
    def filter_by_post(df, selected_post, search_text=None):
        """فیلتر کردن داده‌ها بر اساس نام پست"""
        if search_text and search_text.strip():
            # جستجوی متنی
            search_term = search_text.strip().lower()
            mask = df['post_name_normalized'].str.contains(search_term, na=False)
            return df[mask].copy(), f"جستجو برای '{search_text}'"
        elif selected_post and selected_post != "همه پست‌ها":
            # انتخاب از لیست
            selected_normalized = selected_post.lower().strip()
            mask = df['post_name_normalized'] == selected_normalized
            return df[mask].copy(), selected_post
        else:
            # همه داده‌ها
            return df.copy(), "همه پست‌ها"
    
    @staticmethod
    def prepare_gas_columns(df):
        """آماده‌سازی ستون‌های گازی"""
        gas_cols = ['hydrogen', 'Methane', 'Ethane', 'Ethylene', 'Acetylene', 
                   'CarbonMonoxide', 'CarbonDioxide']
        
        for col in gas_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
        return df
    
    @staticmethod
    def calculate_risk_level(df):
        """محاسبه سطح خطر"""
        thresholds = {
            'hydrogen': {'normal': 100, 'high': 700},
            'Methane': {'normal': 25, 'high': 50},
            'Ethane': {'normal': 10, 'high': 50},
            'Ethylene': {'normal': 20, 'high': 50},
            'Acetylene': {'normal': 1, 'high': 3},
            'CarbonMonoxide': {'normal': 200, 'high': 500},
        }
        
        def get_risk(row):
            high_count = 0
            total_gases = 0
            
            for gas, limits in thresholds.items():
                if gas in row:
                    total_gases += 1
                    if row[gas] > limits['high']:
                        high_count += 1
            
            if total_gases == 0:
                return 'نامشخص'
            
            risk_ratio = high_count / total_gases
            if risk_ratio >= 0.5:
                return 'پرخطر'
            elif risk_ratio >= 0.2:
                return 'متوسط'
            else:
                return 'سالم'
        
        df['risk_level'] = df.apply(get_risk, axis=1)
        return df

class Visualizer:
    """کلاس نمودارسازی"""
    
    @staticmethod
    def create_risk_pie_chart(risk_counts):
        """نمودار دایره‌ای سطح خطر"""
        colors = {'سالم': '#2ecc71', 'متوسط': '#f39c12', 'پرخطر': '#e74c3c'}
        
        fig = go.Figure(data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker=dict(colors=[colors.get(label, '#bdc3c7') for label in risk_counts.index])
            )
        ])
        
        fig.update_layout(
            title="توزیع سطح خطر",
            height=400,
            showlegend=True
        )
        return fig
    
    @staticmethod
    def create_gas_bar_chart(df):
        """نمودار میله‌ای غلظت گازها"""
        gas_cols = ['hydrogen', 'Methane', 'Ethane', 'Ethylene', 'Acetylene']
        present_cols = [col for col in gas_cols if col in df.columns]
        
        if not present_cols:
            return None
            
        gas_means = df[present_cols].mean()
        
        fig = go.Figure(data=[
            go.Bar(
                x=present_cols,
                y=gas_means.values,
                marker_color='skyblue',
                text=[f'{val:.1f}' for val in gas_means.values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="میانگین غلظت گازها (ppm)",
            xaxis_title="نوع گاز",
            yaxis_title="غلظت (ppm)",
            height=400
        )
        return fig
    
    @staticmethod
    def create_duval_triangle(df):
        """مثلث دووال"""
        # محاسبه درصدها
        total = df['Methane'] + df['Ethylene'] + df['Acetylene']
        mask = total > 0
        
        if not mask.any():
            return None
            
        df_duval = df[mask].copy()
        total_filtered = total[mask]
        
        ch4_pct = (df_duval['Methane'] / total_filtered * 100)
        c2h4_pct = (df_duval['Ethylene'] / total_filtered * 100)
        c2h2_pct = (df_duval['Acetylene'] / total_filtered * 100)
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatterternary(
                a=ch4_pct,
                b=c2h4_pct,
                c=c2h2_pct,
                mode='markers',
                marker=dict(size=10, color='red', opacity=0.7),
                name='نمونه‌ها'
            )
        )
        
        fig.update_layout(
            title="مثلث دووال (CH4-C2H4-C2H2)",
            ternary=dict(
                sum=100,
                aaxis=dict(title='CH4 %'),
                baxis=dict(title='C2H4 %'),
                caxis=dict(title='C2H2 %')
            ),
            height=500
        )
        return fig

# ================== رابط کاربری اصلی ==================

# Sidebar برای تنظیمات
with st.sidebar:
    st.header("⚙️ تنظیمات")
    
    # بارگذاری فایل
    uploaded_file = st.file_uploader(
        "فایل CSV را آپلود کنید", 
        type=["csv"],
        help="فایل باید شامل ستون‌های گازی و pos111 باشد"
    )
    
    # تنظیمات نمایش
    st.subheader("🔍 فیلتر")
    show_raw_data = st.checkbox("نمایش داده‌های خام", value=False)
    max_records = st.slider("حداکثر تعداد رکورد", 10, 1000, 100)

# ================== بارگذاری و پردازش داده ==================
if uploaded_file:
    df = DataProcessor.load_data(uploaded_file)
else:
    csv_url = "https://raw.githubusercontent.com/Saeid13590705/11/main/2_0.csv"
    df = pd.read_csv(csv_url)

if df is None:
    st.error("❌ هیچ داده‌ای بارگذاری نشد")
    st.stop()

# پیدا کردن ستون پست
post_column = DataProcessor.find_post_column(df)

if post_column is None:
    st.error("❌ ستون pos111 در فایل یافت نشد! لطفاً فایل صحیح را آپلود کنید.")
    st.info("ستون‌های موجود در فایل:")
    st.write(list(df.columns))
    st.stop()

# آماده‌سازی داده‌ها (بدون نمایش پیام)
df = DataProcessor.prepare_post_data(df, post_column)
df = DataProcessor.prepare_gas_columns(df)
df = DataProcessor.calculate_risk_level(df)

# ================== انتخاب پست ==================
st.markdown("---")
st.subheader("🏭 انتخاب پست مورد نظر")

# نمایش پست‌های موجود
unique_posts = sorted(df['post_name'].dropna().unique())

# گزینه‌های انتخاب پست
col1, col2 = st.columns([1, 1])

with col1:
    selected_post = st.selectbox(
        "📍 انتخاب از فهرست پست‌ها:",
        ["همه پست‌ها"] + list(unique_posts),
        help="پست مورد نظر را انتخاب کنید"
    )

with col2:
    search_text = st.text_input(
        "🔍 جستجوی نام پست:",
        placeholder="نام پست را تایپ کنید...",
        help="برای جستجوی جزئی در نام پست‌ها"
    )
    
    # نمایش پیشنهادات بر اساس جستجو
    if search_text and search_text.strip():
        search_term = search_text.strip().lower()
        matching_posts = [post for post in unique_posts if search_term in post.lower()]
        
        if matching_posts:
            st.write("🔽 **پست‌های مرتبط:**")
            # ساخت دکمه‌های کلیکی برای هر پیشنهاد
            for i, post in enumerate(matching_posts[:10]):  # محدود به 10 مورد اول
                if st.button(f"📌 {post}", key=f"post_btn_{i}"):
                    # با کلیک روی دکمه، متن جستجو را پاک کرده و پست را انتخاب می‌کنیم
                    st.session_state.selected_post_from_search = post
                    st.rerun()
            
            if len(matching_posts) > 10:
                st.info(f"+ {len(matching_posts) - 10} مورد دیگر... جستجو را دقیق‌تر کنید")
        else:
            st.warning("❌ هیچ پست مرتبطی یافت نشد")

# بررسی اگر پستی از طریق دکمه انتخاب شده
if 'selected_post_from_search' in st.session_state:
    selected_post = st.session_state.selected_post_from_search
    search_text = ""  # پاک کردن متن جستجو
    del st.session_state.selected_post_from_search  # پاک کردن از session state

# اعمال فیلتر
filtered_df, filter_description = DataProcessor.filter_by_post(df, selected_post, search_text)

if filtered_df.empty:
    st.warning("❌ هیچ داده‌ای با این فیلتر یافت نشد!")
    st.stop()

# نمایش اطلاعات فیلتر شده (بدون پیام موفقیت)
# محدود کردن تعداد رکوردها برای نمایش
df_display = filtered_df.head(max_records)

# ================== نمایش اطلاعات کلی ==================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("نمونه‌های فیلتر شده", len(filtered_df))

with col2:
    safe_count = len(filtered_df[filtered_df['risk_level'] == 'سالم'])
    st.metric("نمونه‌های سالم", safe_count)

with col3:
    medium_count = len(filtered_df[filtered_df['risk_level'] == 'متوسط'])
    st.metric("نمونه‌های متوسط", medium_count, delta_color="off")

with col4:
    danger_count = len(filtered_df[filtered_df['risk_level'] == 'پرخطر'])
    st.metric("نمونه‌های پرخطر", danger_count, delta_color="inverse")

st.markdown("---")

# ================== نمودارهای اصلی ==================
tab1, tab2, tab3 = st.tabs(["📊 نمودارها", "🔺 مثلث دووال", "📋 جزئیات داده"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # نمودار دایره‌ای سطح خطر
        risk_counts = filtered_df['risk_level'].value_counts()
        if not risk_counts.empty:
            pie_fig = Visualizer.create_risk_pie_chart(risk_counts)
            st.plotly_chart(pie_fig, use_container_width=True)
    
    with col2:
        # نمودار میله‌ای گازها
        bar_fig = Visualizer.create_gas_bar_chart(filtered_df)
        if bar_fig:
            st.plotly_chart(bar_fig, use_container_width=True)

with tab2:
    # مثلث دووال
    duval_fig = Visualizer.create_duval_triangle(df_display)
    if duval_fig:
        st.plotly_chart(duval_fig, use_container_width=True)
    else:
        st.info("برای رسم مثلث دووال، داده‌های گازی کافی وجود ندارد")

with tab3:
    # نمایش جدول داده‌ها
    if show_raw_data:
        st.subheader("داده‌های خام")
        st.dataframe(df_display, use_container_width=True)
    
    # آمار توصیفی
    st.subheader("آمار توصیفی گازها")
    gas_cols = ['hydrogen', 'Methane', 'Ethane', 'Ethylene', 'Acetylene']
    present_cols = [col for col in gas_cols if col in df.columns]
    
    if present_cols:
        st.dataframe(filtered_df[present_cols].describe(), use_container_width=True)

# ================== نمایش جزئیات پست انتخابی ==================
if len(filtered_df) > 0 and filter_description != "همه پست‌ها":
    st.markdown("---")
    st.subheader(f"📋 جزئیات پست: {filter_description}")
    
    # اطلاعات خلاصه پست
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**تعداد آزمایش‌های ثبت شده:** {len(filtered_df)}")
        latest_risk = filtered_df['risk_level'].iloc[-1] if len(filtered_df) > 0 else 'نامشخص'
        st.info(f"**آخرین وضعیت خطر:** {latest_risk}")
    
    with col2:
        if 'post_name' in filtered_df.columns:
            st.info(f"**نام دقیق پست:** {filtered_df['post_name'].iloc[0]}")
    
    # نمودار روند زمانی (اگر داده زمانی موجود باشد)
    date_cols = [col for col in filtered_df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols and len(filtered_df) > 1:
        date_col = date_cols[0]
        try:
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
            filtered_df_sorted = filtered_df.sort_values(date_col).dropna(subset=[date_col])
            
            if len(filtered_df_sorted) > 1:
                fig_trend = go.Figure()
                gas_cols = ['hydrogen', 'Methane', 'Ethylene', 'Acetylene']
                colors = ['blue', 'red', 'green', 'orange']
                
                for i, gas in enumerate(gas_cols):
                    if gas in filtered_df_sorted.columns:
                        fig_trend.add_trace(go.Scatter(
                            x=filtered_df_sorted[date_col],
                            y=filtered_df_sorted[gas],
                            mode='lines+markers',
                            name=gas,
                            line=dict(color=colors[i % len(colors)])
                        ))
                
                fig_trend.update_layout(
                    title="روند زمانی غلظت گازها",
                    xaxis_title="تاریخ",
                    yaxis_title="غلظت (ppm)",
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        except:
            pass

# ================== دانلود نتایج ==================
st.markdown("---")
st.subheader("💾 دانلود نتایج")

col1, col2 = st.columns(2)
with col1:
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ دانلود داده‌های فیلتر شده (CSV)",
        data=csv_data,
        file_name=f"oil_analysis_{filter_description.replace(' ', '_')}.csv",
        mime="text/csv"
    )

with col2:
    summary_data = filtered_df['risk_level'].value_counts().to_csv().encode('utf-8')
    st.download_button(
        "⬇️ دانلود خلاصه آمار (CSV)",
        data=summary_data,
        file_name=f"risk_summary_{filter_description.replace(' ', '_')}.csv",
        mime="text/csv"
    )

# پیام پایانی
st.success("✅ گزارش با موفقیت ساخته شد!")
