import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from datetime import datetime
from matplotlib import font_manager as fm

def main(): 

    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')  # Windows: 맑은 고딕
    elif platform.system() == 'Darwin':
        plt.rc('font', family='AppleGothic')    # Mac: 애플 고딕
    else:
        plt.rc('font', family='NanumGothic')    # Linux: 나눔 고딕 (설치 필요할 수 있음)

    if platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    elif platform.system() == 'Darwin':
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    else:
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['axes.unicode_minus'] = False

    st.set_page_config(page_title='고객 이탈 예측 대시보드', layout='wide')
    
    model = joblib.load('./best_model.pkl')

    st.title('🚀 고객 이탈 예측 대시보드')
    st.write('CSV 파일을 업로드하면 고객 이탈 확률을 예측하고 다양한 시각화 결과를 보여줍니다.')


    if 'data' not in st.session_state:
        st.session_state.data = None

    uploaded_file = st.file_uploader("📁 CSV 파일 업로드", type=["csv"])

    if uploaded_file is not None:
        origin_data = pd.read_csv(uploaded_file)
        st.write("업로드한 데이터 : ", origin_data.head())

        if 'id' in origin_data.columns:
            data = origin_data.drop(['id','is_tv_subscriber', 'is_movie_package_subscriber', 'remaining_contract'], axis=1)
        else:
            data = origin_data.drop(['is_tv_subscriber', 'is_movie_package_subscriber', 'remaining_contract'], axis=1)
        probs = model.predict_proba(data)[:,1]
        origin_data['Churn Probability'] = (probs * 100).round(2)

        st.session_state.data = origin_data
    
    if st.session_state.data is not None:

        origin_data = st.session_state.data

        st.subheader('📊 고객 이탈 분석 결과')
        
        if 'id' in origin_data.columns:
            st.dataframe(origin_data[['id','Churn Probability']].style.format({'Churn Probability': "{:.2f}%"}))
        else:
            st.dataframe(origin_data[['Churn Probability']].style.format({'Churn Probability': "{:.2f}%"}))

        col1, col2, col3 = st.columns([1.0, 0.8, 1.2])
        
        with col1:
            st.markdown('### 🎯 고객 이탈 확률 분포')
        
            fig1, ax1 = plt.subplots(figsize=(6,5))
            sns.set_theme(style='whitegrid')
            sns.histplot(origin_data['Churn Probability'],
                    bins=20,
                    kde=True,
                    color='cornflowerblue',
                    edgecolor='black',
                    linewidth=1.2,
                    ax=ax1)
            ax1.set_title('고객 이탈 확률 분포 (히스토그램 + 밀도추정)', fontsize=18, fontweight='bold', pad=20)
            ax1.set_xlabel('이탈 확률', fontsize=14, labelpad=15)
            ax1.set_ylabel('고객 수', fontsize=14, labelpad=15)
            
            for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
                label.set_fontproperties(font_prop)
                label.set_fontsize(12)

            st.pyplot(fig1)

        with col2:
            st.markdown('### 🍰 이탈 vs 유지 고객 비율')

            churned = (origin_data['Churn Probability'] > 50).sum()
            not_churned = (origin_data['Churn Probability'] <= 50).sum()

            fig2, ax2 = plt.subplots(figsize=(4,4)) 
            ax2.pie([churned, not_churned], 
                labels=['이탈 고객', '유지 고객'], 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['tomato', 'lightgreen'], 
                textprops={'fontsize': 12, 'fontproperties':font_prop})
            
            ax2.set_title('고객 이탈 비율', fontsize=18, fontweight='bold', pad=20, fontproperties=font_prop)
            st.pyplot(fig2)

        with col3:
            st.markdown('### 🔥 이탈 확률 Top 10 고객')
            fig3, ax3 = plt.subplots(figsize=(6,5))

            top10 = origin_data[['id', 'Churn Probability']].sort_values(by='Churn Probability', ascending=False).head(10)
            top10['id'] = top10['id'].astype(str)

            

            sns.barplot(x='id', y='Churn Probability', data=top10, palette='Reds_r', ax=ax3)

            ax3.set_title('Top 10 고객 이탈 확률', fontsize=18, fontweight='bold', pad=20, fontproperties=font_prop)
            ax3.set_xlabel('고객 ID', fontsize=14, labelpad=15, fontproperties=font_prop)
            ax3.set_ylabel('이탈 확률', fontsize=14, labelpad=15, fontproperties=font_prop)
            
            

                    
            # Tick label 폰트도 적용
            for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
                label.set_fontproperties(font_prop)
                label.set_fontsize(12)

            plt.xticks(rotation=45)

            ax3.set_ylim(0, 100)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

            for p in ax3.patches:
                value = p.get_height()
                ax3.annotate(f'{value:.2f}%', 
                            (p.get_x() + p.get_width()/2., value),
                            ha='center', va='bottom',
                            fontsize=9, fontproperties=font_prop)

            st.pyplot(fig3)

        st.subheader('이탈 vs 유지 고객 비율')

        

        high_risk = origin_data[origin_data['Churn Probability'] > 70]
        low_risk = origin_data[origin_data['Churn Probability'] < 30]


        feature_columns = origin_data.select_dtypes(include=['number']).columns.difference(['id', 'Churn Probability'])

        high_risk_stats = high_risk[feature_columns].describe().T[['mean', 'std']]
        low_risk_stats = low_risk[feature_columns].describe().T[['mean','std']]

        comparison = high_risk_stats.join(low_risk_stats, lsuffix='_high_risk', rsuffix='_low_risk')
        comparison['mean_diff'] = comparison['mean_high_risk'] - comparison['mean_low_risk']

        comparison = comparison.round(2)

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("⚠️ 이탈 확률 70% 초과 고객 리스트")
            st.write(f'총 고객 수 : {len(origin_data)}명 / 고위험 고객 수 : {len(high_risk)}명')

            st.dataframe(high_risk)
        
        with col2:


            st.subheader('📈 특성별 평균값 비교 (이탈 위험 고객 vs 유지 고객)')

            plot_data = comparison[['mean_high_risk', 'mean_low_risk']].reset_index()

            plot_data = plot_data.dropna()

            plot_data = plot_data.melt(id_vars='index', 
                                value_vars=['mean_high_risk', 'mean_low_risk'], 
                                var_name='Group', 
                                value_name='Mean')
            plot_data['Group'] = plot_data['Group'].map({
            'mean_high_risk': '이탈 위험 고객 (>70%)',
            'mean_low_risk': '유지 고객 (<30%)'
            })

            feature_name_mapping = {
            'bill_avg': '월 평균 요금',
            'download_avg': '월 평균 다운로드',
            'upload_avg': '월 평균 업로드',
            'download_over_limit': '다운로드 한도 초과 비율',
            'is_movie_package_subscriber': '영화 패키지 가입 여부',
            'is_tv_subscriber': 'TV 가입 여부',
            'reamining_contract': '잔여 계약 기간',
            'service_failure_count': '서비스 장애 경험 수',
            'subscription_age': '가입 기간'
            }

            # plot_data의 'index' 컬럼에 적용
            plot_data['index'] = plot_data['index'].map(feature_name_mapping)

            fig4, ax4 = plt.subplots(figsize=(10, 8))
            sns.barplot(data=plot_data, x='Mean', y='index', hue='Group', palette='Set2', ax=ax4)

            ax4.set_xscale('log')

            ax4.set_title('특성별 이탈 위험 고객 vs 유지 고객 평균 비교 (Log Scale)', fontsize=16, fontweight='bold', fontproperties=font_prop)
            ax4.set_xlabel('평균 값', fontsize=12, fontproperties=font_prop)
            ax4.set_ylabel('특성명', fontsize=12, fontproperties=font_prop)
            ax4.legend(title='그룹', prop=font_prop, title_fontproperties=font_prop, loc='best')

            for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
                label.set_fontproperties(font_prop)
                label.set_fontsize(11)

            st.pyplot(fig4)


        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'churn_predictions_{now}.csv'

        st.download_button(
            label="📥 예측 결과 다운로드",
            data = origin_data.to_csv(index=False),
            file_name=filename,
            mime='text/csv',
        )







if __name__ == '__main__':
    main()