import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import requests
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Global Findex 2025 データ分析",
    page_icon="📊",
    layout="wide"
)

COUNTRY_MAP = {
    'Guatemala': 'グアテマラ', 'El Salvador': 'エルサルバドル', 'Honduras': 'ホンジュラス',
    'Nicaragua': 'ニカラグア', 'Costa Rica': 'コスタリカ', 'Panama': 'パナマ',
    'Belize': 'ベリーズ', 'Mexico': 'メキシコ', 'Dominican Republic': 'ドミニカ共和国',
    'Japan': '日本', 'China': '中国', 'Vietnam': 'ベトナム', 'India': 'インド',
    'Kenya': 'ケニア', 'United Kingdom': '英国', 'United States': '米国',
    'Colombia': 'コロンビア', 'Ecuador': 'エクアドル', 'Bolivia': 'ボリビア',
    'Peru': 'ペルー', 'Paraguay': 'パラグアイ', 'Argentina': 'アルゼンチン', 'Brazil': 'ブラジル', 'Chile': 'チリ',
    'Uruguay': 'ウルグアイ', 'Venezuela, RB': 'ベネズエラ', 'Jamaica': 'ジャマイカ',
    'Bahamas, The': 'バハマ', 'Trinidad and Tobago': 'トリニダードドバコ', 'Suriname': 'スリナム',
    'Thailand': 'タイ', 'Indonesia': 'インドネシア', 'Philippines': 'フィリピン', 
    'Malaysia': 'マレーシア', 'Lao PDR': 'ラオス', 'Cambodia': 'カンボジア',
    'Pakistan': 'パキスタン', 'Bangladesh': 'バングラデシュ', 'Sri Lanka': 'スリランカ',
    'Tanzania': 'タンザニア', 'Ethiopia': 'エチオピア', 'Zambia': 'ザンビア',
    'Nigeria': 'ナイジェリア', 'Uganda': 'ウガンダ', 'Senegal': 'セネガル',
    'Ghana': 'ガーナ', 'Mozambique': 'モザンビーク', 'Rwanda': 'ルワンダ',
    "Côte d'Ivoire": 'コートジボワール', 'Malawi': 'マラウイ',
    'Egypt, Arab Rep.': 'エジプト', 'Türkiye': 'トルコ', 'Turkey': 'トルコ',
    'Saudi Arabia': 'サウディアラビア', 'Iran, Islamic Rep.': 'イラン',
    'Jordan': 'ヨルダン', 'Armenia': 'アルメニア',
    'Algeria': 'アルジェリア', 'Angola': 'アンゴラ', 'Benin': 'ベナン', 'Bhutan': 'ブータン',
    'Cabo Verde': 'カーボベルデ', 'Cameroon': 'カメルーン', 'Comoros': 'コモロ', 'Congo, Rep.': 'コンゴ共和国',
    'Djibouti': 'ジブチ', 'Guinea': 'ギニア', 'Haiti': 'ハイチ', 'Kyrgyz Republic': 'キルギス',
    'Lesotho': 'レソト', 'Madagascar': 'マダガスカル', 'Mauritania': 'モーリタニア', 'Mongolia': 'モンゴル',
    'Morocco': 'モロッコ', 'Myanmar': 'ミャンマー', 'Namibia': 'ナミビア', 'Nepal': 'ネパール',
    'Papua New Guinea': 'パプアニューギニア', 'Sao Tome and Principe': 'サントメ・プリンシペ',
    'Sierra Leone': 'シエラレオネ', 'Solomon Islands': 'ソロモン諸島', 'Sudan': 'スーダン',
    'Eswatini': 'エスワティニ', 'Syrian Arab Republic': 'シリア', 'Tajikistan': 'タジキスタン',
    'Timor-Leste': '東ティモール', 'Togo': 'トーゴ', 'Tunisia': 'チュニジア', 'Ukraine': 'ウクライナ',
    'Uzbekistan': 'ウズベキスタン', 'Zimbabwe': 'ジンバブエ',
    'Albania': 'アルバニア', 'Azerbaijan': 'アゼルバイジャン', 'Belarus': 'ベラルーシ', 'Botswana': 'ボツワナ',
    'Bulgaria': 'ブルガリア', 'Croatia': 'クロアチア', 'Cuba': 'キューバ', 'Fiji': 'フィジー',
    'Gabon': 'ガボン', 'Grenada': 'グレナダ', 'Greece': 'ギリシャ', 'Guam': 'グアム',
    'Hungary': 'ハンガリー', 'Iraq': 'イラク', 'Kazakhstan': 'カザフスタン', 'Lebanon': 'レバノン',
    'Libya': 'リビア', 'Maldives': 'モルディブ', 'Montenegro': 'モンテネグロ', 'Nauru': 'ナウル',
    'Poland': 'ポーランド', 'Romania': 'ルーマニア', 'Russian Federation': 'ロシア', 'Serbia': 'セルビア',
    'South Africa': '南アフリカ', 'Turkmenistan': 'トルクメニスタン', 'West Bank and Gaza': '西岸・ガザ',
    'St. Lucia': 'セントルシア', 'St. Vincent and the Grenadines': 'セントビンセント・グレナディーン',
    'Dominica': 'ドミニカ国', 'Palau': 'パラオ', 'Marshall Islands': 'マーシャル諸島',
    'Micronesia, Fed. Sts.': 'ミクロネシア', 'Bosnia and Herzegovina': 'ボスニア・ヘルツェゴビナ',
    'North Macedonia': '北マケドニア',
    'Latin America & Caribbean (excluding high income)': '中南米・カリブ（高所得国除く）',
    'Sub-Saharan Africa (excluding high income)': 'サブサハラ（高所得国除く）',
    'Middle East & North Africa (excluding high income)': '中東・北アフリカ（高所得国除く）',
    'East Asia & Pacific (excluding high income)': '東アジア・太平洋（高所得国除く）',
    'world': '世界全体',
    'Upper middle income': '高位中所得国',
    'Lower middle income': '低位中所得国'
}

INDICATOR_GROUPS = {
    '口座関連': {
        '口座保有率': 'Account (%, age 15+)',
        '銀行口座保有率': 'Bank or similar financial institution account (%, age 15+)',
        'モバイルマネー口座保有率': 'Mobile money account (%, age 15+)',
    },
    '貯蓄': {
        '貯蓄率': 'Saved any money (%, age 15+)',
        '金融機関での貯蓄': 'Saved at a bank or similar financial institution (%, age 15+)',
        '老後のための貯蓄': 'Saved for old age (%, age 15+)',
    },
    '借入': {
        '借入経験率': 'Borrowed any money (%, age 15+)',
        '金融機関からの借入': 'Borrowed from a formal bank or similar financial institution (%, age 15+)',
        '家族・友人からの借入': 'Borrowed from family or friends (%, age 15+)',
        '医療目的の借入': 'Borrowed for health or medical purposes (%, age 15+)',
    },
    '緊急時資金': {
        '緊急資金調達（困難なし）': 'Coming up with emergency funds in 30 days: possible and not difficult at all (%, age 15+)',
        '緊急資金調達（やや困難）': 'Coming up with emergency funds in 30 days: possible and somewhat difficult (%, age 15+)',
        '緊急資金調達（非常に困難）': 'Coming up with emergency funds in 30 days: possible and very difficult (%, age 15+)',
    },
    'デジタル決済': {
        'デジタル決済利用率': 'Made a digital payment (%, age 15+)',
        'デジタル決済受領率': 'Received digital payments (%, age 15+)',
        'デジタル店舗決済': 'Made a digital merchant payment (%, age 15+)',
    },
    '携帯・インターネット': {
        '携帯電話保有率': 'Own a mobile phone (%, age 15+)',
        'スマートフォン保有率': 'Main mobile phone is a smartphone (%, age 15+)',
        'インターネット利用率': 'Used the internet in the past three months (%, age 15+)',
        '毎日のインターネット利用': 'Daily internet use (%, age 15+)',
    },
    'その他': {
        'クレジットカード保有': 'Owns a credit card (%, age 15+)',
        '送金経験': 'Sent or received domestic remittances (%, age 15+)',
        '国際送金受領': 'Received international remittances  (%, age 15+)',
    }
}

CENTRAL_AMERICA = ['グアテマラ', 'エルサルバドル', 'ホンジュラス', 'ニカラグア', 'コスタリカ', 'パナマ', 'ベリーズ', 'メキシコ', 'ドミニカ共和国']
SOUTH_AMERICA = ['コロンビア', 'エクアドル', 'ボリビア', 'ペルー', 'チリ', 'アルゼンチン', 'ブラジル', 'パラグアイ']
SOUTHEAST_ASIA = ['フィリピン', 'タイ', 'ベトナム', 'インドネシア', 'マレーシア', 'ラオス', 'カンボジア']
SOUTH_ASIA = ['パキスタン', 'インド', 'バングラデシュ', 'スリランカ']
SUB_SAHARAN = ['ケニア', 'タンザニア', 'エチオピア', 'ザンビア', 'ナイジェリア', 'ウガンダ', 'セネガル', 'ガーナ', 'モザンビーク', 'ルワンダ', 'コートジボワール', 'マラウイ']
MIDDLE_EAST = ['エジプト', 'トルコ', 'サウディアラビア', 'イラン', 'ヨルダン', 'アルメニア']

REGIONS = ['中南米・カリブ（高所得国除く）', 'サブサハラ（高所得国除く）', '中東・北アフリカ（高所得国除く）', '東アジア・太平洋（高所得国除く）', '世界全体', '高位中所得国', '低位中所得国']
COMPARISON = ['日本', '中国', 'ベトナム', 'インド', 'ケニア', '英国', '米国'] + SOUTH_AMERICA + SOUTHEAST_ASIA + REGIONS

REGION_GROUPS = {
    '中米9カ国': CENTRAL_AMERICA,
    '南米8カ国': SOUTH_AMERICA,
    '東南アジア7カ国': SOUTHEAST_ASIA,
    '南アジア4カ国': SOUTH_ASIA,
    'サブサハラ12カ国': SUB_SAHARAN,
    '中近東6カ国': MIDDLE_EAST
}

ML_COUNTRY_GROUPS = {
    '中南米諸国23カ国': [
        'Guatemala', 'Honduras', 'El Salvador', 'Belize', 'Mexico', 'Costa Rica', 'Nicaragua',
        'Dominican Republic', 'Panama', 'Colombia', 'Ecuador', 'Bolivia', 'Peru', 'Chile',
        'Argentina', 'Brazil', 'Paraguay', 'Uruguay', 'Venezuela, RB', 'Jamaica',
        'Bahamas, The', 'Trinidad and Tobago', 'Suriname'
    ],
    '中所得国110カ国': [
        'Algeria', 'Angola', 'Bangladesh', 'Benin', 'Bhutan', 'Bolivia', 'Cabo Verde', 'Cameroon',
        'Comoros', 'Congo, Rep.', "Côte d'Ivoire", 'Djibouti', 'Egypt, Arab Rep.', 'El Salvador',
        'Ghana', 'Guatemala', 'Guinea', 'Haiti', 'Honduras', 'India', 'Indonesia', 'Iran, Islamic Rep.',
        'Kenya', 'Kyrgyz Republic', 'Lao PDR', 'Lesotho', 'Madagascar', 'Malawi', 'Mauritania',
        'Mongolia', 'Morocco', 'Myanmar', 'Namibia', 'Nepal', 'Nicaragua', 'Nigeria', 'Pakistan',
        'Papua New Guinea', 'Philippines', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Sierra Leone',
        'Solomon Islands', 'Sri Lanka', 'Sudan', 'Eswatini', 'Syrian Arab Republic', 'Tajikistan',
        'Tanzania', 'Timor-Leste', 'Togo', 'Tunisia', 'Uganda', 'Ukraine', 'Uzbekistan', 'Vietnam',
        'Zambia', 'Zimbabwe', 'Albania', 'Argentina', 'Azerbaijan', 'Belarus', 'Botswana', 'Brazil', 
        'Bulgaria', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Cuba', 'Dominican Republic', 'Ecuador', 
        'Fiji', 'Gabon', 'Grenada', 'Greece', 'Guam', 'Hungary', 'Iraq', 'Jamaica', 'Kazakhstan', 'Lebanon', 
        'Libya', 'Malaysia', 'Maldives', 'Mexico', 'Montenegro', 'Nauru', 'Paraguay', 'Peru', 'Poland', 
        'Romania', 'Russian Federation', 'Serbia', 'South Africa', 'Suriname', 'Thailand', 'Turkmenistan', 
        'Turkey', 'Venezuela, RB', 'West Bank and Gaza', 'Jordan', 'St. Lucia', 'St. Vincent and the Grenadines',
        'Dominica', 'Palau', 'Marshall Islands', 'Micronesia, Fed. Sts.', 'Bosnia and Herzegovina',
        'North Macedonia', 'Türkiye', 'Bahamas, The', 'Trinidad and Tobago'
    ]
}

COUNTRY_CODE_MAP = {
    'Guatemala': 'GT', 'El Salvador': 'SV', 'Honduras': 'HN', 'Nicaragua': 'NI',
    'Costa Rica': 'CR', 'Panama': 'PA', 'Belize': 'BZ', 'Mexico': 'MX',
    'Dominican Republic': 'DO', 'Japan': 'JP', 'China': 'CN', 'Vietnam': 'VN',
    'India': 'IN', 'Kenya': 'KE', 'United Kingdom': 'GB', 'United States': 'US',
    'Colombia': 'CO', 'Ecuador': 'EC', 'Bolivia': 'BO', 'Peru': 'PE',
    'Paraguay': 'PY', 'Argentina': 'AR', 'Brazil': 'BR', 'Uruguay': 'UY',
    'Thailand': 'TH', 'Indonesia': 'ID', 'Philippines': 'PH', 'Chile': 'CL',
    'Venezuela, RB': 'VE', 'Jamaica': 'JM', 'Bahamas, The': 'BS', 'Trinidad and Tobago': 'TT',
    'Suriname': 'SR', 'Algeria': 'DZ', 'Angola': 'AO', 'Bangladesh': 'BD', 'Benin': 'BJ',
    'Bhutan': 'BT', 'Cabo Verde': 'CV', 'Cameroon': 'CM', 'Comoros': 'KM',
    'Congo, Rep.': 'CG', "Côte d'Ivoire": 'CI', 'Djibouti': 'DJ', 'Egypt, Arab Rep.': 'EG',
    'Ghana': 'GH', 'Guinea': 'GN', 'Haiti': 'HT', 'Iran, Islamic Rep.': 'IR',
    'Kyrgyz Republic': 'KG', 'Lao PDR': 'LA', 'Lesotho': 'LS', 'Madagascar': 'MG',
    'Malawi': 'MW', 'Mauritania': 'MR', 'Mongolia': 'MN', 'Morocco': 'MA',
    'Myanmar': 'MM', 'Namibia': 'NA', 'Nepal': 'NP', 'Nigeria': 'NG',
    'Pakistan': 'PK', 'Papua New Guinea': 'PG', 'Rwanda': 'RW', 'Sao Tome and Principe': 'ST',
    'Senegal': 'SN', 'Sierra Leone': 'SL', 'Solomon Islands': 'SB', 'Sri Lanka': 'LK',
    'Sudan': 'SD', 'Eswatini': 'SZ', 'Syrian Arab Republic': 'SY', 'Tajikistan': 'TJ',
    'Tanzania': 'TZ', 'Timor-Leste': 'TL', 'Togo': 'TG', 'Tunisia': 'TN',
    'Uganda': 'UG', 'Ukraine': 'UA', 'Uzbekistan': 'UZ', 'Zambia': 'ZM', 'Zimbabwe': 'ZW',
    'Albania': 'AL', 'Azerbaijan': 'AZ', 'Belarus': 'BY', 'Botswana': 'BW',
    'Bulgaria': 'BG', 'Croatia': 'HR', 'Cuba': 'CU', 'Fiji': 'FJ',
    'Gabon': 'GA', 'Grenada': 'GD', 'Greece': 'GR', 'Guam': 'GU', 'Hungary': 'HU',
    'Iraq': 'IQ', 'Kazakhstan': 'KZ', 'Lebanon': 'LB', 'Libya': 'LY',
    'Malaysia': 'MY', 'Maldives': 'MV', 'Montenegro': 'ME', 'Nauru': 'NR',
    'Poland': 'PL', 'Romania': 'RO', 'Russian Federation': 'RU', 'Serbia': 'RS',
    'South Africa': 'ZA', 'Turkmenistan': 'TM', 'Turkey': 'TR',
    'West Bank and Gaza': 'PS', 'Jordan': 'JO', 'St. Lucia': 'LC',
    'St. Vincent and the Grenadines': 'VC', 'Dominica': 'DM', 'Palau': 'PW',
    'Marshall Islands': 'MH', 'Micronesia, Fed. Sts.': 'FM', 'Bosnia and Herzegovina': 'BA',
    'North Macedonia': 'MK', 'Türkiye': 'TR'
}

WB_INDICATORS = {
    "一人当たりGDP（実質）": "NY.GDP.PCAP.KD",
    "貧困率": "SI.POV.NAHC",
    "純移民数": "SM.POP.NETM",
    "経済成長率": "NY.GDP.MKTP.KD.ZG",
    "総資本形成（対GDP比）": "NE.GDI.TOTL.ZS",
    "金融深化度（民間融資の対GDP比）": "FD.AST.PRVT.GD.ZS"
}

@st.cache_data
def load_findex_data():
    try:
        df = pd.read_excel('attached_assets/Findex2025_1760415783997.xlsx', sheet_name='Data')
        df['Economy_JP'] = df['Economy'].map(COUNTRY_MAP)
        df = df.dropna(subset=['Economy_JP'])
        return df
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return pd.DataFrame()

@st.cache_data
def get_all_indicators(df):
    """データフレームからすべての指標列を取得"""
    indicator_cols = [col for col in df.columns if '(%, age 15+)' in col]
    return sorted(indicator_cols)

@st.cache_data
def get_world_bank_data(indicator_code, year=2024, country_list=None):
    try:
        if country_list:
            country_codes = [COUNTRY_CODE_MAP.get(c) for c in country_list if c in COUNTRY_CODE_MAP]
        else:
            country_codes = list(COUNTRY_CODE_MAP.values())
        
        if not country_codes:
            return pd.DataFrame()
        
        url = f"https://api.worldbank.org/v2/country/{';'.join(country_codes)}/indicator/{indicator_code}"
        params = {'date': str(year), 'format': 'json', 'per_page': 1000}
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
                return pd.DataFrame(data[1])
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"世界銀行API接続エラー: {e}")
        return pd.DataFrame()

def get_data_for_indicator(df, indicator_eng, demographic_group='all', year=2024):
    filtered = df[(df['Demographic group'] == demographic_group) & (df['Year'] == year)]
    result = filtered[['Economy_JP', indicator_eng]].dropna()
    result.columns = ['国', '値']
    result['値'] = result['値'] * 100
    return result

def get_gender_data(df, indicator_eng, year=2024):
    data_all = df[(df['Demographic group'] == 'gender') & (df['Demographic sub-group'] == 'men') & (df['Year'] == year)]
    male_data = data_all[['Economy_JP', indicator_eng]].dropna()
    male_data.columns = ['国', '男性']
    male_data['男性'] = male_data['男性'] * 100
    
    data_all = df[(df['Demographic group'] == 'gender') & (df['Demographic sub-group'] == 'women') & (df['Year'] == year)]
    female_data = data_all[['Economy_JP', indicator_eng]].dropna()
    female_data.columns = ['国', '女性']
    female_data['女性'] = female_data['女性'] * 100
    
    merged = pd.merge(male_data, female_data, on='国', how='inner')
    return merged

def get_income_data(df, indicator_eng, year=2024):
    rich_data = df[(df['Demographic group'] == 'income') & (df['Demographic sub-group'] == 'richest 60%') & (df['Year'] == year)]
    rich_df = rich_data[['Economy_JP', indicator_eng]].dropna()
    rich_df.columns = ['国', '富裕層60%']
    rich_df['富裕層60%'] = rich_df['富裕層60%'] * 100
    
    poor_data = df[(df['Demographic group'] == 'income') & (df['Demographic sub-group'] == 'poorest 40%') & (df['Year'] == year)]
    poor_df = poor_data[['Economy_JP', indicator_eng]].dropna()
    poor_df.columns = ['国', '貧困層40%']
    poor_df['貧困層40%'] = poor_df['貧困層40%'] * 100
    
    merged = pd.merge(rich_df, poor_df, on='国', how='inner')
    return merged

def indicator_analysis(df):
    st.header("📈 指標別グラフ可視化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏛️ 対象国・地域選択")
        selected_ca = st.multiselect("中米9カ国を選択", CENTRAL_AMERICA, default=CENTRAL_AMERICA[:3])
        selected_comp = st.multiselect("比較対象国・地域を選択", COMPARISON, default=['日本', '米国'])
    
    with col2:
        st.subheader("📊 指標・カテゴリ選択")
        group_options = list(INDICATOR_GROUPS.keys()) + ['すべての指標（英語名）']
        selected_group = st.selectbox("指標グループを選択", group_options)
        
        if selected_group == 'すべての指標（英語名）':
            all_indicators = get_all_indicators(df)
            selected_indicator_jp = st.selectbox("具体的指標を選択", all_indicators)
            indicator_eng = selected_indicator_jp
        else:
            selected_indicator_jp = st.selectbox("具体的指標を選択", list(INDICATOR_GROUPS[selected_group].keys()))
            indicator_eng = INDICATOR_GROUPS[selected_group][selected_indicator_jp]
        
        chart_type = st.selectbox("グラフタイプを選択", ["棒グラフ", "折れ線グラフ（時系列）"])
        
        if chart_type == "棒グラフ":
            category = st.selectbox("分析カテゴリを選択", ["全体", "男女別", "所得水準別"], key="category_bar")
        else:
            category = st.selectbox("分析カテゴリを選択", ["全体"], key="category_line")
    
    selected_countries = selected_ca + selected_comp
    
    if not selected_countries:
        st.warning("少なくとも1つの国を選択してください")
        return
    
    if chart_type == "棒グラフ":
        available_years = sorted([int(y) for y in df['Year'].unique() if pd.notna(y)])
        if available_years:
            year = st.select_slider("表示年を選択", options=available_years, value=available_years[-1])
        else:
            st.error("データに利用可能な年がありません")
            return
        
        if category == "全体":
            data = get_data_for_indicator(df, indicator_eng, 'all', year)
            data = data[data['国'].isin(selected_countries)]
            
            fig = px.bar(data, x='国', y='値', title=f"{selected_indicator_jp} ({year}年)",
                        labels={'値': '割合 (%)'}, text='値')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            
        elif category == "男女別":
            data = get_gender_data(df, indicator_eng, year)
            data = data[data['国'].isin(selected_countries)]
            
            if data.empty:
                st.warning(f"選択した国・年度（{year}年）の男女別データが見つかりません。別の年を選択してください。")
                return
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='男性', x=data['国'], y=data['男性'], marker_color='blue', text=data['男性']))
            fig.add_trace(go.Bar(name='女性', x=data['国'], y=data['女性'], marker_color='red', text=data['女性']))
            fig.update_layout(title=f"{selected_indicator_jp} - 男女別比較 ({year}年)", 
                            xaxis_title="国名", yaxis_title="割合 (%)", barmode='group')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            
        else:
            income_data = get_income_data(df, indicator_eng, year)
            if income_data.empty:
                st.warning(f"選択した年度（{year}年）の所得水準別データが見つかりません。別の年を選択してください。")
                return
            
            data = income_data[income_data['国'].isin(selected_countries)]
            if data.empty:
                st.warning(f"選択した国の所得水準別データが見つかりません。")
                return
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='富裕層60%', x=data['国'], y=data['富裕層60%'], marker_color='green', text=data['富裕層60%']))
            fig.add_trace(go.Bar(name='貧困層40%', x=data['国'], y=data['貧困層40%'], marker_color='orange', text=data['貧困層40%']))
            fig.update_layout(title=f"{selected_indicator_jp} - 所得水準別比較 ({year}年)", 
                            xaxis_title="国名", yaxis_title="割合 (%)", barmode='group')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        years = sorted(df['Year'].unique())
        
        if category == "全体":
            fig = go.Figure()
            for country in selected_countries:
                y_values = []
                for year in years:
                    data = get_data_for_indicator(df, indicator_eng, 'all', year)
                    data = data[data['国'] == country]
                    if not data.empty:
                        y_values.append(data['値'].values[0])
                    else:
                        y_values.append(None)
                
                fig.add_trace(go.Scatter(x=years, y=y_values, mode='lines+markers', name=country,
                                        line=dict(width=2), marker=dict(size=6), connectgaps=True))
            
            fig.update_layout(title=f"{selected_indicator_jp} の時系列推移", xaxis_title="年", 
                            yaxis_title="割合 (%)", hovermode='x unified')
            
        elif category == "男女別":
            fig = go.Figure()
            for country in selected_countries:
                male_values = []
                female_values = []
                for year in years:
                    data = get_gender_data(df, indicator_eng, year)
                    data = data[data['国'] == country]
                    if not data.empty:
                        male_values.append(data['男性'].values[0])
                        female_values.append(data['女性'].values[0])
                    else:
                        male_values.append(None)
                        female_values.append(None)
                
                fig.add_trace(go.Scatter(x=years, y=male_values, mode='lines+markers',
                                        name=f'{country}（男性）', line=dict(width=2, dash='solid'),
                                        marker=dict(size=6, color='blue')))
                fig.add_trace(go.Scatter(x=years, y=female_values, mode='lines+markers',
                                        name=f'{country}（女性）', line=dict(width=2, dash='dash'),
                                        marker=dict(size=6, color='red')))
            
            fig.update_layout(title=f"{selected_indicator_jp} の時系列推移（男女別）",
                            xaxis_title="年", yaxis_title="割合 (%)", hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)

def country_profile(df):
    st.header("🌍 国別プロファイル分析")
    
    selected_country = st.selectbox("分析対象国を選択", CENTRAL_AMERICA + ['日本', '中国', 'ベトナム', 'インド', 'ケニア', '英国', '米国'])
    year = st.slider("表示年を選択", 2011, 2024, 2024, step=1)
    
    st.subheader(f"📊 {selected_country} の金融包摂指標プロファイル ({year}年)")
    
    indicators_for_radar = {
        '口座保有率': 'Account (%, age 15+)',
        'デジタル決済': 'Made a digital payment (%, age 15+)',
        '携帯電話保有': 'Own a mobile phone (%, age 15+)',
        'インターネット利用': 'Used the internet in the past three months (%, age 15+)',
        '貯蓄率': 'Saved any money (%, age 15+)',
        '借入経験': 'Borrowed any money (%, age 15+)'
    }
    
    values = []
    labels = []
    for label, indicator in indicators_for_radar.items():
        data = get_data_for_indicator(df, indicator, 'all', year)
        country_data = data[data['国'] == selected_country]
        if not country_data.empty:
            values.append(country_data['値'].values[0])
            labels.append(label)
    
    if values:
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=labels, fill='toself', name=selected_country))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                         title=f"{selected_country} の金融包摂プロファイル")
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💳 口座・決済")
        for label, indicator in [('口座保有率', 'Account (%, age 15+)'), 
                                ('デジタル決済', 'Made a digital payment (%, age 15+)')]:
            data = get_data_for_indicator(df, indicator, 'all', year)
            country_data = data[data['国'] == selected_country]
            if not country_data.empty:
                st.metric(label, f"{country_data['値'].values[0]:.1f}%")
    
    with col2:
        st.subheader("📱 デジタル利用")
        for label, indicator in [('携帯電話保有', 'Own a mobile phone (%, age 15+)'),
                                ('インターネット利用', 'Used the internet in the past three months (%, age 15+)')]:
            data = get_data_for_indicator(df, indicator, 'all', year)
            country_data = data[data['国'] == selected_country]
            if not country_data.empty:
                st.metric(label, f"{country_data['値'].values[0]:.1f}%")
    
    with col3:
        st.subheader("💰 貯蓄・借入")
        for label, indicator in [('貯蓄率', 'Saved any money (%, age 15+)'),
                                ('借入経験', 'Borrowed any money (%, age 15+)')]:
            data = get_data_for_indicator(df, indicator, 'all', year)
            country_data = data[data['国'] == selected_country]
            if not country_data.empty:
                st.metric(label, f"{country_data['値'].values[0]:.1f}%")

def correspondence_analysis(df):
    st.header("🔍 主成分分析（PCA：2024年データ）")
    
    col1, col2 = st.columns(2)
    
    all_indicators = []
    for group_indicators in INDICATOR_GROUPS.values():
        for jp_name in group_indicators.keys():
            all_indicators.append(jp_name)
    
    with col1:
        st.subheader("🏛️ 分析対象国選択")
        
        region_group = st.selectbox("地域グループから選択", 
                                    ['カスタム選択'] + list(REGION_GROUPS.keys()),
                                    key='pca_region_group')
        
        if region_group == 'カスタム選択':
            all_countries = CENTRAL_AMERICA + SOUTH_AMERICA + SOUTHEAST_ASIA + SOUTH_ASIA + SUB_SAHARAN + MIDDLE_EAST + ['日本', '中国', '英国', '米国']
            selected_countries = st.multiselect("分析対象国を選択（複数選択可）",
                                               all_countries,
                                               default=CENTRAL_AMERICA[:5],
                                               key='pca_countries_custom')
        else:
            selected_countries = REGION_GROUPS[region_group]
            st.info(f"選択された地域：{region_group}（{len(selected_countries)}カ国）")
    
    with col2:
        st.subheader("📊 分析対象指標選択")
        selected_indicators_jp = st.multiselect("分析対象指標を選択（複数選択可）",
                                               all_indicators, default=all_indicators[:6],
                                               key='pca_indicators')
    
    if len(selected_countries) < 3 or len(selected_indicators_jp) < 3:
        st.warning("主成分分析には最低3カ国と3指標が必要です")
        return
    
    indicator_mapping = {}
    for group_indicators in INDICATOR_GROUPS.values():
        indicator_mapping.update(group_indicators)
    
    data_matrix = []
    valid_countries = []
    
    for country in selected_countries:
        row = []
        for indicator_jp in selected_indicators_jp:
            indicator_eng = indicator_mapping[indicator_jp]
            data = get_data_for_indicator(df, indicator_eng, 'all', 2024)
            country_data = data[data['国'] == country]
            if not country_data.empty:
                row.append(country_data['値'].values[0])
            else:
                row.append(np.nan)
        
        if not any(np.isnan(row)):
            data_matrix.append(row)
            valid_countries.append(country)
    
    if len(valid_countries) < 3:
        st.warning("十分なデータがある国が不足しています")
        return
    
    data_matrix = np.array(data_matrix)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    pca = PCA(n_components=2)
    country_scores = pca.fit_transform(data_scaled)
    
    indicator_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=country_scores[:, 0], y=country_scores[:, 1],
                            mode='markers+text', text=valid_countries,
                            textposition="top center", name='国',
                            marker=dict(size=12, color='blue')))
    
    fig.add_trace(go.Scatter(x=indicator_loadings[:, 0], y=indicator_loadings[:, 1],
                            mode='markers+text', text=selected_indicators_jp,
                            textposition="top center", name='指標',
                            marker=dict(size=10, color='red', symbol='diamond')))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(title="主成分分析結果（PCA: 2024年）",
                     xaxis_title=f"第1主成分 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                     yaxis_title=f"第2主成分 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                     showlegend=True, height=600)
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 第1主成分の負荷量")
        loadings_pc1 = pd.DataFrame({
            '指標': selected_indicators_jp,
            '負荷量': indicator_loadings[:, 0]
        }).sort_values('負荷量', key=lambda x: x.abs(), ascending=False)
        st.dataframe(loadings_pc1, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📊 第2主成分の負荷量")
        loadings_pc2 = pd.DataFrame({
            '指標': selected_indicators_jp,
            '負荷量': indicator_loadings[:, 1]
        }).sort_values('負荷量', key=lambda x: x.abs(), ascending=False)
        st.dataframe(loadings_pc2, use_container_width=True, hide_index=True)
    
    with st.expander("📈 分析結果の解釈"):
        st.markdown(f"""
        **累積寄与率:** {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.1f}%
        
        **主成分負荷量の見方:**
        - 負荷量が大きい（絶対値が大きい）指標ほど、その主成分への寄与が大きい
        - 第1主成分で負荷量が大きい指標が、その主成分の特徴を表す
        - 第2主成分で負荷量が大きい指標が、第2主成分の特徴を表す
        
        **国の位置:**
        - 近い位置にある国は類似した金融包摂パターンを持っています
        - 指標に近い国はその指標が高い傾向にあります
        - 原点から離れている国ほど特徴的なパターンを持っています
        """)

def machine_learning_analysis(df):
    st.header("🤖 機械学習による回帰分析（2024年データ）")
    
    col1, col2 = st.columns(2)
    
    all_indicators = get_all_indicators(df)
    
    default_features_eng = [
        'Bank or similar financial institution account (%, age 15+)',
        'Mobile money account (%, age 15+)',
        'Saved any money (%, age 15+)',
        'Made a digital payment (%, age 15+)',
        'Borrowed from a formal bank or similar financial institution (%, age 15+)'
    ]
    default_features = [f for f in default_features_eng if f in all_indicators][:5]
    
    target_options = {}
    for wb_name in WB_INDICATORS.keys():
        target_options[f"【世銀】{wb_name}"] = ('wb', wb_name)
    for findex_indicator in all_indicators:
        target_options[f"【Findex】{findex_indicator}"] = ('findex', findex_indicator)
    
    with col1:
        st.subheader("🎯 目的変数選択")
        default_target = "【世銀】経済成長率"
        target_variable_display = st.selectbox("目的変数を選択", list(target_options.keys()), 
                                              index=list(target_options.keys()).index(default_target) if default_target in target_options else 0)
    
    with col2:
        st.subheader("📊 説明変数選択")
        feature_variables = st.multiselect("説明変数（Findex指標）を選択", all_indicators,
                                          default=default_features)
    
    region_scope = st.selectbox("分析対象地域", list(ML_COUNTRY_GROUPS.keys()))
    model_type = st.selectbox("使用するモデル", ["線形回帰", "ランダムフォレスト", "勾配ブースティング"])
    
    if st.button("🚀 分析実行"):
        if not feature_variables:
            st.warning("少なくとも1つの説明変数を選択してください")
            return
        
        target_type, target_indicator = target_options[target_variable_display]
        target_countries_eng = ML_COUNTRY_GROUPS[region_scope]
        
        feature_data_list = []
        for country_eng in target_countries_eng:
            if country_eng in COUNTRY_MAP:
                country_jp = COUNTRY_MAP[country_eng]
                row_data = {'国_英': country_eng, '国_日': country_jp}
                
                for feature_eng in feature_variables:
                    data = get_data_for_indicator(df, feature_eng, 'all', 2024)
                    country_data = data[data['国'] == country_jp]
                    if not country_data.empty:
                        row_data[feature_eng] = country_data['値'].values[0]
                    else:
                        row_data[feature_eng] = np.nan
                
                if target_type == 'findex':
                    target_data = get_data_for_indicator(df, target_indicator, 'all', 2024)
                    target_country_data = target_data[target_data['国'] == country_jp]
                    if not target_country_data.empty:
                        row_data['target_value'] = target_country_data['値'].values[0]
                    else:
                        row_data['target_value'] = np.nan
                
                if not all(np.isnan(row_data[f]) if isinstance(row_data.get(f), float) and np.isnan(row_data.get(f)) else False for f in feature_variables):
                    feature_data_list.append(row_data)
        
        feature_df = pd.DataFrame(feature_data_list)
        
        if target_type == 'wb':
            with st.spinner("世界銀行APIからデータを取得中..."):
                wb_data = get_world_bank_data(WB_INDICATORS[target_indicator], 2024, target_countries_eng)
                
                if wb_data.empty:
                    st.error("世界銀行APIからデータを取得できませんでした")
                    return
                
                wb_data['country_name'] = wb_data['country'].apply(lambda x: x.get('value') if isinstance(x, dict) else None)
                wb_data['target_value'] = wb_data['value']
                wb_data = wb_data[['country_name', 'target_value']].dropna()
                wb_data = wb_data[wb_data['country_name'].isin(target_countries_eng)]
                
                merged_data = pd.merge(wb_data, feature_df, left_on='country_name', right_on='国_英', how='inner')
                merged_data = merged_data.dropna()
        else:
            merged_data = feature_df.dropna()
        
        if len(merged_data) > 0:
            analyzed_countries = merged_data['国_日'].unique()
            st.success(f"✓ {region_scope}の{len(merged_data)}カ国のデータで分析を実行します")
            st.info(f"分析対象国: {', '.join(analyzed_countries)}")
        
        if len(merged_data) < 5:
            st.warning(f"⚠️ 分析に十分なデータがありません（{len(merged_data)}カ国のみ）")
            st.info("ヒント: データが揃っている指標を選択するか、異なる目的変数を試してください。")
            return
        
        X = merged_data[feature_variables].values
        y = merged_data['target_value'].values
        
        st.success("分析完了！")
        
        if model_type == "線形回帰":
            X_with_const = sm.add_constant(X)
            lr_model = sm.OLS(y, X_with_const).fit()
            lr_pred = lr_model.predict(X_with_const)
            lr_r2 = lr_model.rsquared
            lr_mse = mean_squared_error(y, lr_pred)
            lr_mae = mean_absolute_error(y, lr_pred)
            
            st.subheader("📈 モデル性能")
            cols = st.columns(3)
            with cols[0]:
                st.metric("決定係数 (R²)", f"{lr_r2:.3f}")
            with cols[1]:
                st.metric("平均二乗誤差", f"{lr_mse:.2f}")
            with cols[2]:
                st.metric("平均絶対誤差", f"{lr_mae:.2f}")
            
            st.subheader("📊 回帰係数")
            coef_df = pd.DataFrame({
                '指標': feature_variables,
                '回帰係数': lr_model.params[1:],
                'P値': lr_model.pvalues[1:]
            })
            fig_coef = go.Figure()
            colors = ['red' if c < 0 else 'blue' for c in coef_df['回帰係数']]
            fig_coef.add_trace(go.Bar(x=coef_df['回帰係数'], y=coef_df['指標'], 
                                     orientation='h', marker_color=colors))
            fig_coef.update_layout(title="各指標の回帰係数", xaxis_title="回帰係数", yaxis_title="指標")
            st.plotly_chart(fig_coef, use_container_width=True)
            
            st.subheader("📋 回帰係数とP値")
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            
        elif model_type == "ランダムフォレスト":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            train_score = rf_model.score(X_train, y_train)
            test_score = rf_model.score(X_test, y_test)
            rf_pred_test = rf_model.predict(X_test)
            rf_r2 = r2_score(y_test, rf_pred_test)
            rf_mse = mean_squared_error(y_test, rf_pred_test)
            
            st.subheader("📈 モデル性能")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Train Score", f"{train_score:.3f}")
            with cols[1]:
                st.metric("Test Score", f"{test_score:.3f}")
            with cols[2]:
                st.metric("Test R²", f"{rf_r2:.3f}")
            with cols[3]:
                st.metric("Test MSE", f"{rf_mse:.2f}")
            
            st.subheader("📊 特徴量重要度")
            importance_df = pd.DataFrame({
                '指標': feature_variables,
                '重要度': rf_model.feature_importances_
            }).sort_values('重要度', ascending=False)
            fig_imp = px.bar(importance_df, x='重要度', y='指標', orientation='h',
                            title="特徴量重要度ランキング")
            st.plotly_chart(fig_imp, use_container_width=True)
            
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train)
            train_score = gb_model.score(X_train, y_train)
            test_score = gb_model.score(X_test, y_test)
            gb_pred_test = gb_model.predict(X_test)
            gb_r2 = r2_score(y_test, gb_pred_test)
            gb_mse = mean_squared_error(y_test, gb_pred_test)
            
            st.subheader("📈 モデル性能")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Train Score", f"{train_score:.3f}")
            with cols[1]:
                st.metric("Test Score", f"{test_score:.3f}")
            with cols[2]:
                st.metric("Test R²", f"{gb_r2:.3f}")
            with cols[3]:
                st.metric("Test MSE", f"{gb_mse:.2f}")
            
            st.subheader("📊 特徴量重要度")
            importance_df = pd.DataFrame({
                '指標': feature_variables,
                '重要度': gb_model.feature_importances_
            }).sort_values('重要度', ascending=False)
            fig_imp = px.bar(importance_df, x='重要度', y='指標', orientation='h',
                            title="特徴量重要度ランキング（勾配ブースティング）")
            st.plotly_chart(fig_imp, use_container_width=True)

def main():
    st.title("📊 Global Findex 2025 データ分析アプリケーション")
    st.markdown("### 中米グアテマラを中心とした金融包摂データの多角的分析")
    st.markdown("---")
    
    df = load_findex_data()
    
    if df.empty:
        st.error("データを読み込めませんでした")
        return
    
    st.sidebar.title("🔍 分析機能選択")
    analysis_type = st.sidebar.selectbox("分析機能を選択してください",
        ["指標別グラフ可視化", "国別プロファイル", "PCA（主成分分析）", "機械学習分析"])
    
    if analysis_type == "指標別グラフ可視化":
        indicator_analysis(df)
    elif analysis_type == "国別プロファイル":
        country_profile(df)
    elif analysis_type == "PCA（主成分分析）":
        correspondence_analysis(df)
    elif analysis_type == "機械学習分析":
        machine_learning_analysis(df)

if __name__ == "__main__":
    main()
