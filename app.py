import streamlit as st
import pandas as pd, joblib, json
from pathlib import Path
from utils.preprocess import load_transactions
from utils.budget_recommend import recommend_budget

st.set_page_config(page_title='BudgetWise', layout='wide')

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / 'data' / 'transactions.csv'
MODEL_PATH = BASE / 'model' / 'expense_classifier.pkl'
FORECAST_CONF = BASE / 'model' / 'forecast_config.json'

# ✅ Load or create transaction file
if DATA_PATH.exists():
    df = load_transactions(DATA_PATH)
else:
    df = pd.DataFrame(columns=['date', 'description', 'amount', 'type', 'category'])

clf = joblib.load(MODEL_PATH)

# -------------------- Sidebar: Add Transaction --------------------
st.sidebar.header('Add Transaction')
with st.sidebar.form('tx_form', clear_on_submit=True):
    ttype = st.selectbox('Type', ['expense', 'income'])
    desc = st.text_input('Description')
    amt = st.number_input('Amount', min_value=0.0, format="%.2f")
    date = st.date_input('Date')
    submitted = st.form_submit_button('Add')

    if submitted:
        new = {
            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
            'description': desc,
            'amount': amt,
            'type': ttype,
            'category': None
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)

        # Auto-categorize if expense
        if ttype == 'expense' and desc.strip() != '':
            pred = clf.predict([desc])[0]
            df.loc[df.index[-1], 'category'] = pred

        df.to_csv(DATA_PATH, index=False)
        st.success('Transaction added and saved.')

# -------------------- Dashboard --------------------
st.header('Dashboard')
st.subheader('Recent Transactions')
if not df.empty:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_display = df.sort_values('date', ascending=False).reset_index(drop=True)
    st.dataframe(df_display.tail(15))
else:
    st.info('No transactions yet. Add one from the sidebar.')

# -------------------- Charts --------------------
st.subheader('Expenses by Category')
if 'category' in df.columns and not df[df['type'] == 'expense'].empty:
    cat = df[df['type'] == 'expense'].groupby('category')['amount'].sum().sort_values(ascending=False)
    st.bar_chart(cat)
else:
    st.info('No categorized expenses to show.')

st.subheader('Monthly Expense Trend (Naive Forecast)')
if not df.empty:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    expenses = df[df['type'] == 'expense'].copy()
    monthly = expenses.set_index('date').resample('M')['amount'].sum()
    st.line_chart(monthly)

    conf = json.load(open(FORECAST_CONF))
    window = conf.get('window', 3)
    if len(monthly) >= 1:
        ma = monthly.rolling(window=window).mean()
        last_ma = ma.dropna().iloc[-1] if not ma.dropna().empty else monthly.iloc[-1]
        st.info(f'Naive forecast for next month (moving average, window={window}): {last_ma:.2f}')
else:
    st.info('No expense data for trend/forecast.')

# -------------------- Recommendations --------------------
st.header('Recommendations & Alerts')
try:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    recs = recommend_budget(df, monthly_limit=200)
    for r in recs:
        st.write('- ' + r)
except Exception as e:
    st.write('Could not compute recommendations:', e)

# -------------------- Export --------------------
st.header('Export')
csv = df.to_csv(index=False).encode('utf-8')
st.download_button('Download Transactions CSV', data=csv,
                   file_name='transactions.csv', mime='text/csv')
