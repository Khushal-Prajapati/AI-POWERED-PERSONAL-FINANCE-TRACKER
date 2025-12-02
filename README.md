BudgetWise - AI-Powered Personal Finance Tracker
================================================

Location: /mnt/data/BudgetWise

Quick start (in VS Code or terminal):
1. Create a virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate   # mac/linux
   venv\Scripts\activate    # windows

2. Install requirements:
   pip install -r requirements.txt

3. Run Streamlit app:
   streamlit run app.py

Project structure:
- app.py                 : Streamlit frontend + glue code
- data/transactions.csv  : Sample transaction dataset
- data/train_samples.csv : Training examples for classifier
- model/expense_classifier.pkl : Trained classifier pipeline
- model/forecast_config.json   : Forecast helper config (moving average)
- utils/preprocess.py    : Data loader and helpers
- utils/budget_recommend.py : Recommendation logic
- utils/train_models.py  : Script to retrain the classifier

Notes:
- Forecasting uses a naive moving average to avoid heavy dependencies.
- You can retrain the classifier by editing data/train_samples.csv and running utils/train_models.py
