# DeliveryGuard: beginner setup and test guide

## What this project does

DeliveryGuard analyses the Olist Brazilian e-commerce dataset to understand which orders arrived later than their estimated delivery date. It cleans raw CSV files, calculates summary statistics, displays charts in a Streamlit dashboard, and has an optional Logistic Regression model.

## Folder map

- `data/raw/`: original CSV datasets. Do not edit these by hand.
- `data/processed/cleaned_orders.csv`: one row per delivered order.
- `data/processed/item_level.csv`: one row per item, used for category comparisons.
- `src/data_preprocessing.py`: creates the processed datasets.
- `src/analytics.py`: calculates the numbers shown in the dashboard.
- `src/prediction.py`: trains and evaluates the optional model.
- `app.py`: launches the dashboard.
- `notebooks/exploration.py`: optional first-look analysis and static charts.

## Windows setup (PowerShell)

Open a terminal in the `DeliveryGuard` folder, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Or use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

## Run in this order

1. Rebuild processed data:

   ```powershell
   python -m src.data_preprocessing
   ```

2. Run the automated smoke checks:

   ```powershell
   python smoke_test.py
   ```

3. Launch the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

4. Optional: explore the raw dataset and save static charts:

   ```powershell
   python notebooks/exploration.py
   ```

## What the important terms mean

- **DataFrame:** a table in Pandas.
- **Feature:** a column used to describe an order, such as item count.
- **Target:** the outcome being predicted; here `is_late` (1 = late, 0 = not late).
- **Late rate:** late orders divided by all included orders, multiplied by 100.
- **Train/test split:** train the model on older orders and evaluate it on newer orders.
- **Data leakage:** accidentally giving a model information that would not have been available when making the prediction.

## Interpreting results carefully

This dataset is historical and from a particular Brazilian e-commerce platform. Patterns show associations, not proof of causes. Model results are not a guarantee of future delivery performance. Always report the number of observations alongside a rate.
