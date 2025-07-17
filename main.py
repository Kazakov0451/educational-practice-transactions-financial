import kagglehub
import pandas as pd
import os
import numpy as np
import shutil
import seaborn as sns
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

path = kagglehub.dataset_download('computingvictor/transactions-fraud-datasets')

source_path = f'{path}/transactions_data.csv'
destination_path = 'data/transactions_data.csv'

os.makedirs('data', exist_ok=True)
shutil.copy(source_path, destination_path)

df = pd.read_csv(destination_path)

print("=== Первые 5 строк ===")
print(df.head())

print("=== Последние 5 строк ===")
print(df.tail())

print("=== Случайная выборка строк ===")
print(df.sample(5, random_state=42))

print("=== Информация о датафрейме ===")
df.info(memory_usage='deep')

print("\n=== Основная статистика числовых признаков ===")
print(df.describe())

print("\n=== Уникальные значения в категориальных столбцах ===")
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\nУникальные значения в {col}:")
    print(df[col].value_counts())

df.columns = [col.strip().lower() for col in df.columns]
print("\n=== Новые имена столбцов ===")
print(df.columns.tolist())

for col in categorical_cols:
    df[col] = df[col].str.lower()

print("\n=== Пропуски в данных ===")
missing = df.isnull().sum()
print(missing[missing > 0])

df['merchant_state'] = df['merchant_state'].fillna('unknown')
print("Пропуски в merchant_state заполнены значением 'unknown'.")

df['zip'] = df['zip'].fillna(0)
print("Пропуски в zip заполнены значением 0 (нет данных).")

df['errors'] = df['errors'].fillna('no_error')
print("Пропуски в errors заполнены значением 'no_error'.")

print("Удаление не нужных столбцов id, client_id, card_id")
df.drop(columns=['id', 'client_id', 'card_id'], inplace=True)

print("\n=== Пропуски в данных ===")
missing = df.isnull().sum()
print(missing[missing > 0])

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df['amount'] = df['amount'].replace('[$]', '', regex=True)
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

df['merchant_city'] = df['merchant_city'].astype('string')

df['merchant_state'] = df['merchant_state'].astype('string')

df['use_chip'] = df['use_chip'].astype('string')

df['errors'] = df['errors'].astype('string')

print("=== Информация о датафрейме ===")
df.info(memory_usage='deep')

duplicates = df.duplicated().sum()
print(f"\nКоличество дубликатов: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"Удалены {duplicates} дубликатов.")
  
print("Пересчет индексов")
df.reset_index(drop=True, inplace=True)

print(f"Финальный размер датасета: {df.shape}")
print("Предобработка завершена: пропуски обработаны, дубликаты удалены, имена столбцов приведены к единому стилю.")

print("1. Первые 5 строк с колонками от 1 до 4:")
print(df.iloc[:5, 1:5])

print("2. Последние 5 строк с определёнными столбцами:")
print(df.loc[df.index[-5:], ['amount', 'date']])

print("3. 10 случайных строк только с колонками amount и errors:")
print(df.loc[df.sample(10).index, ['amount', 'errors']])

print("4. Строки с чётными индексами:")
print(df.iloc[::2].head())

print("5. Строки по диапазону с шагом:")
print(df.iloc[1000:1100:10])

print("\n ================== \n")

print("1. Транзакции более 500:")
print(df[df['amount'] > 500])

print("2. Транзакции с ошибками:")
print(df[df['errors'] != 'no_error'])

print("3. Транзакции Chip Transaction:")
print(df[df['use_chip'] == 'chip transaction'])

print("4. Транзакции в штате CA:")
print(df[df['merchant_state'] == 'ca'])

print("5. Транзакции с MCC в категории 5411 (продуктовые магазины):")
print(df[df['mcc'] == 5411])

print("Наибольшие суммы:")
print(df.sort_values(by='amount', ascending=False).head())

print("Наименьшие суммы:")
print(df.sort_values(by='amount', ascending=True).head())

print("Топ-10 транзакций с ошибками:")
print(df[df['errors'] != 'no_error'].sort_values(by='amount', ascending=False).head(10))

print("Где amount > 1000 и use_chip == 'online transaction':")
print(df.query("amount > 1000 and use_chip == 'online transaction'"))

print("Где merchant_state == 'tx' и errors != 'no_error':")
print(df.query("merchant_state == 'tx' and errors != 'no_error'"))

print("Где mcc == 5411 и amount < 100:")
print(df.query("mcc == 5411 and amount < 100"))

print("Где use_chip == 'chip transaction' и amount > 2000:")
print(df.query("use_chip == 'chip transaction' and amount > 2000"))

print("Где merchant_city == 'los angeles':")
print(df.query("merchant_city == 'los angeles'"))

large_amount = np.where(df['amount'] > 1000)
print("Где amount > 1000:")
print(df.iloc[large_amount])

with_errors = np.where(df['errors'] != 'no_error')
print("Где errors != 'no_error':")
print(df.iloc[with_errors])

in_state = np.where(df['merchant_state'] == 'ny')
print("Где merchant_state == 'ny':")
print(df.iloc[in_state])

low_amount = np.where((df['amount'] < 100) & (df['use_chip'] == 'online transaction'))
print("Где amount < 100 и use_chip == 'online transaction':")
print(df.iloc[low_amount])

in_city = np.where(df['merchant_city'] == 'chicago')
print("Где merchant_city == 'chicago':")
print(df.iloc[in_city])

pivot1 = df.pivot_table(index='merchant_state', values='amount', aggfunc='count')
print(pivot1.sort_values(by='amount', ascending=False).head())

pivot2 = df.pivot_table(index='mcc', values='amount', aggfunc='mean')
print(pivot2.sort_values(by='amount', ascending=False).head())

pivot3 = df.pivot_table(index='merchant_state', columns='errors', values='date', aggfunc='count', fill_value=0)
print(pivot3.head())

agg1 = df.groupby('merchant_state')['amount'].agg(['sum', 'mean', 'max'])
print(agg1.sort_values(by='sum', ascending=False).head())

agg2 = df.groupby('mcc')['amount'].agg(['count', 'mean']).sort_values(by='count', ascending=False)
print(agg2.head())

print("\nЧастота ошибок:")
print(df['errors'].value_counts().head(10))

top_error_states = df[df['errors'] != 'no_error']['merchant_state'].value_counts().head()
print("\nГде больше всего ошибок:")
print(top_error_states)

chip_usage = df['use_chip'].value_counts(normalize=True)
print("\nИспользование чипа:")
print(chip_usage)

numeric_cols = df.select_dtypes(include='number').columns
print("=== describe() по числовым признакам ===")
print(df[numeric_cols].describe())

print("\n=== Медианы ===")
print(df[numeric_cols].median())

print("\n=== Мода ===")
print(df[numeric_cols].mode().iloc[0])

print("\n=== Стандартное отклонение ===")
print(df[numeric_cols].std())

print("\n=== 25%, 50%, 75% квартиль ===")
print(df[numeric_cols].quantile([0.25, 0.5, 0.75]))

corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Матрица корреляций между числовыми признаками')
plt.show()

high_corr = corr[(corr.abs() > 0.7) & (corr.abs() < 1.0)]
print("=== Высокие корреляции (|r| > 0.7) ===")
print(high_corr.dropna(how='all').dropna(axis=1, how='all'))

for col in ['amount', 'zip']:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f'Распределение признака {col}')
        plt.xlabel(col)
        plt.ylabel('Частота')
        plt.show()

cross_tab = pd.crosstab(df['use_chip'], df['errors'], normalize='index') * 100
print("=== Таблица сопряжённости use_chip и errors (в процентах) ===")
print(cross_tab)

contingency = pd.crosstab(df['use_chip'], df['errors'])
chi2, p, dof, expected = chi2_contingency(contingency)

print(f"Chi²-статистика: {chi2:.2f}")
print(f"p-value: {p:.4f}")
if p < 0.05:
    print("→ Есть статистически значимая зависимость между использованием чипа и наличием ошибки.")
else:
    print("→ Нет статистически значимой зависимости.")

plt.figure(figsize=(8, 4))
plt.hist(df['amount'], bins=100, color='skyblue', edgecolor='black')
plt.title('Распределение суммы транзакций (amount)')
plt.xlabel('Сумма')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

print('\n')

plt.figure(figsize=(14, 7))
df['errors'].value_counts().head(10).plot(kind='bar', color='coral')
plt.title('10 самых частых ошибок в транзакциях')
plt.ylabel('Количество')
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y')
plt.show()

print('\n')

plt.figure(figsize=(6, 6))
df['use_chip'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'orange', 'lightblue'])
plt.title('Распределение способов оплаты')
plt.ylabel('')
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='use_chip', y='amount', data=df)
plt.title('Распределение суммы транзакций по типу оплаты')
plt.xlabel('Тип оплаты')
plt.ylabel('Сумма')
plt.grid(True)
plt.show()

print('\n')

plt.figure(figsize=(10, 5))
sns.countplot(data=df[df['errors'] != 'no_error'], x='use_chip', hue='errors', order=df['use_chip'].value_counts().index)
plt.title('Ошибки по типу оплаты')
plt.xticks(rotation=45)
plt.legend(title='Тип ошибки', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y')
plt.show()
