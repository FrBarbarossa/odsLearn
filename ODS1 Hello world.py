import numpy as np
import pandas as pd

pd.set_option('display.precision', 2)
df = pd.read_csv('BigML_Dataset_1.csv')

# pd.set_option("display.max_columns", None)
# ^ можно убрать ограничение на вывод строк/столбцов

print(df.head())  # Печать первых пяти строчек

print(df.shape)  # Вывод данных о размерности таблицы

print(df.columns)  # Вывод имён колонок

print(df.info())  # Вывод общих сведений о таблице

df["churn"] = df["churn"].astype("int64")  # Меняем тип данных колонки и проверяем в .info()
print(df.info())

print(df.describe())  # Вывод основной статистики по исчисляемым столбцам

print(df.describe(include=['object', 'bool']))  # Вывод для неисчисляемых типов столбцов, указывать по отдельности

df["churn"] = df["churn"].astype("bool")
print(df['churn'].value_counts())  # Вывод статистики для столбца типа bool

print(df["churn"].value_counts(
    normalize=True))  # Переведём значения bool в "статистику" в виде долей от общего числа записей

print(df.sort_values(by="total day charge",
                     ascending=False).head())  # Сортировка по одному параметру, False в обратном порядке

print(df.sort_values(by=["churn", "state"],
                     ascending=[False, True]).head())  # Сортировка по нескольким столбцам, в разглм порядке

print(df["churn"].mean())  # Соотношение по данному признаку. Аналог value_counts

print(df[df["churn"] == 0].mean())  # Средние значения среди тех, у кого значение churn принимает 0

print(
    df[df["churn"] == 1]["total day minutes"].mean())  # Среднее значение конкретного параметра у заданных пользователей

'''''Объединение критериев отбора и обработка кнкретного параметра'''
print(df[(df["churn"] == 0) & (df["international plan"] == "no")]["total intl minutes"].max())

print(df.loc[0:5, "state":"area code"])  # Вывод среза, первые 5 строк, столбцы от state до area code

print(df.iloc[0:7, 1:4])  # вывод среза только по индексам: строки/столбцы

'''Если нужна первая или последняя строка фрейма, то разрешено использовать:'''
print(df[-1:])
print(df[:1])

'''Применить какую-либо функцию ко всем строкам фрейма:'''
print(df.apply(np.max))

'''Применяем лямба-функцию ко всем элементам столбца state c помощью lambda'''
print(df[df["state"].apply(lambda state: state[0] == "W")].head())

'''Группируем по значениям определённого столбца, выводим основную стат.информацию об интересующих столбцах (columns_to_show)'''
columns_to_show = ["state", "account length", "customer service calls"]
print(df.groupby(["state"])[columns_to_show].describe())

'''Выводим отдельные данные (применяя функции numpy), группируя по столбцу'''
print(df.groupby(['state'])[columns_to_show].agg([np.max, np.min]))

'''Построение таблицы для сравнения значений по двум столбцам (парамертрам)'''
print(pd.crosstab(df["state"], df["churn"], normalize=True))

'''Составим сводную таблицу по заданным параметрам (ищем минимумы и максимумы), группируя по state'''
print(df.pivot_table(
    ["total intl minutes", "total day charge", "total night minutes"],
    ["state"],
    aggfunc=["min", "max"]))
