# Определение возраста по фотографии

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue) ![TensorFlow](https://img.shields.io/badge/-TensorFlow-orange) ![Keras](https://img.shields.io/badge/-Keras-red) ![ResNet50](https://img.shields.io/badge/-ResNet50-lightgrey) ![Pandas](https://img.shields.io/badge/-Pandas-blue) ![NumPy](https://img.shields.io/badge/-NumPy-yellow) ![Matplotlib](https://img.shields.io/badge/-Matplotlib-orange) ![Seaborn](https://img.shields.io/badge/-Seaborn-lightblue)

## Цель проекта

Построить модель компьютерного зрения, которая по фотографии лица
определяет возраст человека.\
Модель обучена на датасете **ChaLearn Looking at People** и достигает
качества **MAE ≈ 6.27**, что удовлетворяет требованиям (\< 7).

## Данные

Используется датасет:

    datasets/faces/
     ├── labels.csv
     └── final_files/

**labels.csv** содержит два столбца:

-   `file_name`
-   `real_age`

Всего изображений: **7591**

## Итоговые метрики

    Test MAE: 6.2721

## Анализ результата

-   Модель хорошо предсказывает частые возраста (20--40)
-   Ошибается чаще на редких (дети, пожилые)
-   Оптимальное число эпох ≈ 8

## Итог

Модель достигает **MAE ≈ 6.27**, успешно справляется с задачей и готова
к дальнейшему улучшению.



