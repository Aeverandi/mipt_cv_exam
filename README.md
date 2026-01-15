# Экзаменационный проект по дисциплине «Компьютерное зрение» 
Проект представляет собой разворачиваемый сервис с демонстрацией моделями машинного зрения функций детекции (с возможностью дообучения) и генерации. Процесс первичного обучения детективной и генеративной моделей представлен в ноутбуке `FinalWork_ЛобанКМ.ipynb`, рабочая копия которого **[расположена в Google Colab](https://colab.research.google.com/drive/1JsAeqcO9ZGdQYJ7muCw3le2j53yhbXGx?usp=sharing)**. Для обучения был использован датасет [5 Celebrities Face Classification Dataset](https://www.kaggle.com/datasets/raahimrizwan/5-celebrities-face-classification-dataset) с Kaggle.
## Локальный запуск
Для локального разворачивания проекта необходим установленные программы python3 и git, не менее 4,5 Гб физической памяти и желательно (но не обязательно) наличие GPU.
```
git clone https://github.com/Aeverandi/mipt_cv_exam
cd mipt_cv_exam
pip install -r  requirements.txt
streamlit run app.py
```
Первый раз загружается долго, потому что локально загружается более 4Гб модель Stable Diffusion (Yolo8 тоже загружается, но значительно быстрее). Интерфейс реализован на [Streamlit](https://streamlit.io).

Примеры для тестирования функций детекции и дообучения представлены в папке `demo/`. Файлы логов в `logs/` создаются автоматически. Приложение можно запустить и пользователям без GPU - оно оптимизировано и для работы на CPU (правда генерация невероятно долго происходит).

## Файлы для экзаменационной оценки:
* Ноутбук с обучением моделей: **[FinalWork_ЛобанКМ.ipynb](FinalWork_ЛобанКМ.ipynb)** (его [копия в Google Colab](https://colab.research.google.com/drive/1JsAeqcO9ZGdQYJ7muCw3le2j53yhbXGx?usp=sharing))
* Основной исполняемый файл программы с GUI: **[app.py](app.py)** (часть выполняемых функций вынесена в [modules](modules/))
* Отчет: **[report.pdf](report/report.pdf)** (для лучшей наглядности я сделал также **[видеоотчет](https://github.com/Aeverandi/mipt_cv_exam/raw/refs/heads/main/report/video_report.mp4)**)
