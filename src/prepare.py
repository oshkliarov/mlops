import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def prepare_data(raw_dir, output_dir, img_size=(32, 32), limit=2000):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(raw_dir, 'Train.csv')
    df = pd.read_csv(csv_path)

    # Для лаби обмежуємо кількість, щоб пайплайн бігав швидко
    if limit:
        # Перевіряємо, чи є в нас достатньо даних для вибірки
        if len(df) > limit:
            print(f"Обмежуємо датасет до {limit} зображень для DVC-пайплайну.")
            df = df.sample(n=limit, random_state=42)
        else:
            print(f"Датасет малий ({len(df)} рядків). Використовуємо всі дані.")


    data, labels = [], []
    print("Підготовка зображень (Grayscale + Resize + Flatten)...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            img_path = os.path.join(raw_dir, row['Path'])

            image = Image.open(img_path).resize(img_size).convert("L")
            img_array = np.array(image).flatten() / 255.0

            data.append(img_array)
            labels.append(row['ClassId'])
        except Exception as e:
            pass

    X = np.array(data)
    y = np.array(labels)

    #розділення на трейн/тест тут, щоб DVC кешував саме готові вибірки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Збереження підготовлених даних...")
    np.savez_compressed(os.path.join(output_dir, 'train_data.npz'), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_dir, 'test_data.npz'), X=X_test, y=y_test)
    print(f"Файли збережено у {output_dir}")


if __name__ == "__main__":
    prepare_data(raw_dir='data/raw', output_dir='data/prepared')
