from airflow import DAG
import requests
import pandas as pd
import mysql.connector
import os
from utils import image
from pandarallel import pandarallel
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator


# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

root_path = '/Users/nitinnandansingh/Documents/workspace/e-commerce_data_pipeline'
images_dir = os.path.join(root_path, 'assets/product_images')

products_data_endpoint = "https://fakestoreapi.com/products"
users_data_endpoint = "https://fakestoreapi.com/users"

current_datetime = datetime.now()

default_args = {
    'owner': 'nitin',
    'depends_on_past': False,
    'start_date': current_datetime,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    dag_id = 'ecommerce_data_pipeline',
    default_args=default_args,
    description="A simple e-commerce data pipeline",
    schedule_interval=timedelta(days=1)
)

def fetch_product_data():
    response = requests.get(products_data_endpoint)
    data = response.json()
    return pd.DataFrame(data)

def fetch_users_data():
    response = requests.get(users_data_endpoint)
    data = response.json()
    return pd.DataFrame(data)

def preprocess_products_data(products_df):
    def download_product_images(url, images_dir):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            filename = os.path.join(images_dir, url.split('/')[-1])
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return filename
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None
    
    def parallel_download(url):
        save_dir = images_dir
        return download_product_images(url, save_dir)
    
    def get_count_of_ratings(dict_val):
        return dict_val['count']

    def get_ratings(dict_val):
        return dict_val['rate']

    products_df['images_dir'] = products_df['image'].parallel_apply(parallel_download)

    products_df.drop_duplicates(subset=['id'], inplace=True)

    products_df['count_of_ratings'] = products_df['rating'].parallel_apply(get_count_of_ratings)
    products_df['rating'] = products_df['rating'].parallel_apply(get_ratings)

    products_df['image_valid'] = products_df['images_dir'].parallel_apply(image.is_valid_image)

    products_df = products_df.drop(products_df[products_df['image_valid'] == False].index).reset_index(drop=True)
    products_df.drop(columns=['image_valid'], axis=1, inplace=True)
    return products_df

def preprocess_users_data(users_df):
    users_df.drop(columns=['__v'], axis=1, inplace=True)
    return users_df


def store_data(df, table_name):
    conn = mysql.connector.connect(
        host="localhost",
        user="ecommerce_user",
        password="password",
        database="ecommerce_data"
    )
    cursor = conn.cursor()
    for _, row in df.iterrows():
        if table_name == "products":
            cursor.execute("""
            INSERT INTO products (id, title, price, description, category, image, rating, images_dir, count_of_ratings)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                price = VALUES(price),
                description = VALUES(description),
                category = VALUES(category),
                image = VALUES(image),
                rating = VALUES(rating),
                count_of_ratings = VALUES(count_of_ratings),
                images_dir = VALUES(images_dir);
            """, tuple(row))
        elif table_name=="users":
            cursor.execute("""
            INSERT INTO users (id, email, username, password, name, address, phone)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                email = VALUES(email),
                username = VALUES(username),
                password = VALUES(password),
                name = VALUES(name),
                address = VALUES(address),
                phone = VALUES(phone);
            """, tuple(row))
    conn.commit()
    cursor.close()
    conn.close()

def run_pipeline():
    products_df = fetch_product_data()
    users_df = fetch_users_data()

    products_df = preprocess_products_data(products_df)
    users_df  = preprocess_users_data(users_df)

    store_data(products_df, "products")
    store_data(users_df, "users")

fetch_data_task = PythonOperator(
    task_id='fetch_data',
    python_callable=run_pipeline,
    dag=dag,
)

fetch_data_task






    

