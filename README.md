# E-commerce Data Pipeline

This project is an ETL (Extract, Transform, Load) data pipeline for an e-commerce platform, utilizing Apache Airflow for scheduling and orchestration. The pipeline fetches product and user data from an API, preprocesses it, and stores it in a MySQL database.

## Requirements

- Python 3.8+
- Apache Airflow
- Pandas
- Pandarallel
- MySQL Connector

## Setup

1. **Clone the Repository**

   ```sh
   git clone https://github.com/YOUR-USERNAME/e-commerce_data_pipeline.git
   cd e-commerce_data_pipeline

2. Install Dependencies
   pip install -r requirements.txt

3.	Setup Airflow
Initialize the Airflow database:
   airflow db init

3.	Create a user for the Airflow web interface:
airflow users create \
   --username admin \
   --password admin \
   --firstname YOUR_FIRST_NAME \
   --lastname YOUR_LAST_NAME \
   --role Admin \
   --email YOUR_EMAIL


4.	Configure MySQL Database
Login with correct credentials.

You can create the database and tables with:

```
CREATE DATABASE ecommerce_data;

USE ecommerce_data;
```

```
CREATE TABLE products (
    id INT PRIMARY KEY,
    title VARCHAR(255),
    price FLOAT,
    description TEXT,
    category VARCHAR(255),
    image VARCHAR(255),
    rating FLOAT,
    images_dir VARCHAR(255),
    count_of_ratings INT
);

CREATE TABLE users (
    id INT PRIMARY KEY,
    email VARCHAR(255),
    username VARCHAR(255),
    password VARCHAR(255),
    name VARCHAR(255),
    phone VARCHAR(255),
    city VARCHAR(255),
    street VARCHAR(255),
    number VARCHAR(255),
    zipcode VARCHAR(255),
    lat VARCHAR(255),
    long VARCHAR(255)
);
```
Usage

1.	Start Airflow Scheduler and Web Server
    ```
    airflow scheduler
    airflow webserver

1.	Access the Airflow web interface at http://localhost:8080.

2.	Trigger the DAG
    
    In the Airflow web interface, trigger the ecommerce_data_pipeline DAG to start the ETL process.

DAG Workflow

1.	Fetch Data
Fetch product and user data from the Fake Store API.

2.	Preprocess Data

	•	Download product images.

	•	Extract rating details from the product data.

	•	Clean and transform user data.

3.	Convert Data Types
    Ensure the data types match the database schema.

4.	Store Data
    Insert the processed data into the MySQL database.