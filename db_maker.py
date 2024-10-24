import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env file
load_dotenv()

# Get environment variables
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST').split(':')[0]  # Remove the port
POSTGRES_PORT = os.getenv('POSTGRES_HOST').split(':')[1]

def create_database(db_name):
    conn = None
    try:
        # Use psycopg2 directly to avoid transaction issues
        conn = psycopg2.connect(
            dbname='postgres',
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        conn.autocommit = True  # Ensure autocommit is enabled
        cursor = conn.cursor()
        
        # Check if the database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            # Create the new database
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")
    except psycopg2.Error as e:
        print(f"Error creating database: {str(e)}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# Create a new database named 'my_new_db'
create_database('my_new_db')
