# To run this file, use the command:
# fastapi dev main.py   

from fastapi import FastAPI
from pydantic import BaseModel

# This is how you connect to and run queries
import mysql.connector

# Login information for MySQL database
# Remember to turn the server on first
config = {
    "host": "gdp-dojo-2025.mysql.database.azure.com",
    "user": "zofia",
    "password": "Password123!",
    "database": "testdb",
    # Change this to file path of your certificate
    "ssl_ca": "/Users/jackm/Documents/DigiCertGlobalRootCA.crt.pem"
}

try:
    # Establish connection
    conn = mysql.connector.connect(**config)

    if conn.is_connected():
        print("Connected to MySQL on Azure successfully!")

    # Create a cursor object
    cursor = conn.cursor()

    # Run query
    cursor.execute("SELECT DATABASE();")
    record = cursor.fetchone()
    print("You're connected to database:", record)
    cursor.execute("SELECT * FROM transact;")
    result = cursor.fetchone()

    # Close resources
    cursor.close()
    conn.close()

except mysql.connector.Error as err:
    print(f"Error: {err}")

# Create Endpoint on 127.0.0.1:8000
# Create app instance
app = FastAPI()


class TransactionParameters(BaseModel):
    trans_date_trans_time: str
    cc_num: int
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: int
    merch_lat: int
    merch_long: int
    is_fraud: int


class Result(BaseModel):
    fraudulent: bool


class Message(BaseModel):
    message: str


class Report(BaseModel):
    # Change this to fit daily report info
    message: str


# Might change these from get requests, if I feel its more fitting
@app.get("/get-daily-report")
async def get_report() -> Report:
    # Get report
    # Change the inputs of Report as needed
    return Report(message="all good")


@app.get("/get-fraudulent-status")
async def check_transaction(transaction: TransactionParameters) -> Result:
    # Run whatever functions you want, all input needed is in the transaction parameter
    # Set fraudulent equal to whatever you decide
    return result(fraudulent=True)

