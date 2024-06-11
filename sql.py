# Import module 
import sqlite3 
  
# Connecting to sqlite 
conn = sqlite3.connect('train_information.db') 
  
# Creating a cursor object using the  
# cursor() method 
cursor = conn.cursor() 
  
# Creating table 
table ="""CREATE TABLE train_information (
  train_id SERIAL PRIMARY KEY,
  train_name VARCHAR(255) NOT NULL,
  source_station VARCHAR(255) NOT NULL,
  destination_station VARCHAR(255) NOT NULL,
  current_location VARCHAR(255),
  ticket_price DECIMAL(10, 2) NOT NULL,
  departure_time TIME,
  arrival_time TIME
);"""
cursor.execute(table) 
  
# Queries to INSERT records. 
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Howrah Express', 'Howrah Junction', 'Mumbai Central', 'Kharagpur Junction', 1200.50, '06:00:00', '23:59:00')''') 
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Duronto Express', 'Kolkata', 'Delhi', 'Bhopal Junction', 1800.00, '10:00:00', '18:00:00')''') 
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Shatabdi Express', 'Chennai Central', 'Bangalore City', 'Vijayawada Junction', 1500.25, '08:30:00', '14:30:00')''') 
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Goa Express', 'Mumbai Central', 'Goa', 'Pune Junction', 950.75, '14:00:00', '22:00:00')''')
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Uttaranchal Express', 'Delhi', 'Dehradun', 'Saharanpur Junction', 800.00, '07:30:00', '13:30:00')''') 
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Rajdhani Express', 'New Delhi', 'Mumbai Central', 'Nagpur Junction', 2500.00, '18:00:00', '12:00:00')''') 
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Intercity Express', 'Chennai Central', 'Hyderabad', 'Vijayawada Junction', 1100.50, '12:00:00', '20:00:00')''')
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Garib Rath Express', 'Ahmedabad', 'Jaipur', 'Vadodara Junction', 600.25, '21:00:00', '05:00:00') ''') 
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Himalayan Queen', 'Kolkata', 'Darjeeling', 'Malda Town', 750.00, '09:00:00', '17:00:00') ''') 
cursor.execute('''INSERT INTO train_information (train_name, source_station, destination_station, current_location, ticket_price, departure_time, arrival_time)
VALUES ('Sampark Kranti Express', 'Bangalore City', 'Pune Junction', 'Hubli Junction', 1300.75, '16:30:00', '06:30:00') ''') 
  
# Display data inserted 
print("Data Inserted in the table: ") 
data=cursor.execute('''SELECT * FROM train_information''') 
for row in data: 
    print(row) 
  
# Commit your changes in the database     
conn.commit() 
  
# Closing the connection 
conn.close()