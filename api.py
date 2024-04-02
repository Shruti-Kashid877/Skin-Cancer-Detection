from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from flask import Flask,send_file, render_template, request,session,redirect,url_for
import mysql.connector
from forms import UserForm
import pymysql.cursors

app = Flask(__name__)
UPLOAD_FOLDER = r"D:\7Cancer\static\uploads"


# Load the metadata.csv file
metadata_path = "D:/7Cancer/HAM10000_metadata.csv"
metadata = pd.read_csv(metadata_path)

# Create a dictionary mapping image IDs to diagnoses (dx)
image_dx_mapping = dict(zip(metadata['image_id'], metadata['dx']))

app.secret_key = 'asdfghjkl'
# Connect to your MySQL database (replace the placeholders with your actual database details)
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="admin",
    database="cancer"
)

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='admin',
                             database='cancer',
                             cursorclass=pymysql.cursors.DictCursor)

cursor = db.cursor()

# Load the trained model
model = tf.keras.models.load_model("D:/7Cancer/skin_cancer_detection7.h5")

# Function to preprocess image
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [90, 120])
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Function to predict
def predict_skin_cancer(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    # Assuming you have the labels defined somewhere
    labels = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        form = UserForm(request.form)
        if form.validate():
            username = request.form['username'] 
            password = request.form['password']
            email = request.form['email']
            gender = request.form['gender']
            age = request.form['age']
        

            # Check if the username or email is already registered
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()

            if existing_user:
                return "Username or email already exists. Please choose a different one."

            # Store the registration data in the database
            cursor.execute("INSERT INTO users (username, email, password, gender, age) VALUES (%s, %s, %s, %s, %s)", (username, email, password, gender, age))
            db.commit()
            session['username'] = username
            session['email'] = email
            session['gender'] = gender
            session['age'] = age
        
            return render_template("register.html", msg="Registration Successful!!")
        error_username = form.errors.get('username')[0] if 'username' in form.errors else ''
        error_email = form.errors.get('email')[0] if 'email' in form.errors else ''
        error_password = form.errors.get('password')[0] if 'password' in form.errors else ''
        error_gender = form.errors.get('gender')[0] if 'gender' in form.errors else ''
        error_age = form.errors.get('age')[0] if 'age' in form.errors else ''
        return render_template("register.html", msg="", error_username=error_username, error_email=error_email, error_password=error_password,error_gender=error_gender,error_age=error_age)

    # If it's a GET request, render the registration page
    return render_template("register.html", msg="")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username and password match a user in the database
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        if user:
            # Store user information in the session
            session['user_id'] = user[0]
            session['username'] = user[1]

            return redirect(url_for('home'))

        else:
            return render_template('login1.html', msg='Invalid username or password')

    return render_template('login1.html', msg='')


@app.route('/home')
def home():
    # Check if the user is logged in (session contains user information)
    if 'user_id' in session:
        return render_template('home2.html', username=session['username'])

    # Redirect to login if not logged in
    return redirect(url_for('login'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            message = request.form['message']

            # Store the registration data in the database
            cursor.execute("INSERT INTO contact (username, email, message) VALUES (%s, %s, %s)", (name, email, message))
            db.commit()

            return render_template("contact2.html", msg="Message is delivered!!")
        except KeyError:
            return render_template("contact2.html", error="Error: Please fill out all the fields.")
        except Exception as e:
            return render_template("contact2.html", error=f"Error: {str(e)}")
    else:
        return render_template("contact2.html")



@app.route('/logout')
def logout():
    # Clear the session to log out the user
    session.clear()
    return redirect(url_for('index'))

@app.route('/skininfo')
def skininfo():
    # Add any necessary data to pass to the template
    return render_template('info.html')
    
@app.route('/aboutus')
def aboutus():
    # Add any necessary data to pass to the template
    return render_template('aboutus2.html')

@app.route('/consult',methods=["GET", "POST"])
def consult():
    if request.method == 'POST':
        # handle form data
        name = request.form['name']
        age = request.form['age'] 
        doctor = request.form['doctor']
        conditions = request.form['conditions']
        meds = request.form['meds']
        
        cursor.execute("INSERT INTO consultant (name, age, doctor, conditions, meds) VALUES (%s, %s, %s, %s, %s)", (name, age, doctor, conditions, meds))
        db.commit()
        return render_template('consultant.html', msg="Form Submitted Successfully!")
    return render_template('consultant.html')



@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.root_path, UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            prediction = predict_skin_cancer(file_path)
            # Extract image ID from filename
            image_id = os.path.splitext(file.filename)[0]
            # Get the diagnosis (dx) corresponding to the image ID
            dx = image_dx_mapping.get(image_id, "Unknown")
            

            cursor.execute("INSERT INTO predictions (user_id, image_filename, prediction) VALUES (%s, %s, %s)",
                           (session.get('user_id'), file.filename, prediction))
            db.commit()
            session['prediction'] = prediction
            session['image_filename'] = file.filename
            return jsonify({'prediction': prediction, 'diagnosis': dx})
            return render_template('detect1.html', prediction=prediction, image_loc=file.filename)
            
    return render_template('detect1.html', prediction="", image_loc=None)

@app.route('/view_report')
def view_report():
    prediction = session.get('prediction')
    image_filename = session.get('image_filename')
    if prediction and image_filename:
        # Render report.html with prediction and image filename
        return render_template('report.html', prediction=prediction, image_filename=image_filename)
    else:
        return "Prediction data not found."

# Fetch report data for a specific prediction
@app.route('/get_report_data/<int:prediction_id>', methods=['GET'])
def get_report_data(prediction_id):
    try:
        with connection.cursor() as cursor:
            # Query the database to fetch prediction data based on the prediction ID
            cursor.execute("SELECT * FROM predictions WHERE id = %s", (prediction_id,))
            prediction_data = cursor.fetchone()

            if prediction_data:
                # If prediction data is found, return it as JSON
                return jsonify(prediction_data)
            else:
                # If prediction data is not found, return an error message
                return jsonify({'error': 'Prediction data not found for the given ID'}), 404
    except Exception as e:
        # Handle any exceptions that might occur during database operations
        return jsonify({'error': str(e)}), 500
    finally:
        # Close the database connection after executing the query
        connection.close()

# Render the report template with the prediction data
@app.route('/view_report1/<int:prediction_id>', methods=['GET'])
def view_report1(prediction_id):
    try:
        with connection.cursor() as cursor:
            # Query the database to fetch prediction data based on the prediction ID
            cursor.execute("SELECT * FROM predictions WHERE id = %s", (prediction_id,))
            prediction_data = cursor.fetchone()

            if prediction_data:
                # If prediction data is found, render the report template with the prediction data
                return render_template('report.html', prediction=prediction_data)
            else:
                # If prediction data is not found, render an error message
                return render_template('report.html', error='Prediction data not found for the given ID')
    except Exception as e:
        # Handle any exceptions that might occur during database operations
        return render_template('report.html', error=str(e))
    finally:
        # Close the database connection after executing the query
        connection.close()


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Retrieve predictions from the database
    cursor.execute("SELECT * FROM predictions WHERE user_id = %s ORDER BY timestamp DESC", (session.get('user_id'),))
    predictions = cursor.fetchall()
    
    return render_template('dashboard3.html', predictions=predictions)
    
@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    # SQL query to delete the prediction with the given ID
    delete_query = "DELETE FROM predictions WHERE id = %s"
    cursor.execute(delete_query, (prediction_id,))
    db.commit()
    return redirect(url_for('dashboard'))

@app.route('/profile')
def profile():
    # Check if the user is logged in
    if 'username' in session:
        username = session['username']
        
        # Fetch user data from the database based on the username
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user_data = cursor.fetchone()
        
        # Pass the user data to the profile page template
        return render_template("profile.html", user=user_data)
    else:
        # If the user is not logged in, redirect to the login page
        return redirect(url_for('login'))



@app.route('/symptom', methods=['GET', 'POST'])
def symptom():
    if request.method == 'POST':
        q1 = request.form['q1']
        q2 = request.form['q2']
        q3 = request.form['q3']
        q4 = request.form['q4']
        q5 = request.form['q5']

        # Calculate the number of 'yes' answers
        yes_count = sum([1 for q in [q1, q2, q3, q4, q5] if q.lower() == 'yes'])

        # Determine if skin cancer is malignant or benign based on the count
        result = "Malignant" if yes_count >= 3 else "Benign"

        return render_template('symptom.html', result=result)

    return render_template('symptom.html')


if __name__ == '__main__':
    app.run(debug=True)
