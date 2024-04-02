from wtforms import Form, StringField, EmailField,PasswordField, validators,IntegerField

class UserForm(Form):
    username =StringField('username',[
        validators.DataRequired(),
        validators.length(min=4,max=10, message="Username must be between 4 and 10 characters.")])

    email=EmailField('email',[
        validators.DataRequired(message="Email is required."),
        validators.Email(message="Invalid email address.")])
    
    password = PasswordField('Password', [
        validators.DataRequired(),  # Ensures the field is not empty
        validators.Regexp(
            regex=r'^(?=.*[A-Z])(?=.*[!@#$%^&*()])(?=.*[0-9]).{8,}$',
            message="Password must contain at least one capital letter, one symbol, one number, and be at least 8 characters long."
        )
    ])

    gender = StringField('username',[
        validators.DataRequired(message="Gender is required.")
    ])

    age = IntegerField('Age', [
        validators.DataRequired(message="Age is required."),
        validators.NumberRange(min=1, message="Age must be a number greater than 0.")
    ])
