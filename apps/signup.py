import time
import os
from typing import Dict
from apps.authentication import check_if_user_exist, create_new_user
import streamlit as st
import apps.authentication
from hydralit import HydraHeadApp

import re


class SignUpApp(HydraHeadApp):
    """
    This is an example signup application to be used to secure access within a HydraApp streamlit application.

    This application is an example of allowing an application to run from the login without requiring authentication.
    
    """

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title


    def run(self) -> None:
        """
        Application entry point.

        """

        st.markdown("<h3 style='text-align: center;'>Secure Lanchang Mekong AI System Signup</h3>", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,2,2])
        c3.image("./resources/lock.png",width=40,)
        

        pretty_btn = """
        <style>
        div[class="row-widget stButton"] > button {
            width: 100%;
        }
        </style>
        <br><br>
        """
        c2.markdown(pretty_btn,unsafe_allow_html=True)
        
        if 'MSG' in os.environ.keys():
            st.info(os.environ['MSG'])
            
        form_data = self._create_signup_form(c2)

        pretty_btn = """
        <style>
        div[class="row-widget stButton"] > button {
            width: 100%;
        }
        </style>
        <br><br>
        """
        c2.markdown(pretty_btn,unsafe_allow_html=True)

        if form_data['submitted']:
            self._do_signup(form_data, c2)


    def _create_signup_form(self, parent_container) -> Dict:

        login_form = parent_container.form(key="login_form", clear_on_submit=True)

        form_state = {}
        form_state['first_name'] = login_form.text_input("Firstname")
        form_state['last_name'] = login_form.text_input("Lastname")
        form_state['username'] = login_form.text_input('Email')
        form_state['password'] = login_form.text_input('Password',type="password")
        form_state['password2'] = login_form.text_input('Confirm Password',type="password")
        #form_state['access_level'] = login_form.selectbox('Example Access Level',(1,2))
        form_state['submitted'] = login_form.form_submit_button('Sign Up')

        if parent_container.button('Login',key='loginbtn'):
            # set access level to a negative number to allow a kick to the unsecure_app set in the parent
            self.set_access(0, None)

            #Do the kick to the signup app
            self.do_redirect()

        return form_state

    def validate_password(self, pwd):
        while True:  
            if (len(pwd)<8):
                return False, "Minimum password 8 characters"
            elif not re.search("[a-z]", pwd):
                return False, "The alphabets must be between [a-z]"
            elif not re.search("[A-Z]", pwd):
                return False, "At least one alphabet should be of Upper Case [A-Z]"
            elif not re.search("[0-9]", pwd):
                return False, "At least 1 number or digit between [0-9]"
            elif not re.search("[_@$]", pwd):
                return False, "At least 1 character from [ _ or @ or $ ]"
            elif re.search("\s", pwd):
                return False, "No white space is allowed"
            else:
                return True, "Valid password"

    def _do_signup(self, form_data, msg_container) -> None:
        if form_data['submitted'] and (form_data['password'] != form_data['password2']):
            st.error('Passwords do not match, please try again.')
        elif check_if_user_exist(form_data['username']):
            st.error("This email user already exists!")
        elif (not form_data['password']) or (not form_data['password2']):
            st.error("Please enter password!")
        elif not re.fullmatch(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', form_data['username']):
            st.error("Invalid email address!")
        else:
            validated_pwd = self.validate_password(form_data['password'])
            if not validated_pwd[0]:
                st.error(validated_pwd[1])
            else:
                with st.spinner("🤓 now redirecting to login...."):
                    create_new_user(form_data['username'], form_data['password'], form_data['first_name'], form_data['last_name'])

                    st.success("New user has been created successfully, please Login to use the system.")

                    #access control uses an int value to allow for levels of permission that can be set for each user, this can then be checked within each app seperately.
                    self.set_access(0, None)

                    #Do the kick back to the login screen
                    #self.do_redirect()

    def _save_signup(self, signup_data):
        #get the user details from the form and save somehwere

        #signup_data
        # this is the data submitted

        #just show the data we captured
        what_we_got = f"""
        captured signup details: \n
        username: {signup_data['username']} \n
        password: {signup_data['password']} \n
        access level: {signup_data['access_level']} \n
        """

        st.write(what_we_got)

