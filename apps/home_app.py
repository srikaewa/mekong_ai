import os
import streamlit as st
from hydralit import HydraHeadApp

MENU_LAYOUT = [1,1,1,7,2]

class HomeApp(HydraHeadApp):


    def __init__(self, title = 'Hydralit Explorer', **kwargs):
        self.__dict__.update(kwargs)
        #self.title = title


    #This one method that must be implemented in order to be used in a Hydralit application.
    #The application must also inherit from the hydrapp class in order to correctly work within Hydralit.
    def run(self):

        try:
            st.markdown("<h2 style='text-align: center;'>Welcome to Rice Pest Outbreak & Natural Disaster Monitoring, Forecasting & Warning</h2>",unsafe_allow_html=True)

            col_header_logo_left_far, col_header_logo_left,col_header_text,col_header_logo_right,col_header_logo_right_far = st.columns([1,2,2,2,1])
            
            #col_header_logo_right_far.image(os.path.join(".","resources","hydra.png"),width=100,)

            #if col_header_text.button('This will open a new tab and go'):
            #    self.do_redirect("https://hotstepper.readthedocs.io/index.html")

            _,_,col_logo, col_text,_ = st.columns(MENU_LAYOUT)
            col_logo.image(os.path.join(".","resources","data.png"),width=80,)
            col_text.subheader("This explorer has multiple applications, each application could be run individually, Check out below apps.")

            st.markdown('<br><br>',unsafe_allow_html=True)


            _,_,col_logo, col_text,col_btn = st.columns(MENU_LAYOUT)
            # if col_text.button('Cheat Sheet ➡️'):
            #     self.do_redirect('Cheat Sheet')
            col_logo.image(os.path.join(".","resources","classroom.png"),width=50,)
            col_text.info("This application is about rice pest outbreak modeling. You can all load up related data and build ANN-based LSTM to analyze the pattern of rice pest, Brown Plant Hopper, and predict the future!")

            #The sample content in a sub-section with jump to format.
            _,_,col_logo, col_text,col_btn = st.columns(MENU_LAYOUT)
            # if col_text.button('Sequency Denoising ➡️'):
            #     self.do_redirect('Sequency Denoising')
                
            col_logo.image(os.path.join(".","resources","denoise.png"),width=50,)
            col_text.info("Modeling rice blast outbreak shares the same time-series characteristics with rice pest outbreak model. You can configure different related factor and see the future results!")

            _,_,col_logo, col_text,col_btn = st.columns(MENU_LAYOUT)
            # if col_text.button('Solar Mach ➡️'):
            #     self.do_redirect('Solar Mach')
            col_logo.image(os.path.join(".","resources","satellite.png"),width=50,)
            col_text.info("The natural disaster model is the attemp to utilize the power of deep learning-based LSTM to recognize the pattern of flood and drought. With sufficient data, the model can capture some pattern of these disaster. However, higher quantity and quality of data is still needed to improve this model. Check out and see.")
        
        except Exception as e:
            st.image(os.path.join(".","resources","failure.png"),width=100,)
            st.error('An error has occurred, someone will be punished for your inconvenience, we humbly request you try again.')
            st.error('Error details: {}'.format(e))





