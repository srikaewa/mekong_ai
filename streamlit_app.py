from hydralit import HydraApp
import hydralit_components as hc
import apps
import streamlit as st

#Only need to set these here as we are add controls outside of Hydralit, to customise a run Hydralit!
st.set_page_config(page_title='MeKong Institute - Rice Pest Outbreak & Natural Disaster Monitoring, Forecasting & Warning Center',page_icon="🐙",layout='wide',initial_sidebar_state='auto',)

if __name__ == '__main__':

    # initialize session state
    st.session_state['model_id'] = "-"

    #---ONLY HERE TO SHOW OPTIONS WITH HYDRALIT - NOT REQUIRED, use Hydralit constructor parameters.
    #st.write('Some options to change the way our Hydralit application looks and feels')
    #c1,c2,c3,c4,_ = st.columns([2,2,2,2,8])
    #hydralit_navbar = c1.checkbox('Use Hydralit Navbar',True)
    #sticky_navbar = c2.checkbox('Use Sticky Navbar',False)
    #animate_navbar = c3.checkbox('Use Animated Navbar',True)
    #hide_st = c4.checkbox('Hide Streamlit Markers',True)

    over_theme = {'txc_inactive': 'black','menu_background':'white','txc_active':'white','option_active':'#809A6F'}

    #this is the host application, we add children to it and that's it!
    app = HydraApp(
        title='Secure Hydralit Data Explorer',
        favicon="🐙",
        #hide_streamlit_markers=hide_st,
        #add a nice banner, this banner has been defined as 5 sections with spacing defined by the banner_spacing array below.
        use_banner_images=["./resources/mekong.png",None,{'header':"<h1 style='text-align:center;padding: 0px 0px;color:grey;font-size:200%;'>Rice Pest Outbreak & Natural Disaster Monitoring, Forecasting & Warning</h1><br>"},None], 
        banner_spacing=[30,10,60,10],
        #use_navbar=hydralit_navbar, 
        #use_navbar=True,
        navbar_sticky=True,
        #navbar_animation=animate_navbar,
        navbar_animation=False,
        navbar_theme=over_theme,
    )

    #Home button will be in the middle of the nav list now
    app.add_app("Home", icon="🏠", app=apps.HomeApp(title='Home'),is_home=True)

    #add all your application classes here
    app.add_app("Cheat Sheet", icon="📚", app=apps.CheatApp(title="Cheat Sheet"))
    app.add_app("Sequency Denoising",icon="🔊", app=apps.WalshApp(title="Sequency Denoising"))
    app.add_app("Sequency (Secure)",icon="🔊🔒", app=apps.WalshAppSecure(title="Sequency (Secure)"))
    app.add_app("Solar Mach", icon="🛰️", app=apps.SolarMach(title="Solar Mach"))
    app.add_app("Spacy NLP", icon="⌨️", app=apps.SpacyNLP(title="Spacy NLP"))
    app.add_app("Uber Pickups", icon="🚖", app=apps.UberNYC(title="Uber Pickups"))
    app.add_app("Solar Mach", icon="🛰️", app=apps.SolarMach(title="Solar Mach"))
    app.add_app("Loader Playground", icon="⏲️", app=apps.LoaderTestApp(title="Loader Playground"))
    app.add_app("Cookie Cutter", icon="🍪", app=apps.CookieCutterApp(title="Cookie Cutter"))

    #app.add_app("Rice Pest Dashboard", icon="💾", app=apps.RicePestDashboard(title="Rice Pest Dashboard"))
    app.add_app("Rice Pest Data Layers", icon="💾", app=apps.RicePestDataLayers(title="Rice Pest Data Layers"))
    app.add_app("Rice Pest Import Data", icon="💾", app=apps.RicePestImportData(title="Rice Pest Import Data"))
    app.add_app("Rice Pest Data Configuration", icon="💾", app=apps.RicePestConfig(title="Rice Pest Data Configuration"))
    app.add_app("Rice Pest - AI Trainer", icon="💾", app=apps.RicePestAIDashboard(title="Rice Pest - AI Trainer"))

    app.add_app("Rice Blast Dashboard", icon="💾", app=apps.RiceBlastDashboard(title="Rice Blast Dashboard"))
    app.add_app("Rice Blast Data Layers", icon="💾", app=apps.RiceBlastDataLayers(title="Rice Pest Data Layers"))
    app.add_app("Rice Blast Import Data", icon="💾", app=apps.RiceBlastImportData(title="Rice Blast Import Data"))
    app.add_app("Rice Blast - AI Trainer", icon="💾", app=apps.RiceBlastAIDashboard(title="Rice Blast - AI Trainer"))
    
    app.add_app("Natural Disaster Dashboard", icon="💾", app=apps.DisasterDashboard(title="Natural Disaster Dashboard"))
    app.add_app("Natural Disaster Data Layers", icon="💾", app=apps.DisasterDataLayers(title="Natural Disaster Data Layers"))
    app.add_app("Natural Disaster - AI Trainer", icon="💾", app=apps.DisasterAIDashboard(title="Natural Disaster - AI Trainer"))

    #we have added a sign-up app to demonstrate the ability to run an unsecure app
    #only 1 unsecure app is allowed
    app.add_app("Signup", icon="🛰️", app=apps.SignUpApp(title='Signup'), is_unsecure=True)

    #we want to have secure access for this HydraApp, so we provide a login application
    #optional logout label, can be blank for something nicer!
    app.add_app("Login", apps.LoginApp(title='Login'),is_login=True) 

    #specify a custom loading app for a custom transition between apps, this includes a nice custom spinner
    app.add_loader_app(apps.MyLoadingApp(delay=0))

    #we can inject a method to be called everytime a user logs out
    #---------------------------------------------------------------------
    # @app.logout_callback
    # def mylogout_cb():
    #     print('I was called from Hydralit at logout!')
    #---------------------------------------------------------------------

    #we can inject a method to be called everytime a user logs in
    #---------------------------------------------------------------------
    # @app.login_callback
    # def mylogin_cb():
    #     print('I was called from Hydralit at login!')
    #---------------------------------------------------------------------

    #if we want to auto login a guest but still have a secure app, we can assign a guest account and go straight in
    app.enable_guest_access()

    #check user access level to determine what should be shown on the menu
    user_access_level, username = app.check_access()

    # If the menu is cluttered, just rearrange it into sections!
    # completely optional, but if you have too many entries, you can make it nicer by using accordian menus
    if user_access_level > 1:
        complex_nav = {
            'Home': ['Home'],
            'Rice Pest 🗺': ['Rice Pest Data Layers', 'Rice Pest Import Data', 'Rice Pest Data Configuration'],
            'Rice Blast 🗺': ['Rice Blast Data Layers', 'Rice Blast Import Data'],
            'Natural Disaster 🗺': ['Natural Disaster Dashboard', 'Natural Disaster Data Layers'],
            'AI Backend 🖥': ['Rice Pest - AI Trainer', 'Rice Blast - AI Trainer', 'Natural Disaster - AI Trainer'],
            #'Loader Playground': ['Loader Playground'],
            #'Intro 🏆': ['Cheat Sheet',"Solar Mach"],
            #'Hotstepper 🔥': ["Sequency Denoising","Sequency (Secure)"],
            #'Clustering': ["Uber Pickups"],
            #'NLP': ["Spacy NLP"],
            #'Cookie Cutter': ['Cookie Cutter']
        }
    elif user_access_level == 1:
        complex_nav = {
            'Home': ['Home'],
            'Rice Pest': ['Rice Pest Data Layers', 'Rice Pest Import Data', 'Rice Pest Data Configuration'],
            'Rice Blast': ['Rice Blast Data Layers', 'Rice Blast Import Data'],
            'Natural Disaster': ['Natural Disaster Dashboard', 'Natural Disaster Data Layers'],
            'AI Backend': ['Rice Pest - AI Trainer', 'Rice Blast - AI Trainer', 'Natural Disaster - AI Trainer'],
            #'Loader Playground': ['Loader Playground'],
            #'Intro 🏆': ['Cheat Sheet',"Solar Mach"],
            #'Hotstepper 🔥': ["Sequency Denoising"],
            #'Clustering': ["Uber Pickups"],
            #'NLP': ["Spacy NLP"],
            #'Cookie Cutter': ['Cookie Cutter']
        }
    else:
        complex_nav = {
            'Home': ['Home'],
        }

  
    #and finally just the entire app and all the children.
    app.run(complex_nav)


    #print user movements and current login details used by Hydralit
    #---------------------------------------------------------------------
    # user_access_level, username = app.check_access()
    # prev_app, curr_app = app.get_nav_transition()
    # print(prev_app,'- >', curr_app)
    # print(int(user_access_level),'- >', username)
    # print('Other Nav after: ',app.session_state.other_nav_app)
    #---------------------------------------------------------------------

