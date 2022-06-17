import streamlit as st
from PIL import Image
import os
import random
import altair as alt
import plotly.express as px
import pandas as pd
import webcolors
from wordcloud import WordCloud
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import base64
import openai
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from credentials import api
from gpt import GPT
from gpt import Example

def load_image(image_file):
    img = Image.open(image_file)
    return img

def get_image_pixel(file_name):
    with Image.open(file_name) as rgb_image:
        image_pixel = rgb_image.getpixel((30,30))
    return image_pixel

def load_image_with_cv(image_file):
    image = Image.open(image_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def color_name(hex_color):
    color =webcolors.hex_to_name(str(hex_color))
    return color



def rgb_to_hex(rgb_color):
    hex_color ='#'
    for i in rgb_color:
        i=int(i)
        hex_color += ("{:02x}".format(i))

    return hex_color

def color_name(hex_color):
    color =webcolors.hex_to_name(str(hex_color))
    return color

def prep_image(raw_img):
    modified_img = cv2.resize(raw_img,(900,600), interpolation = cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img

def color_analysis(img):
    clf =KMeans(n_clusters = 5)
    color_labels= clf.fit_predict(img)
    center_colors= clf.cluster_centers_
    counts= Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    df = pd.DataFrame({'label':hex_colors,'Counts': counts.values()})

    return df

@st.cache
def load_data():
    data = pd.read_csv('df.csv')
    return data
watches = load_data()

def main():
    st.title("Amazon Analytics Made Easy - Watch Collection")

    menu = ["Amazon Watch Collection", "Amazon Watches Insights", "Amazon Search", "Color Identification App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Amazon Watch Collection":
        st.markdown("""
        This App is about collection of 10,000 + watches that are scraped from Amazon.ca
        1. This has many interesting findings like " TOP TEN(10) WATCH BRANDS at AMAZON"

        2. It uses Machine learning alogrithem that analyized 10,000+ watch products and found
        trends for yor easy reference

        3. This is a PoC app and will be soon ready with many more addtional product lines
            a. Fashion Apparels
            b. Shoes
            c. Rugs
        Many Python Libraries are used for this app development:
        WordCloud, Sklearn, OpenAI, Opencv,Streamlit, Pandas, Ploty, seaborn, matlplotlib, OpenCv, Pillow  streamlit, BeautifulSoup


        """ )

        st.subheader("Watches Collection from Amazon")
        text = ' '.join(watches['Brand'])
        wordcloud1= WordCloud(background_color="white", width=3000, height=2000, max_words=200).generate(str(text))
        plt.imshow(wordcloud1)
        plt.axis("off")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.pyplot()


        st.subheader("Top 20 Alltime Reviewed Watches across 10K+ Collection  ")
        top_reviewed_watches =watches.groupby(['Url_img'])['ReviewCount'].sum().sort_values(ascending=False)[:20]

        images = list(top_reviewed_watches.index)
        st.image(images, use_column_width=False, caption=["Top Watch "+str(i) for i in range(1,21)])




    elif choice =="Amazon Watches Insights":

        st.subheader('Amazon Watches Insights')

        if st.checkbox('Show Watches Information Table'):
            st.subheader('Amazon Watches Data')
            st.write(watches)
        df_bar = watches.groupby(["Brand"])["Price"].mean().sort_values(ascending=False)[:10]
        st.subheader('Top 10 brands with highest average price across their productline')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df_bar.plot(kind="bar", ylabel="Average Price in CAD ($)", xlabel="Brand Names")
        figsize =(100,100)
        st.pyplot()

        df_bar = watches.groupby(["Brand"])["Rating"].mean().sort_values(ascending=False)[:10]
        st.subheader('Top 10 rating brands with highest rating across their productline')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df_bar.plot(kind="bar", ylabel="Rating out of 5, Where 5 is being the best", xlabel="Brand Names")
        figsize =(100,100)
        st.pyplot()

        df_bar = watches.groupby(["Brand"])["ReviewCount"].sum().sort_values(ascending=False)[:10]
        st.subheader('Top 10 Watch Brands with Maximum Reviews across their productline')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df_bar.plot(kind="bar", ylabel="Total Number of Reviews Posted", xlabel="Brand Names")
        figsize =(100,100)
        st.pyplot()

    elif choice =="Amazon Search":
        st.subheader("Amazon Search")
        st.markdown("""
        Few Questions that you can ask
        1. How many unique values in brand ?
        2. What is the average price of Fossil?
        3. Find the median price of Timex Brand?
        4. Find the ratings above 4 and price greater than 100 ?

        """)
        if st.checkbox('Show Watches Information Table :'):
            st.subheader('Amazon Watches Data')
            st.write(watches)
        prompt = st.text_input(label = "Please enter your watch search here ...")
        openai.api_key = api
        gpt = GPT(engine='davinci', temperature=0.5, max_tokens=100)
        gpt.add_example(Example('How many unique values in brand?','watches["Brand"].nunique()'))
        gpt.add_example(Example('What is the average price of Fossil?','np.mean(watches.loc[(watches.loc[:,"Brand"]=="Fossil"),"Price"])'))
        gpt.add_example(Example('Find the ratings above 4 and price greater than 100',
                        'watches.loc[(watches.loc[:,"Rating"]>4)&(watches.loc[:,"Price"]>100),["Brand","Description"]]'))
        gpt.add_example(Example('Find the ratings below 4 and price less than 100',
                        'watches.loc[(watches.loc[:,"Rating"]<4)&(watches.loc[:,"Price"]<100),["Brand","Description"]]'))
        gpt.add_example(Example('What is Casio brand maximum price','watches.loc[(watches.loc[:,"Brand"]=="Casio"),"Price"].max()'))
        gpt.add_example(Example('Find the median price of Timex Brand',
                       'np.median(watches.loc[(watches.loc[:,"Brand"]=="Timex"),"Price"])'))
        response = gpt.get_top_reply(prompt)
        st.write(response)
        modified_response = response.split("output: ")[-1].strip('\n')
        ans =compile(modified_response,"<string>","eval")
        st.write(eval(ans))



    else:
        st.subheader("Upload any Image to find Top 5 Colors")

        image_file = st.file_uploader("Upload below", type=['JPEG','PNG','JPG'])
        if image_file is not None:
            img = load_image(image_file)
            st.image(img)

            # Analysis
            # Image Pixel

            image_pixel = get_image_pixel(image_file)
            st.write(image_pixel)

            #Distribution via Kmeans and Counter
            myimage = load_image_with_cv(image_file)
            modified_image = prep_image(myimage)
            pix_df = color_analysis(modified_image)
            p01 = px.pie(pix_df, names='label', values='Counts',color='label')
            st.plotly_chart(p01)


            col1, col2 = st.columns([1,2])
            with col1:
                st.write(image_pixel)
                st.info("Color Distribution")
                st.write(pix_df)

            with col2:
                p02 = px.bar(pix_df, x='label',y='Counts', color="label")
                st.plotly_chart(p02)




if __name__ =='__main__':
    main()
