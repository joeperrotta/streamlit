import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import got
from PIL import Image

image = Image.open('Pulse Logo.png')
st.image(image, use_column_width=True)

st.title('Pulse Network Analytics')
# make Network show itself with repr_html

#def net_repr_html(self):
#  nodes, edges, height, width, options = self.get_network_data()
#  html = self.template.render(height=height, width=width, nodes=nodes, edges=edges, options=options)
#  return html

#Network._repr_html_ = net_repr_html
st.sidebar.title('Choose your favorite Graph')
option=st.sidebar.selectbox('select graph',('Simple','Test_2', 'Test_3'))
physics=st.sidebar.checkbox('add physics interactivity?')
got.simple_func(physics)

if option=='Simple':
  HtmlFile = open("test.html", 'r', encoding='utf-8')
  source_code = HtmlFile.read()
  components.html(source_code, height = 900,width=900)


got.got_func(physics)

if option=='Test_2':
  HtmlFile = open("Test_2.html", 'r', encoding='utf-8')
  source_code = HtmlFile.read()
  components.html(source_code, height = 1200,width=1000)



got.karate_func(physics)

if option=='Test_3':
  HtmlFile = open("Test_3.html", 'r', encoding='utf-8')
  source_code = HtmlFile.read()
  components.html(source_code, height = 1200,width=1000)