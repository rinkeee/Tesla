import streamlit as st
import tesla_functions as tf
import plotly.graph_objects as go
from PIL import Image

photo = Image.open('tesla.png')

st.set_page_config(layout='wide')

a, b = st.columns(2)
b.title('Investment Insights')
a.image(photo, width=600)
line = "_"*50
st.write(line)
st.header("1. My portfolio in yearly metrics")

df = tf.get_stock_data()
df_e, df_deposit = tf.get_etoro_data(df=df)
df_tot = tf.combine_etoro_stock_data(df_e=df_e, df=df)
df_yearly = tf.get_yearly_values(df_tot=df_tot, df_e=df_e, df_deposit=df_deposit)
a, b = st.columns(2)

bar_fig1 = tf.make_bar_chart2(df_yearly=df_yearly, width=600, height=400, 
                             name='Stock purchased [$]', column='Stock purchased',red_green=False)
bar_fig2 = tf.make_bar_chart2(df_yearly=df_yearly, width=600, height=400, 
                             name='Profit/loss [$]', column='Profil/loss [$]')
bar_fig3 = tf.make_bar_chart2(df_yearly=df_yearly, width=600, height=400, 
                             name='Portfolio value [$]', column='Portfolio value', red_green=False)
bar_fig4 = tf.make_bar_chart2(df_yearly=df_yearly, width=600, height=400, 
                             name='Profit/loss - deposits [%]', column='Profit/loss - deposits [%]')
bar_fig5 = tf.make_bar_chart2(df_yearly=df_yearly, width=600, height=400, 
                             name='Profit/loss - deposits [%]', column='Profit/loss - deposits [%]', name_y= '%')
bar_fig6 = tf.make_bar_chart2(df_yearly=df_yearly, width=600, height=400, 
                             name='Profit/loss [%]', column='Profit/loss [%]',name_y= '%')


#  go.Bar(name='Portfolio value [$]', x=df_yearly.index, y=df_yearly['Portfolio value'], marker_color='rgb(0,128,128)'),
#         go.Bar(name='Profit/loss [$]', x=df_yearly.index, y=df_yearly['Profil/loss [$]'], marker_color='rgb(0,128,0)'),
#         go.Bar(name='Profit/loss - deposits [$]', x=df_yearly.index, y=df_yearly['Profit/loss - deposits [$]'],
#               marker_color='rgb(128,0,0)')
        
        
        

fig_tot = tf.make_total_plot(df_tot=df_tot, width=600, height=600)
fig_2 = tf.make_portfolio_plot(df_tot=df_tot, width=600, height=600)

a.plotly_chart(bar_fig1, use_container_width=True)
a.plotly_chart(bar_fig2, use_container_width=True)
b.plotly_chart(bar_fig3, use_container_width=True)
b.plotly_chart(bar_fig4, use_container_width=True)
b.plotly_chart(bar_fig5, use_container_width=True)
a.plotly_chart(bar_fig6, use_container_width=True)

st.plotly_chart(fig_tot, use_container_width=True)
st.plotly_chart(fig_2, use_container_width=True)
