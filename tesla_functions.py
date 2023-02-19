import pandas as pd 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# ## Import Tesla stock data over 5J horizon 

# In[2]:

def get_stock_data():
    pd.set_option('mode.chained_assignment', None) #suppres warning
    df = pd.read_csv('Tesla_5J_trading_data.csv', usecols=('Date', 'Close/Last'))
    df['Close/Last'] = df['Close/Last'].str.replace('$','', regex=True).astype(float)
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    del df['Date']
    df.columns = ['Closing price']
    df = df.sort_index()
    return df 

    # ## Import Etoro trading data
    # 

    # In[3]:

def get_etoro_data(df):
    df_e = pd.read_excel('etoro_data.xlsx', sheet_name='Accountactiviteit', 
                         usecols=['Datum', 'Type', 'Bedrag', 'Eenheden', 'Details', 'Positie-ID'])
    # set datetime to date for compatability with daily Tesla stock data
    df_e.index = pd.to_datetime(df_e['Datum'], format='%d/%m/%Y %H:%M:%S').dt.date
    df_e_index = df_e.index
    del df_e['Datum']
    # create datetime range for indexing data later on
    index_date = pd.date_range(start=df.index[0], end=df.index[-1], freq="D")

    # df with deposits only, for profit calculations later on
    df_deposit = df_e.loc[(df_e['Type'] == 'Storten') | (df_e['Type'] == 'Withdraw Request')]
    del df_deposit['Details']
    del df_deposit['Eenheden']
    del df_deposit['Positie-ID']
    del df_deposit['Type']
    df_deposit.index = pd.to_datetime(df_deposit.index, format='%Y-%m-%d')
    # grab useful rows
    df_e = df_e[df_e['Type'].isin(['Winst/verlies van handelstransactie', 
                                   'Positie openen', 'corp action: Split']) ]
    # make every NaN a 0, for calculation purposes later on
    df_e['Eenheden'] =pd.to_numeric(df_e['Eenheden'], errors ='coerce').fillna(0).astype('float')
    # aply stock splits on the 'eenheden' column, for the trades done prior to split
    previous_split_date = ''
    for x in range(0, len(df_e)):
        if df_e['Type'][x] == 'corp action: Split':
            split = df_e['Details'][x][-3]
            if df_e.index[x] == previous_split_date:
                continue
            else:    
                for y in range(0, x):
                    df_e['Eenheden'][y] = df_e['Eenheden'][y]*float(split)
                previous_split_date = df_e.index[x]    
        else:            
            continue 
    # delete stock split rows
    df_e = df_e[df_e['Type'].isin(['Winst/verlies van handelstransactie', 'Positie openen']) ]
    # calculate stock price at every opening
    df_e['Stock price'] = df_e['Bedrag']/df_e['Eenheden']
    #deleting them for these lines, as they are no openings
    df_e.loc[df_e['Type'] == 'Winst/verlies van handelstransactie', ['Stock price']] = 'NaN' 

    # calculate stock price at ever closing, based on tock price at opening/amount/units
    for x in range(0, len(df_e)):
        if df_e['Type'][x] == 'Positie openen':
            continue 
        else:
            df_ID = df_e.loc[df_e['Positie-ID'] == df_e['Positie-ID'][x]].copy()
            df_e['Stock price'][x] = df_ID['Stock price'][0]+df_ID['Bedrag'][1]/df_ID['Eenheden'][1]
    del df_e['Details']
    #create profit column wit NaN's
    df_e['Profit'] = ['NaN']*len(df_e)

    # simplify despription 'Type' and calculate Profit for closing positions and amount of dollars withdrawn
    for x in range(0, len(df_e)):
        if df_e['Type'][x] == 'Positie openen':
            df_e['Type'][x] = 'Open'
        else:
            df_e['Type'][x] = 'Close'
            df_e['Profit'][x] = df_e['Bedrag'][x]
            df_e['Bedrag'][x] = df_e['Eenheden'][x]*df_e['Stock price'][x]
    # rename stocks to English
    df_e.columns = ['Action','Amount','Units','ID','Stock price', 'Profit']

    return df_e, df_deposit

    # ## Create data for potfolio worth over time

    # In[4]:

def combine_etoro_stock_data(df, df_e):
    index_date = pd.date_range(start=df.index[0], end=df.index[-1], freq="D")
    # split data in df_open and df_close dataframes
    df_open = df_e.loc[df_e['Action'] == 'Open']
    del df_open['Action']
    df_close = df_e.loc[df_e['Action'] == 'Close']
    del df_close['Action']
    # find time span for opened stock positions
    # grab date (index) and value of asset at opening and closing
    for index_open in range(0, len(df_open)):
        start = df_open.index[index_open]
        start_amount = df_open['Amount'][index_open]
        df_select = df_close.loc[df_close['ID'] == df_open['ID'][index_open]]
        if len(df_select) == 0:
            end = df.index[-1] # not closed
            end_amount = df['Closing price'][-1]*df_open['Units'][index_open]
        else:
            end = df_select.index[0]
            end_amount = df_select['Amount'][0]

        # create a new column in the 5J Tesla stock price dataframe for ever trade done
        df[index_open] = [0]*len(df)
        # set value of stock for closing and opening date
        for index_day in range(0,len(df)):
            # if this was the first trade of the day
            if df[index_open][index_day] == 0:
                if df.index[index_day] == start:
                    df[index_open][index_day] = start_amount
                    start_index = index_day
                if df.index[index_day] == end:
                    if df.index[-1] == end:
                        df[index_open][index_day] = end_amount
                        end_index = index_day
                    else:
                        df[index_open][index_day] = 0
                        end_index = index_day
            else:
                if df.index[index_day] == start:
                    df[index_open][index_day] = df[index_open][index_day] + start_amount
                    start_index = index_day
                if df.index[index_day] == end:
                    end_index = index_day
                    if df.index[-1] == end:
                        df[index_open][index_day] = df[index_open][index_day]+ end_amount
                    else:
                        df[index_open][index_day] = df[index_open][index_day]

        for trading_range in range(start_index+1,end_index):
            df[index_open][trading_range] = df['Closing price'][trading_range]*df_open['Units'][index_open]
    # make new dataset to delete 'closing price' tu summ all transactions together        
    df_tot = df.copy()
    del df_tot['Closing price']
    df_tot['Portfolio'] = [0]*len(df)        
    for column in range(0,len(df_open)):
        # add all columns togeter and delete them afterwards
        df_tot['Portfolio'] = df_tot['Portfolio']+df_tot[column]
        del df_tot[column]
    # add the tesla stock price to the new dataframe
    df_tot['TSLA'] = df['Closing price']
    
    del df_open['Units']
    del df_open['ID']
    del df_open['Profit']
    del df_open['Stock price']
    df_open.columns = ['Open']
    del df_close['Units']
    del df_close['ID']
    del df_close['Profit']
    del df_close['Stock price']
    df_close.columns = ['Close']

    df_open = df_open.groupby(df_open.index).sum()
    df_close = df_close.groupby(df_close.index).sum()

    df_open.reindex(index_date)
    df_close.reindex(index_date)
    df_tot = df_tot.join(df_close)
    df_tot = df_tot.join(df_open)

    return df_tot
    # In[25]:

def get_yearly_values(df_tot, df_e, df_deposit):
    df_e.index = pd.to_datetime(df_e.index, format='%Y/%m/%d')
    list_years = map(str, list(range(2020, df_e.index[-1].year+1)))
    list_years = [str(x) for x in list_years]
    dict_years = {}
    df_yearly = pd.DataFrame(index=list_years,columns=['Stock purchased','Stock price 31/12','Portfolio value',
                                                      'Profit/loss [%]', 'Profil/loss [$]', 
                                                       'Profit/loss - deposits [%]', 'Profit/loss - deposits [$]'])
    for year in list_years:
        df_slice = df_e.loc[year]
        df_tot_slice = df_tot.loc[year]
        dict_years[year] = df_slice
        df_open2 = dict_years[year].loc[dict_years[year]['Action'] == 'Open']
        df_close2 = dict_years[year].loc[dict_years[year]['Action'] == 'Close']
        purchased = round(df_open2['Amount'].sum()-df_close2['Amount'].sum())
        df_yearly['Stock purchased'][year] = purchased 
        df_yearly['Portfolio value'][year] = round(df_tot.loc[df_tot_slice.index[-1]]['Portfolio'])
        df_yearly['Stock price 31/12'][year] = round(df_tot.loc[df_slice.index[-1]]['TSLA'],2)

    new_index = pd.date_range(start=df_tot.index[0], end=df_tot.index[-1], freq='D')
    df_tot = df_tot.reindex(new_index, method='ffill') 

    for x in list_years:
        if x == '2020':
            start = df_tot.loc['2020-01-07']['Portfolio']
        else:
            start = df_tot.loc[x+'-01-01']['Portfolio']
        if datetime.timestamp(datetime.strptime(x+'-12-31', '%Y-%m-%d')) > df_tot.index[-1].timestamp():
            end = df_tot.loc[df_tot.index[-1]]['Portfolio']
        else:    
            end = df_tot.loc[x+'-12-31']['Portfolio']   
        df_yearly['Profit/loss [%]'][x] = round((end-start)/start*100,1)
        df_yearly['Profil/loss [$]'][x] = round(end-start)
        deposits = df_deposit.loc[x]['Bedrag'].sum()
        if x == '2020':
            df_yearly['Profit/loss - deposits [%]'][x] = round((end-start-deposits+start)/start*100,1)
            df_yearly['Profit/loss - deposits [$]'][x] = round(end-start-deposits+start)
        else:   
            df_yearly['Profit/loss - deposits [%]'][x] = round((end-start-deposits)/start*100,1)
            df_yearly['Profit/loss - deposits [$]'][x] = round(end-start-deposits)
    df_yearly.index = pd.to_datetime(df_yearly.index, format='%Y')   
    return df_yearly

    # In[41]:

def make_bar_chart(df_yearly, width, height):
    import plotly.graph_objects as go


    fig = go.Figure(data=[
        go.Bar(name='Stock purchased [$]', x=df_yearly.index, y=df_yearly['Stock purchased'], marker_color='rgb(128,0,128)'),
        go.Bar(name='Portfolio value [$]', x=df_yearly.index, y=df_yearly['Portfolio value'], marker_color='rgb(0,128,128)'),
        go.Bar(name='Profit/loss [$]', x=df_yearly.index, y=df_yearly['Profil/loss [$]'], marker_color='rgb(0,128,0)'),
        go.Bar(name='Profit/loss - deposits [$]', x=df_yearly.index, y=df_yearly['Profit/loss - deposits [$]'],
              marker_color='rgb(128,0,0)')
    ])
    # Change the bar mode

    fig.update_layout()

    fig.update_layout(
        title_text='Portfolio status per year at 31/12',
        xaxis_tickfont_size=14,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title='USD ',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    fig.update_layout(width=width,height=height)
    return fig

def make_bar_chart2(df_yearly, width, height, name, column, red_green=True, name_y='USD', color="rgb(255,213,0)"):
    import plotly.graph_objects as go

    if red_green == True:
        fig = go.Figure(data=[
            go.Bar(name=name, x=df_yearly.index, y=df_yearly[column], 
                   marker_color=np.where(df_yearly[column]<0, 'red', 'green')),

        ])
    else:
        fig = go.Figure(data=[
            go.Bar(name=name, x=df_yearly.index, y=df_yearly[column], 
                   marker_color=color),

        ])
        
    
    fig.update_layout(
        title_text=name,
        xaxis_tickfont_size=14,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title=name_y,
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    fig.update_layout(width=width,height=height)
    fig.update_yaxes(dtick=1)
    return fig

def make_total_plot(df_tot, width, height):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df_tot.index, y=df_tot['TSLA'], name="Tesla stock price", marker_color="rgb(255,0,255)"),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=df_tot.index, y=df_tot['Portfolio'], name="Portfolio", marker_color="rgb(255,213,0)"),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=df_tot.index, y=df_tot['Open'], name="Open positions", mode='markers', 
                   marker_color='green'),
        secondary_y=True,

    )

    fig.add_trace(
        go.Scatter(x=df_tot.index, y=df_tot['Close'], name="Close positions", mode='markers', 
                   marker_color='red'),
        secondary_y=True,

    )

    # Add figure title
    fig.update_layout(
        title_text="Tesla stock | Portfolio value | Open/close positions ",
        
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Set x-axis title
    fig.update_layout(width=width,height=height)

    # # Set y-axes titles
    fig.update_yaxes(title_text="Tesla stock price [$]", secondary_y=False)
    fig.update_yaxes(title_text="Portfolio value + open/close positions [$]", secondary_y=True)

    return(fig)


    # ## Plot without tesla stock price 

    # In[10]:

def make_portfolio_plot(df_tot, width, height):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces

    fig.add_trace(
        go.Scatter(x=df_tot.index, y=df_tot['Portfolio'], name="Portfolio", marker_color="rgb(255,213,0)"),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=df_tot.index, y=df_tot['Open'], name="Open positions", mode='markers', 
                   marker_color='green'),
        secondary_y=True,

    )

    fig.add_trace(
        go.Scatter(x=df_tot.index, y=df_tot['Close'], name="Close positions", mode='markers', 
                   marker_color='red'),
        secondary_y=True,

    )

    # Add figure title
    fig.update_layout(
        title_text="Portfolio value | Open/close positions",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(width=width,height=height)
    # Set x-axis title
    fig.update_xaxes(title_text="Time")

    # # Set y-axes titles
    fig.update_yaxes(title_text="Portfolio value [$]", secondary_y=False)
    fig.update_yaxes(title_text="buying/closing positions [$]", secondary_y=True)

    return fig

