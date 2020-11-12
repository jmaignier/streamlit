#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Core packages
import streamlit as st
from streamlit import caching
import pandas as pd
import numpy as np
import aiohttp
import asyncio
import datetime as dt
from datetime import datetime,timedelta, date
import humanize
import plotly.graph_objects as go
import plotly_express as px
from PIL import Image

################## HUMANIZE DATA #################
def humanizing(value,var):
    if var in ['network_rx','network_tx','total_data','network_rx_bis','network_tx_bis','total_data_bis']:
        return humanize.naturalsize(value)
    elif var in ['uptime','uptime_bis','error_uptime','seconds']:
        return humanize.naturaldelta(dt.timedelta(seconds=value))
    else:
        return value
        
#################### ALL DEVICES ##################
async def fetch_all_devices(session,offset):
    async with session.get(f'http://coach.vpn/devices/?offset={offset}') as response:
        return await response.json()
        
async def gather_all_devices(nbre_pages):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_all_devices(session, i*100) for i in  range(nbre_pages)]
        return await asyncio.gather(*tasks)
        
@st.cache(suppress_st_warning=True,allow_output_mutation=False)
def load_all_devices(nbre_pages=10):
    temp = asyncio.run(gather_all_devices(nbre_pages))
    df = pd.concat([pd.DataFrame(temp[o]['results']) for o in range(nbre_pages)])
    utile = pd.concat([df.drop('facts',axis=1),df.facts.apply(pd.Series)],axis=1).reset_index(drop=True)
    utile.url = utile.url.apply(lambda x:x.split('/')[-2])
    utile.project_url = utile.project_url.apply(lambda x:'{}'.format(x.split('/')[-2]))
    utile.created_at = utile.created_at.apply(lambda x:datetime.strptime(x[:19],'%Y-%m-%dT%H:%M:%S'))
    utile.modified_at = utile.modified_at.apply(lambda x:datetime.strptime(x[:19],'%Y-%m-%dT%H:%M:%S'))
    utile.last_seen_at = utile.last_seen_at.apply(lambda x:datetime.strptime(x[:19],'%Y-%m-%dT%H:%M:%S'))
    utile.sim_id = utile.sim_id.apply(lambda x:0 if x==None else 1)
    utile.cell_id = utile.cell_id.apply(lambda x:0 if x==None else 1)
    utile.url = utile.url.apply(lambda x:x.split('/')[-1])
    utile = utile.rename(columns={'project_url':'project_id','url':'device'})
    utile = utile.drop(['analytics','events_url','analytics_url','vid','reset_state','predecessor','serial',
                     'mac_ethernet','mac_wifi'],axis=1).set_index('cn')
    
    return utile

###################### ALL PROJECTS ################
async def fetch_all_projects(session,offset):
    async with session.get(f'http://coach.vpn/projects/?offset={offset}') as response:
        return await response.json()
        
async def gather_all_projects(nbre_pages):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_all_projects(session, i*100) for i in  range(nbre_pages)]
        return await asyncio.gather(*tasks)
        
@st.cache(suppress_st_warning=True,allow_output_mutation=False)
def load_all_projects(nbre_pages=4):
    temp = asyncio.run(gather_all_projects(nbre_pages))
    df = pd.concat([pd.DataFrame(temp[o]['results']) for o in range(nbre_pages)])
    df['url'] = df['url'].apply(lambda x :x.split('/')[-2])
    df['client'] = df['name'].apply(lambda x :x.split(";")[0])
    df = df.set_index('url')
    df['created_at'] = pd.to_datetime(df['created_at'].apply(lambda x :x[:-1]))
    df = df.drop(df[df.name.str.contains(" Trash| SAV|Unassigned Devices| Unassigned Devices|Noname")].index)
    return df
    
##################### ONE PROJECT ####################
async def fetch_one_project(session,project,offset):
    async with session.get(f'http://coach.vpn/projects/{project}/?&offset={offset}') as response:
        return await response.json()

async def gather_project(project,nbre_pages=2):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one_project(session,project,i*100) for i in range(nbre_pages)]
        return await asyncio.gather(*tasks)
        
@st.cache(suppress_st_warning=True,allow_output_mutation=False)
def show_project(project):
    data = asyncio.run(gather_project(project))
    try :
        temp = pd.concat([pd.DataFrame(data[o]['devices']) for o in range(len(data))]).reset_index(drop=True)
        df = temp.drop_duplicates()[['cn','connected','state','synchronized']]
        df = df.rename(columns={'cn':'device'})
        return df
    except :
        raise
        st.error("Sorry, this project is empty for now...")
        
##################### MESSAGES #######################
async def fetch_one_message(session,device_id):
    async with session.get(f'http://coach.vpn/events/?device={device_id}') as response:
        return await response.json()

async def gather_messages(devices):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one_message(session,device_id) for device_id in devices]
        return await asyncio.gather(*tasks)

@st.cache(suppress_st_warning=True)
def load_devices_message(devices,condition=None):
    data = asyncio.run(gather_messages(devices))
    df = pd.DataFrame([[data[o]['results'][0]['device'].split('/')[-2],
                        data[o]['results'][0]['message'],
                        data[o]['results'][0]['count'],
                        data[o]['results'][0]['received_at']]\
                        for o in range(len(data)) if len(data[o]['results'])>0 if data[o]['results'][0]['count']>10],
                        columns=['device_id','message','count','received_at']
                      )
    df = df.sort_values(by='count',ascending=False)
    df['received_at'] = df['received_at'].apply(lambda x:x[:19].replace('T',' '))
    df.device_id = df.device_id.apply(lambda x: f'pkw_{x}')
    return df

#################### MESSAGE INVALID CONFIGURATION #####
async def fetch_all_messages(session,condition,offset):
    async with session.get(f'http://coach.vpn/events/?app=&device=&level=&message={condition}+&name=&offset={offset}&pending=true') as response:
        return await response.json()

async def gather_all_messages(condition,nbre_pages):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_all_messages(session,condition,i*100) for i in range(nbre_pages)]
        return await asyncio.gather(*tasks)
    
@st.cache(suppress_st_warning=True,allow_output_mutation=False)
def load_all_messages_invalid(condition='invalid+configuration+file',nbre_pages=3):
    data = asyncio.run(gather_all_messages(condition,nbre_pages))
    df = pd.DataFrame([[data[o]['results'][n]['device'].split('/')[-2],
                        data[o]['results'][n]['message'],
                        data[o]['results'][n]['count'],
                        data[o]['results'][n]['received_at']]\
                        for o in range(len(data)) for n in range(len(data[o]['results']))],
                        columns=['device_id','message','count','received_at']
                       )
    df['received_at'] = df['received_at'].apply(lambda x:x[:19].replace('T',' '))
    df = df.drop_duplicates(subset='device_id', keep='first')
    df.device_id = df.device_id.apply(lambda x: f'pkw_{x}')
    return df
    
################### DEVICES IN PROJECT ################
async def fetch_one_project(session,project):
    async with session.get(f'http://coach.vpn/projects/{project}/') as response:
        return await response.json()

async def gather_project(list_projects):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one_project(session,project) for project in list_projects]
        return await asyncio.gather(*tasks)

@st.cache(suppress_st_warning=True,allow_output_mutation=False)
def load_projects(list_projects):
    data = asyncio.run(gather_project(list_projects))
    df = pd.DataFrame([[data[o]['url'].split('/')[-2],data[o]['name'],data[o]['name'].split(";")[0],data[o]['count'],data[o]['disconnected']] for o in range(len(data))],columns=['project_id','project_name','client','total_devices','number_offline'])
    df = df.drop_duplicates()
    df['project_name'] = df.project_name.apply(lambda x :x.split(";")[-2]+":"+x.split(";")[-1] if len(x.split(";"))==4 else "Name ?")
    df['project'] = df.project_id+":"+df.project_name
    return df[['client','project','total_devices','number_offline']]
    
###################
async def fetch_one_device(session, device_id):
    async with session.get(f'http://coach.vpn/devices/{device_id}') as response:
        return await response.json()

async def gather_devices(devices):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one_device(session, device_id) for device_id in devices]
        return await asyncio.gather(*tasks)

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def data_B(list_projects):
    data = asyncio.run(gather_project(list_projects))
    temp = pd.concat([pd.DataFrame(data[o]['devices'],columns=['url','cn','connected','state','synchronized']) for o in range(len(data))])
    temp['client'] = [data[o]['name'].split(';')[0] for o in range(len(data)) for n in range(len(data[o]['devices']))]
    temp['project_id'] = [data[o]['url'].split('/')[-2] for o in range(len(data)) for n in range(len(data[o]['devices']))]
    temp['project_name'] = [data[o]['name'] for o in range(len(data)) for n in range(len(data[o]['devices']))]
    temp['project_name'] = temp.project_name.apply(lambda x :x.split(";")[-2]+":"+x.split(";")[-1] if len(x.split(";"))==4 else "Name ?")
    temp['project'] = temp.project_id + ":" + temp.project_name
    temp = temp[temp['connected']==0].reset_index(drop=True)
    temp.rename(columns={'cn':'device_id'},inplace=True)
    ref_devices = temp['device_id'].apply(lambda x:x.split('_')[-1]).unique().tolist()
    prez = asyncio.run(gather_devices(ref_devices))
    mess = asyncio.run(gather_messages(ref_devices))
    temp['last_message'] = [mess[o]['results'][0]['message'] for o in range(len(mess))]
    temp['last_message_date'] = [mess[o]['results'][0]['received_at'][:19].replace('T',' ') for o in range(len(mess))]
    temp['last_seen_at'] = [prez[o]['last_seen_at'] for o in range(len(prez))]
    temp['last_seen_at'] = temp.last_seen_at.apply(lambda x:datetime.strptime(x[:19],'%Y-%m-%dT%H:%M:%S'))
    temp['today'] = pd.to_datetime([datetime.now() for o in range(len(prez))])
    temp['offline_since'] = (temp['today']-temp['last_seen_at']).dt.total_seconds()
    temp['input_voltage'] = [prez[o]['analytics']['input_voltage'] for o in range(len(prez))]
    temp = temp.sort_values(by='offline_since',ascending=False)
    temp['offline_since'] = temp['offline_since'].apply(lambda x:humanizing(x,'seconds'))
    return temp[['client','project','device_id','offline_since','input_voltage','last_message','last_message_date']]

#################### VERSION OF DEVICES ###############################
@st.cache(suppress_st_warning=True)
def load_dev_version(nbre_pages=10):
    data = asyncio.run(gather_all_devices(nbre_pages))
    temp = pd.DataFrame([[data[o]['results'][n]['cn'],data[o]['results'][n]['facts']['umk_version'],data[o]['results'][n]['facts']['version']] for o in range(len(data)) for n in range(len(data[o]['results']))],columns=['device_id','umk_version','version'])
    return temp
    
####################### ANALYTICS #####################
async def fetch_analytics(session, device_id, offset):
    async with session.get(f'http://coach.vpn/analytics/?device={device_id}&offset={offset}') as response:
        return await response.json()
        
async def gather_analytics(devices, nbre_pages):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_analytics(session, device_id, i*100) for device_id in devices for i in range(0,nbre_pages)]
        return await asyncio.gather(*tasks)

async def gather_all_analytics(devices,nbre_pages):
    tasks = [gather_analytics(device,nbre_pages) for device in devices]
    return await asyncio.gather(*tasks)
    
@st.cache(suppress_st_warning=True,allow_output_mutation=False)
def load_analytics(devices,nbre_pages):
    data = asyncio.run(gather_analytics(devices,nbre_pages))
    temp = pd.concat([pd.DataFrame(data[o]['results']) for o in range(len(data))  if len(data[o]['results'])!=0]).reset_index(drop=True)
    temp_bis = temp.drop(['pushed_at'],axis=1)
    temp_bis['device'] = temp_bis['device'].apply(lambda x:x.split('/')[-2])
    temp_bis['received_at'] = temp_bis['received_at'].apply(lambda x: x.split('.')[0].replace('T',' '))
    temp_bis['received_at'] = pd.DatetimeIndex(temp_bis['received_at'],tz='Europe/Paris',ambiguous='NaT')
    temp_bis = temp_bis.drop(temp_bis[temp_bis.received_at.isna()==True].index).reset_index(drop=True)
    temp_bis['total_data'] = temp_bis.network_rx + temp_bis.network_tx
    temp_bis['load_avg_1min'],temp_bis['load_avg_5min'],temp_bis['load_avg_15min'] = temp_bis['load_avg'].str.split(' ',2).str
    temp_bis['load_avg_1min'] = temp_bis['load_avg_1min'].astype("float")
    temp_bis['load_avg_5min'] = temp_bis['load_avg_5min'].astype("float")
    temp_bis['load_avg_15min'] = temp_bis['load_avg_15min'].astype("float")
    
    return temp_bis.drop_duplicates().drop(['load_avg','url'],axis=1)
    
######### Deincremente some variables + process side effects
@st.cache(suppress_st_warning=True,allow_output_mutation=False)
def process_analytics(data):
    reboot = data.reset_index(drop=True)
    reboot['received_at'] = pd.DatetimeIndex(reboot['received_at'],tz='Europe/Paris',ambiguous='NaT')
    reboot = reboot.drop(reboot[reboot.received_at.isna()==True].index).reset_index(drop=True)
    reboot.loc[reboot.uptime.diff()>-3000,'reboot_after'] = True
    reboot.loc[reboot.uptime.diff()<-3000,'reboot_after'] = False
    reboot.reboot_after = reboot.reboot_after.fillna(value=False)
    reboot['uptime_bis'] = reboot.uptime.diff(-1)
    reboot['network_rx_bis'] = reboot.network_rx.diff(-1)
    reboot['network_tx_bis'] = reboot.network_tx.diff(-1)
    reboot = reboot.drop(reboot.loc[reboot.uptime_bis==0].index)
    reboot.loc[reboot.uptime_bis<0,'uptime_bis'] = reboot.loc[reboot.uptime_bis<0,'uptime']
    reboot.loc[reboot.network_tx_bis<0,'network_tx_bis'] = reboot.loc[reboot.network_tx_bis<0,'network_tx']
    reboot.loc[reboot.network_rx_bis<0,'network_rx_bis'] = reboot.loc[reboot.network_rx_bis<0,'network_rx']
    reboot.loc[reboot.index[-1],'uptime_bis'] = reboot.loc[reboot.index[-1],'uptime']
    reboot.loc[reboot.index[-1],'network_tx_bis'] = reboot.loc[reboot.index[-1],'network_tx']
    reboot.loc[reboot.index[-1],'network_rx_bis'] = reboot.loc[reboot.index[-1],'network_rx']
    
    for i in range(1,13):
        try:
            temp = reboot.set_index(['received_at']).loc['2020-%.2d' %i]
            comp = (temp.iloc[-1].name-pd.Timestamp('2020-%.2d-01 00:00:00'%i)).total_seconds()
            diff = temp.iloc[-1]['uptime_bis']-comp
            if diff>0:
                temp_prec = reboot.set_index(['received_at']).loc['2020-%.2d' %(i-1)]
                new_date = (temp_prec.iloc[0].name)+pd.Timedelta(diff-2,unit='s')
                reboot.loc[reboot['received_at']==temp_prec.iloc[0].name,'received_at'] = new_date
                reboot.loc[reboot['received_at']==new_date,'uptime_bis'] = reboot.\
                                                                        loc[reboot['received_at']==new_date,'uptime_bis'].apply(lambda x:x+diff)
                reboot.loc[reboot['received_at']==temp.iloc[-1].name,'uptime_bis'] = reboot.\
                                                                        loc[reboot['received_at']==temp.iloc[-1].name,'uptime_bis'].apply(lambda x:x-diff)
        except:
            pass
    
    reboot['total_data_bis'] = reboot.network_rx_bis + reboot.network_tx_bis
    reboot['data_sur_uptime'] = round(reboot.total_data_bis/reboot.uptime_bis)
    reboot.data_sur_uptime = reboot.data_sur_uptime.astype('float')
    reboot['date_shift'] = reboot.received_at.shift(-1).fillna(reboot.loc[reboot.index[-1],'received_at'])
    reboot['error_uptime'] = (reboot.received_at-reboot['date_shift']-pd.to_timedelta(reboot.uptime_bis,unit='s')).dt.total_seconds()
    reboot.loc[reboot.error_uptime<60,'error_uptime'] = 0
    reboot.loc[reboot.error_uptime>3600,'error_uptime'] = 3599
    reboot['month'] = reboot.received_at.dt.month_name()
    return reboot.drop(['date_shift'],axis=1)
    
#################### LIFETIME ##########################
@st.cache(allow_output_mutation=False,suppress_st_warning=True)
def alive_not_alive(df):
    data = pd.DataFrame((df.received_at-pd.to_timedelta(df.uptime,unit='s')).dt.round('1s'),
                            columns=['received_at']).drop_duplicates(subset=['received_at'],keep=False)
    data['life'] = ['not alive' for o in range(data.shape[0])]
    
    data_bis = df[['received_at']]
    data_bis['life'] = ['{}'.format(df.device.unique()[0]) for o in range(data_bis.shape[0])]
    data_bis['life'] = data_bis['life'].astype('str')
    
    fusion = pd.concat([data,data_bis]).reset_index(drop=True).drop_duplicates(subset=['received_at'],keep=False)
    final = fusion.set_index(pd.DatetimeIndex(fusion['received_at'],tz=None,ambiguous='NaT')).sort_index().\
                resample('H').first().fillna(method='bfill').reset_index(drop=True)
    final['date'] = final['received_at'].dt.date.astype('str')
    final['month'] = final.received_at.apply(lambda x :x.month_name())
    
    return final.drop(final[final['life']=='not alive'].index)
    
#########################################################

liste_templates = ['plotly','ggplot2','seaborn','simple_white','plotly_white', 'plotly_dark',
'presentation', 'xgridoff','ygridoff', 'gridon']
colors_quali = [px.colors.qualitative.swatches().data[o]['y'][0] for o in range(len(px.colors.qualitative.swatches().data))]
color_seq = [px.colors.sequential.swatches().data[o]['y'][0] for o in range(len(px.colors.sequential.swatches().data))]

dict_freq = {'Year':'y','Semester':'6M','Trimester':'3M','Month':'M','Week':'7d','Day':'d','Hour':'H',None:None}
dict_month = dict(January='01',February='02',March='03',April='04',May='05',
June='06',July='07',August='08',September='09',October='10',November='11',
December='12')
dict_agg = dict(temp='mean',mem='mean',network_rx='sum',network_tx='sum',
                data_strength='mean',uptime='sum',wifi_strength='mean',total_data='sum',
                data_sur_uptime='mean',day='last',week_nbre='last',month_name='last',
                month_nbre='last',date='last',year='last',load_avg_1min='mean',
                load_avg_5min='mean',load_avg_15min='mean',error_uptime='sum',uptime_bis='sum',
                total_data_bis='sum',network_rx_bis='sum',network_tx_bis='sum')
dict_graph = {'Line':'xy','Scatter':'xy','Bar':'bar','Pie':'domain',
              'Sunburst':'domain','Treemap':'domain',None:None}
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December']

def main():
    st.title("Dashboard project monitoring")
    if st.button("Clear cache to update dashboard"):
        with st.spinner("Cache being cleared, might take some time"):
            caching.clear_cache()
            st.success("Cache cleared successfully")
    
    all_projects = load_all_projects()
    all_devices = load_all_devices().index.tolist()
    
    df = load_projects(all_projects.index.tolist())
    df_A = df.groupby('client')[['total_devices','number_offline']].sum().sort_values(by='total_devices',ascending=False).reset_index()
    df_A = df_A.query('number_offline>0')
    
    fig_A = go.Figure(data=[go.Table(header=dict(values=list(df_A.columns),
                                                 fill_color='paleturquoise',
                                                 align='center'
                                                 ),
                                      cells=dict(values=[df_A[col] for col in df_A.columns],
                                                 fill_color='lavender',
                                                 align='center'
                                                 )
                                      )
                            ],
                        layout=go.Layout(title=dict(text='All installations',
                                                    x=0.5,
                                                    )
                                         )
                        )
    
    with st.beta_expander("Filter buttons"):
        filtre_1,filtre_2,filtre_3 = st.beta_columns(3)
        
        client = filtre_1.selectbox("Client",
                                    options=['']+df_A.client.tolist()
                                    )
        filtre_2_empty = filtre_2.empty()
        projects_list = filtre_2_empty.multiselect("Project",
                                       options=all_projects.index.tolist(),
                                       format_func=lambda x:x +' : '+all_projects.loc[x]['name']
                                       )
    
    col_A,col_B=st.beta_columns(2)
    col_C,col_D=st.beta_columns(2)
    
    if client:
        search_projects = all_projects.query(f'client=={[client]}').index.tolist()
        df_A = load_projects(search_projects).sort_values(by='number_offline',ascending=False)
        projects_list = filtre_2_empty.multiselect(f"Project(s) for {client}",
                                                   options=df_A.project.unique().tolist(),
                                                   )
        fig_A = go.Figure(data=[go.Table(header=dict(values=list(df_A.columns),
                                                     fill_color='paleturquoise',
                                                     align='center'
                                                     ),
                                         columnwidth=[0.2,0.5,0.15,0.15],
                                         cells=dict(values=[df_A[o] for o in df_A.columns],
                                                    fill_color='lavender',
                                                    align=['center','left','center','center']
                                                    )
                                         )
                                ],
                           layout=go.Layout(title=dict(text=f'{client} - Number offline by project',
                                                       x=0.5
                                                       )
                                            ),
                            )
        df_B = data_B(search_projects).drop('client',axis=1)
        
        if not projects_list:
            fig_B = go.Figure(data=[go.Table(header=dict(values=list(df_B.columns),
                                                         fill_color='paleturquoise',
                                                         align='center'
                                                         ),
                                             columnwidth=[0.25,0.1,0.1,0.1,0.3,0.15],
                                             cells=dict(values=[df_B[o] for o in df_B.columns],
                                                        fill_color='lavender',
                                                        align=['left','center','center','center','left','center']
                                                        )
                                             )
                                    ],
                               layout=go.Layout(title=dict(text=f'{client} - offline devices',
                                                           x=0.5
                                                           )
                                                ),
                               )
        elif projects_list:
            df_B = pd.concat([df_B[df_B.project==proj] for proj in projects_list]).reset_index(drop=True)
            
            fig_B = go.Figure(data=[go.Table(header=dict(values=list(df_B.columns),
                                                         fill_color='paleturquoise',
                                                         align='center'
                                                         ),
                                             columnwidth=[0.4,0.2,0.2,0.2],
                                             cells=dict(values=[df_B[o] for o in df_B.columns],
                                                        fill_color='lavender',
                                                        align=['left','center','center','center']
                                                        )
                                             )
                                     ],
                                layout=go.Layout(title=dict(text=f'{client} - All projects',
                                                            x=0.5,
                                                            )
                                                 )
                                )
            
    df_D = pd.concat([load_devices_message([device.split('_')[-1] for device in all_devices]),load_all_messages_invalid()])
    fig_D = go.Figure(data=[go.Table(header=dict(values=list(df_D.columns),
                                                 fill_color='paleturquoise',
                                                 align='center'
                                                 ),
                                     columnwidth=[0.2,0.4,0.1,0.3],
                                     cells=dict(values=[df_D[o] for o in df_D.columns],
                                                fill_color='lavender',
                                                align='center'
                                                )
                                    )
                            ],
                       layout=go.Layout(title=dict(text='Messages',
                                                   x=0.5
                                                   )
                                        )
                      )

    temp_C = load_dev_version()
    plot_C = col_C.empty()
    fig_C = go.Figure(data=[go.Table(header=dict(values=list(temp_C.columns),
                                                 fill_color='paleturquoise',
                                                 align='center'
                                                       ),
                                     cells=dict(values=[temp_C[o] for o in temp_C.columns],
                                                fill_color='lavender',
                                                align='center'
                                                )
                                     )
                            ],
                      )
    col_A.plotly_chart(fig_A)
    try:
        col_B.plotly_chart(fig_B)
    except:
        pass
    try :
        if df_D.shape[0]!=0:
            col_D.plotly_chart(fig_D)
        else:
            col_D.warning("No significant message")
    except:
        pass
    try:
        plot_C.plotly_chart(fig_C)
        umk_version_dev = col_C.multiselect('umk version',
                                            options=temp_C.umk_version.unique()
                                            )
        version_dev = col_C.multiselect('version',
                                        options=temp_C.version.unique()
                                        )
        if umk_version_dev or version_dev:
            if umk_version_dev and version_dev:
                df_C = temp_C.query(f'version=={version_dev} & umk_version=={umk_version_dev}').reset_index(drop=True)
            elif umk_version_dev and not version_dev:
                df_C = temp_C.query(f'umk_version=={umk_version_dev}').reset_index(drop=True)
            elif not umk_version_dev and version_dev:
                df_C = temp_C.query(f'version=={version_dev}').reset_index(drop=True)
            else:
                df_C = df_C
                
            fig_C = go.Figure(data=[go.Table(header=dict(values=list(df_C.columns),
                                                         fill_color='paleturquoise',
                                                         align='center'
                                                         ),
                                             cells=dict(values=[df_C[o] for o in df_C.columns],
                                                        fill_color='lavender',
                                                        align='center'
                                                        )
                                              )
                                    ],
                                )
            plot_C.plotly_chart(fig_C)
    except:
        pass
        
    st.markdown("---")
    st.title("Visualisation")
    st.markdown("""
        <style type="text/css">
        div.Widget.row-widget.stRadio div {flex-direction:row;
                                           }
        </style>
    """,unsafe_allow_html=True)
    
    Part = st.radio(" ",['By devices', 'By project'])
    
    if Part == 'By devices':
        devices = st.multiselect("Choose devices",
                                 options=all_devices
                                 )
        ref_devices = [device.split('_')[-1] for device in devices]
        nbre_pages = st.number_input("Historical Pages",
                                     min_value=0,
                                     max_value=100000,
                                     value=0,
                                     step=5
                                     )
    
    elif Part == 'By project':
        choice_project = st.multiselect("See a project",
                                        options=all_projects.index,
                                        format_func=lambda x:f"{x} : {all_projects.loc[x]['name']}"
                                        )
        if choice_project:
            project = pd.concat([show_project(project) for project in choice_project]).reset_index(drop=True)
            table_project = go.Figure(data=[go.Table(header=dict(values=list(project.columns),
                                                                 fill_color='steelblue',
                                                                 align='center'
                                                                 ),
                                                     cells=dict(values=[project[o] for o in project.columns],
                                                                fill_color='lavenderblush',
                                                                align='center'
                                                                ),
                                                      )
                                             ],
                                        layout=go.Layout(title=dict(text=f'Project Selected <br> {choice_project}',
                                                                    x=0.5,
                                                                    )
                                                         ),
                                        )
            table_project.update_layout(height=300)
            st.plotly_chart(table_project)
        
            devices = st.multiselect("Choose devices",
                                     options=['all devices']+project.device.unique().tolist()
                                     )
            
            if devices == ['all devices']:
                ref_devices = [device.split('_')[-1] for device in project.device.unique()]
            else:
                ref_devices = [device.split('_')[-1] for device in devices]
                
            nbre_pages = st.sidebar.number_input("Historical Pages",
                                                 min_value=0,
                                                 max_value=100000,
                                                 value=0,
                                                 step=5
                                                 )
        else:
            devices = None
            nbre_pages = 0
        
    if devices and nbre_pages>0 and st.checkbox("Collect data"):
        with st.spinner("Collecting all the data ..."):
                raw_analytics = load_analytics(ref_devices,nbre_pages)
                analytics = pd.concat([process_analytics(raw_analytics[raw_analytics.device==device].\
                                            reset_index(drop=True)) for device in raw_analytics.device.unique()]).\
                                            reset_index(drop=True)
                st.success("Data collected")
                st.markdown(" <br />",unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.beta_expander("Download dataset"):
            path = st.text_input("Choose a path for the file")
            if path and st.button("Download"):
                analytics.to_csv(f'{path}')
        
        with st.beta_expander("Show dataset"):
            df = analytics
            st.write(df)
        
        st.markdown(" <br />",unsafe_allow_html=True)
        st.sidebar.title('Graph styling')
        
        options = df.columns.drop('received_at').tolist()
        
        ## LINE
        with st.sidebar.beta_expander(" Line Chart"):
            y_line = st.selectbox('Choose a variable to display against time',
                                  options,
                                  index=options.index('temp')
                                  )
            log_y = st.radio('Log scale',
                             options=[False,True]
                             )
            color_line = st.selectbox('Variable for color',
                                 options,
                                 index=options.index('device')
                                 )
            if color_line and df.dtypes[color_line]=='object':
                color_sequence_line = st.selectbox('Choose a color scale for line',
                                                   options=colors_quali
                                                   )
                color_discrete_sequence_line = px.colors.qualitative.swatches().data[colors_quali.index(color_sequence_line)].marker['color']
                color_continuous_scale_line = None
            elif color_line and df.dtypes[color_line]!='object':
                color_scale_line = st.selectbox('color scale for line',
                                                options=color_seq
                                                )
                color_continuous_scale_line = px.colors.sequential.swatches().data[color_seq.index(color_scale_line)].marker['color']
                color_discrete_sequence_line = None
            else:
                color_continuous_scale_line = None
                color_discrete_sequence_line = None
        
        with st.beta_expander("Line Plot"):
            fig_line = px.line(df,
                               x='received_at',
                               y=y_line,
                               log_y=log_y,
                               color=color_line,
                               color_discrete_sequence=color_discrete_sequence_line,
                               template='seaborn',
                               title=f"<span style='text-decoration:underline'><i>{y_line} through time</i></span>"
                               )
                             
            fig_line.update_xaxes(rangeslider_visible=True,
                                  rangeselector=dict(buttons=list([
                                                                dict(count=1, label="1d", step="day", stepmode="backward"),
                                                                dict(count=7, label="1w", step="day", stepmode="backward"),
                                                                dict(count=14, label="2w", step="day", stepmode="backward"),
                                                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                                                dict(count=2, label="2m", step="month", stepmode="backward"),
                                                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                                                dict(step="all")
                                                                ])
                                                      )
                                   )
            title = st.text_input("Modify title of line plot")
            if title:
                fig_line.update_layout(title=dict(text=f"<span style='text-decoration:underline'><i>{title}</i></span>"))
                
            st.plotly_chart(fig_line)
            
            
        ## BAR
        with st.sidebar.beta_expander("Bar Chart"):
            x_bar = st.selectbox("x axis",
                                 options=[col for col in df.columns if df.dtypes[col] not in ['float64','int64']]
                                 )
            if x_bar == 'received_at':
                freq = st.radio('Frequency',
                                options=['Semester','Month','Day']
                                )

            y_bar = st.selectbox("y axis",
                                 options=[col for col in df.columns if df.dtypes[col] in ['float64','int64']]
                                 )
            log_y = st.radio("Log scale for y",
                             options=[True,False],
                             index=1
                             )
            agg_bar = st.radio("Aggregate function",
                               options=['sum','mean','min','max','std']
                               )
            color = st.selectbox("color variable",
                                 options=options,
                                 index=options.index('device')
                                 )
            text = st.selectbox("Text",
                                options=[None]+options,
                                index=options.index(y_bar)
                                )
            if df.dtypes[color] in ['float64','int64']:
                color_scale = st.selectbox('Choose a color scale',
                                           options=color_seq
                                           )
                color_continuous_scale = px.colors.sequential.swatches().data[color_seq.index(color_scale)].marker['color']
                color_discrete_sequence = None
            else :
                color_sequence = st.selectbox('Choose a color scale',
                                              options=colors_quali
                                              )
                color_discrete_sequence = px.colors.qualitative.swatches().data[colors_quali.index(color_sequence)].marker['color']
                color_continuous_scale = None
            
        with st.beta_expander("Bar plot"):
            if x_bar =='received_at':
                if color == y_bar:
                    df_bar = df.set_index(x_bar).resample(dict_freq[freq]).agg({y_bar:agg_bar}).reset_index()
                else:
                    df_bar = df.set_index(x_bar).groupby(color).resample(dict_freq[freq]).agg({y_bar:agg_bar}).reset_index()
                
                if freq in  ['Month','Semester']:
                    df_bar[x_bar] = df_bar[x_bar].dt.month_name()
                if freq == 'Day':
                    df_bar[x_bar] = df_bar[x_bar].dt.date
            else :
                if color == y_bar:
                    df_bar = df.groupby([var for var in np.unique([x_bar])]).agg({y_bar:agg_bar}).reset_index()
                else:
                    df_bar = df.groupby([var for var in np.unique([x_bar,color])]).agg({y_bar:agg_bar}).reset_index()
            if text:
                text = df_bar[text].apply(lambda x:humanizing(x,text))
            
            fig_bar = px.bar(df_bar,
                             x=x_bar,
                             y=y_bar,
                             log_y=log_y,
                             color=color,
                             text=text,
                             color_continuous_scale=color_continuous_scale,
                             color_discrete_sequence=color_discrete_sequence,
                             barmode='relative',
                             )
                            
            if x_bar != 'received_at':
                fig_bar.update_xaxes(type='category')
                fig_bar.update_layout(title=dict(text=f"<span style='text-decoration:underline'> <i>{agg_bar} of {y_bar} by {x_bar}</i> </span>",
                                                 x=0.5
                                                 )
                                      )
            else:
                fig_bar.update_layout(title=dict(text=f"<span style='text-decoration:underline'> <i>{agg_bar} of {y_bar} by {freq}</i> </span>",
                                                 x=0.5
                                                 )
                                      )
            title = st.text_input("Modify title of bar")
            if title:
                fig_bar.update_layout(title=dict(text=f"<span style='text-decoration:underline'><i>{title}</i></span>",
                                                  x=0.5)
                                          )
            st.plotly_chart(fig_bar)
            
        ## LIFETIME
        with st.sidebar.beta_expander("Lifetime"):
            date_start = st.date_input("Choose date for debut")
            date_end = st.date_input("Choose date for end")
            utile = analytics.set_index('received_at').loc[str(date_start):str(date_end)].reset_index()
        
        with st.beta_expander("Lifetime"):
            temp = pd.concat([utile.query('device=={}'.format([device])) for device in utile.device.unique()]).reset_index()
            utile = pd.concat([alive_not_alive(temp[temp.device==device].reset_index(drop=True)) for device in temp.device.unique()]).reset_index(drop=True)

            fig = px.scatter(utile,x='received_at',y='life',
                             color='life',
                             labels={'received_at':' ','life':'device'},
                             color_discrete_sequence=px.colors.qualitative.Bold,
                             template='ggplot2',
                             )
                             
            fig.update_yaxes(type='category')
            
            title = st.text_input("Modify title")
            if title:
                fig.update_layout(title=dict(text=f"<span style='text-decoration:underline'><i>{title}</i></span>",x=0.5))
            else:
                fig.update_layout(title=dict(text=f"<span style='text-decoration:underline'> Lifetime from <i>{datetime.strftime(date_start,'%b %d %Y')}</i> to <i>{datetime.strftime(date_end,'%b %d %Y')}</i></span>",x=0.5))
            
            st.plotly_chart(fig)
    
    
if __name__=="__main__":
    main()
