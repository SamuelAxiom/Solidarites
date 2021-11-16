import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pickle
import pydeck as pdk
import re
from collections import Counter
from PIL import Image

#import variables

#########################  a faire #########################################
# 
#
###########################################################################"


st.set_page_config(layout="wide")


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	correl=pd.read_csv('graphs.csv',sep='\t')
	questions=pd.read_csv('questions.csv',sep='\t')
	questions.drop([i for i in questions.columns if 'Unnamed' in i],axis=1,inplace=True)
	quest=questions.iloc[3].to_dict()
	codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
	return data,correl,quest,codes

data,correl,questions,codes=load_data()

#st.dataframe(correl)
#st.write(data.columns)
#st.write(correl.shape)

def sankey_graph(data,L,height=600,width=1600):
    """ sankey graph de data pour les catégories dans L dans l'ordre et 
    de hauter et longueur définie éventuellement"""
    
    nodes_colors=["blue","green","grey",'yellow',"coral",'darkviolet','saddlebrown','darkblue','brown']
    link_colors=["lightblue","limegreen","lightgrey","lightyellow","lightcoral",'plum','sandybrown','lightsteelblue','rosybrown']
    
    
    labels=[]
    source=[]
    target=[]
    
    for cat in L:
        lab=data[cat].unique().tolist()
        lab.sort()
        labels+=lab
    
    #st.write(labels)
    
    for i in range(len(data[L[0]].unique())): #j'itère sur mes premieres sources
    
        source+=[i for k in range(len(data[L[1]].unique()))] #j'envois sur ma catégorie 2
        index=len(data[L[0]].unique())
        target+=[k for k in range(index,len(data[L[1]].unique())+index)]
        
        for n in range(1,len(L)-1):
        
            source+=[index+k for k in range(len(data[L[n]].unique())) for j in range(len(data[L[n+1]].unique()))]
            index+=len(data[L[n]].unique())
            target+=[index+k for j in range(len(data[L[n]].unique())) for k in range(len(data[L[n+1]].unique()))]
       
    iteration=int(len(source)/len(data[L[0]].unique()))
    value_prov=[(int(i//iteration),source[i],target[i]) for i in range(len(source))]
    
    
    value=[]
    k=0
    position=[]
    for i in L:
        k+=len(data[i].unique())
        position.append(k)
    
   
    
    for triplet in value_prov:    
        k=0
        while triplet[1]>=position[k]:
            k+=1
        
        df=data[data[L[0]]==labels[triplet[0]]].copy()
        df=df[df[L[k]]==labels[triplet[1]]]
        #Je sélectionne ma première catégorie
        value.append(len(df[df[L[k+1]]==labels[triplet[2]]]))
        
    color_nodes=nodes_colors[:len(data[L[0]].unique())]+["black" for i in range(len(labels)-len(data[L[0]].unique()))]
    #st.write(color_nodes)
    color_links=[]
    for i in range(len(data[L[0]].unique())):
    	color_links+=[link_colors[i] for couleur in range(iteration)]
    #st.write(L,len(L),iteration)
    #st.write(color_links)
   
   
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 30,
      line = dict(color = "black", width = 1),
      label = [i.upper() for i in labels],
      color=color_nodes
      )
      
    ,
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value,
      color = color_links))])
    return fig


def count2(abscisse,ordonnée,dataf,legendtitle='',xaxis=''):
    
    agg=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg2=agg.T/agg.T.sum()
    agg2=agg2.T*100
    agg2=agg2.astype(int)
    
    if abscisse=='MAHFP ':
    	agg=agg.reindex(['january','february','march','april','may','june','july','august','september','october','november','december'])
    	agg2=agg2.reindex(['january','february','march','april','may','june','july','august','september','october','november','december'])
    
    x=agg.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        #st.write(labels,colors)
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Persons','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)
    
    return fig

def pourcent2(abscisse,ordonnée,dataf,legendtitle='',xaxis=''):
    
    agg2=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg=agg2.T/agg2.T.sum()
    agg=agg.T.round(2)*100
    x=agg2.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        
    else:
        #st.write(agg)
        #st.write(agg2)
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)
    
    return fig




continues=['C2_Acresowned','A10_boys','A15_income','D3_LH_income','D15_Nearest_Market_km',\
'F3_Distance_Water_km','F3_Distance_Water_min','B3_FCS','B13_MAHFP','B2_HDDS']
specific=['D17']

img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoSol.png")

def main():	
	
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	topic = st.sidebar.radio('What do you want to do ?',('Display machine learning results','Display maps','Display correlations','Display Sankey Graphs'))
	
	title1, title3 = st.columns([9,2])
	title3.image(img2)
	
	if topic=='Display machine learning results':
		
		title1.title('Machine learning results on predictive model trained on : ')
		title1.title('Question: Do you believe the living conditions in your settlements are adequate?')
		
		
		st.title('')
		st.markdown("""---""")	
		st.subheader('Note:')
		st.write('A machine learning model has been run on the question related to impression of adequate conditions in the settlement, the objective of this was to identify specificaly for these question which are the parameters that influence the most the wellbeing of beneficiaries. The models are run in order to try to predict as precisely as possible the feeling that the respondents expressed in their response to this question. The figures below shows which parameters have a greater impact in the prediction of the model than a normal random aspect (following a statistic normal law)')
		st.write('')
		st.write('Each line of the graph represents one feature of the survey that is important to predict the response to the question.')
		st.write('Each point on the right of the feature name represents one person of the survey. A red point represent a high value to the specific feature and a blue point a low value (a purple one a value inbetween).')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a better note while on the left side, this specific caracter of the person reduced the final result of the prediction. In this case it is 1 for adequate condition and 0 for not adequate so each point on the right tends to contibute to a better feeling.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and the interpretation of the graphs is written below.')
		st.markdown("""---""")	
				
		temp = Image.open('shap.png')
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width = True)
		
		a, b = st.columns([1,1])
		a.caption('WASH Beneficiary: WASH Beneficiary:1 - Not WASH Beneficiary:0')
		a.caption('Household Hunger Scale: Household Hunger Scale Score the highest the most vulnerable')
		a.caption('Food Consumption Score: The highest the most frequent the alimentation is diversified')
		a.caption('Who are your creditors: Shopkeeper: Shopkeeper is creditor:1 - Shopkeeper is not creditor:0')
		a.caption('Household Dietetary Diversity Score: a high score means a higher diversity of alimentation')
		a.caption('Gender of the respondent: Male:1 - Female:0')
		a.caption('Is there any sick person in your household currently?: Yes:1 - No:0')
		b.caption('What is your current MAIN source of water : Other: Responded Other: 1 - Responded something else: 0')
		b.caption('What are the main need in your settlement? Livelihood activities: <br> Responded Livelihood:1 Responded something else:0')
		b.caption('Currently are you receiving any of the following assistance: MPCA? <br> Received MPCA:1 - Did not receive MPCA:0')
		b.caption('According to you what are all the ways of preventing diarrhea? Good personal hygiene: <br> Responded Good personal hygiene: 1 did not respond Good personal hygiene: 0')
		b.caption('Emergency cash assistance beneficiary: <br> Emergency cash assistance beneficiary: 1 - Not Emergency cash assistance beneficiary : 0')
		b.caption('')
		st.write('We can see that the main parameter for feeling adequate is not having been a WASH beneficiary. Is that because those beneficiary were much more vulnerable...???')
		st.write('What comes straight after is the household hunger scale which is to put together with the frequency the family worries about food. So the quantity of food plays an important role in the feeling of the beneficiaries however, surprisingly, the diversity of food tends to deteriorate the feeling of the households when we llok at the FCS and the HDDS. Maybe because wealthier families are less resilient and live their situation with more difficulty that other families that may have been in high vulnerability for a longer period of time.')
		st.write('What comes after is the access to facilities (health, nutrition and water) and the fact not to have a sick person at home.')
		st.write('Another interesting aspect is that people who pay for water tend to feel better than people who do not.')
		st.write('We can see also that having many boys in the household seems also to decrease the feeling of well being. And tha women seem more satisfied with their situation than men.')
		st.write('Finaly we can see that beneficiary of MPCA are among those who feel not adequate however the more they received the better they felt which could be a lead to think that the targeting was right. Livelihood is also the need that the one who are feeling the worst tend to identify as the main priority.')
		
		
		
		st.markdown("""---""")	
	
	
	elif topic=='Display maps':
		
		continues=pickle.load( open( "cont_feat.p", "rb" ) )
		cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
		maps=pd.read_csv('map.csv',sep='\t')
		title1.title('Correlations uncovered from the database in relation to geographic information')
		toshow=correl[correl['variable_x'].fillna('').apply(lambda x: True if 'region' in x else False)]
		#st.write(data['region'].unique())
		st.write(toshow)
		#st.write(questions)
		
		for i in range(len(toshow)):
			
			st.title(toshow.iloc[i]['title'])
			st.write(toshow.iloc[i]['description'])
						
			if toshow.iloc[i]['variable_y'] in cat_cols:
				
				cat,autre=toshow.iloc[i]['variable_y'],toshow.iloc[i]['variable_x']
				df=pd.DataFrame(columns=[cat,autre])
				
				catcols=[j for j in data.columns if cat in j]
				cats=[' '.join(i.split(' ')[1:])[:57] for i in catcols]
				
				for n in range(len(catcols)):
					ds=data[[catcols[n],autre]].copy()
					ds=ds[ds[catcols[n]]==1]
					ds[catcols[n]]=ds[catcols[n]].apply(lambda x: cats[n])
					ds.columns=[cat,autre]
					df=df.append(ds)
							
			else: 
				df=data[[toshow.iloc[i]['variable_x'],toshow.iloc[i]['variable_y']]]
			
			df['persons']=np.ones(len(df))
			
			if toshow.iloc[i]['variable_x']=='region_origin':
				regions=maps[maps['admin'].isin(data['region_origin'].unique().tolist())].copy()
				df=df[df['region_origin']!='0']
			else:
				regions=maps[maps['admin'].isin(data['region'].unique().tolist())].copy()
			#st.write(region)
			
			#st.write(df)
			
			if toshow.iloc[i]['variable_y'] not in continues:
				
				
				
				a=df.groupby([toshow.iloc[i]['variable_x'],toshow.iloc[i]['variable_y']]).aggregate({'persons':'count'}).unstack()
						
				#st.write(toshow.iloc[i]['variable_y']+'##')
				
				coding=codes[codes['list name']==toshow.iloc[i]['variable_y']].copy()
				
				regions_names=regions['admin'].tolist()
				#st.write(coding)
				#st.write(codes)
				#st.write(regions_names)
				region_color=[codes[codes['label']==k].iloc[0]['colorrgb'].lower() for k in regions_names]
				regions['fill_color']=np.array(region_color)
				
				a=a.merge(regions,how='left',left_index=True,right_on='admin').fillna(0)
				#st.write(a)
				#st.write(region)
				
				L=[i[1] for i in a if i[0]=='persons']
				#st.write(L)
				colors=[coding[coding['label']==L[k]].iloc[0]['colorrgb'].lower() for k in range(len(L))]
				labels=[coding[coding['label']==L[k]].iloc[0]['coding'] for k in range(len(L))]
				L=['_'.join(i.split(' ')) for i in L]
				#st.write(L,colors)
				a.columns=L+a.columns.tolist()[len(L):]
				
				a=a.fillna(0)
				a['coordinates']=a['coordinates'].apply(lambda x:eval(x))
				a['centroid']=a['centroid'].apply(lambda x:eval(x))
				a['lat']=a['centroid'].apply(lambda x:x[0])
				a['long']=a['centroid'].apply(lambda x:x[1])
				
				col1,col2=st.columns([2,1])
				
				#st.write(a)
					
				bars=[pdk.Layer('ColumnLayer',data=a,get_position='centroid',get_elevation='+'.join(L[k:]),\
					elevation_scale=500,pickable=True,auto_highlight=True,get_fill_color=colors[k],radius=5000) for k in range(len(L))]
				regions_poly=[pdk.Layer("PolygonLayer",a,id="geojson",opacity=0.5,stroked=False,get_polygon="coordinates",filled=True,\
    					extruded=True,wireframe=True,get_fill_color="fill_color",get_line_color=[0, 0, 0],pickable=True,)]
				text=[pdk.Layer("TextLayer",data=a,get_position=['lat','long-0.15'],filled=False,billboard=False,get_line_color=[180, 180, 180],\
				get_text="admin",get_size=24,get_color=[0,0,0],line_width_min_pixels=1,)]
					
				col1.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',\
				initial_view_state=pdk.ViewState(latitude=14.566,longitude=44.5,zoom=7,height=900,pitch=60),\
				layers=bars+text+regions_poly
				))
				
				
				x=a['admin']
				fig=go.Figure()
				for i in range(len(L)):
					fig.add_trace(go.Bar(x=x, y=a[L[i]], name=labels[i],marker_color='rgb('+colors[i][1:-1]+')'))	
				
				fig.update_layout(barmode='relative',yaxis={'title':'Persons'})
				col2.plotly_chart(fig,use_container_width=True)
				total=a[L[0]].copy()
				for k in range(1,len(L)):
					total+=a[L[k]]
				
				fig2 = go.Figure()
				for i in range(len(L)):
					fig2.add_trace(go.Bar(x=x, y=a[L[i]]/total*100, name=labels[i],marker_color='rgb('+colors[i][1:-1]+')'))	
				fig2.update_layout(barmode='relative',yaxis={'title':'Pourcentage'})
				col2.plotly_chart(fig2,autosize=False,use_container_width=True,height=50)
		
				
			
			elif toshow.iloc[i]['variable_x']!='region_origin':
				
				
				#st.write(df)
				
				col1,col2=st.columns([2,1])
				
				
					
				regions_poly=[pdk.Layer("PolygonLayer",a,id="geojson",opacity=0.5,stroked=False,get_polygon="coordinates",filled=True,\
    					extruded=True,wireframe=True,get_fill_color="fill_color",get_line_color=[0, 0, 0],pickable=True,)]
				text=[pdk.Layer("TextLayer",data=a,get_position=['lat','long'],filled=False,billboard=False,get_line_color=[180, 180, 180],\
				get_text="admin",get_size=24,get_color=[0,0,0],line_width_min_pixels=1,)]
					
				col1.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',\
				initial_view_state=pdk.ViewState(latitude=14.566,longitude=44.5,zoom=7,height=900,pitch=60),\
				layers=text+regions_poly
				))
				
				fig = px.box(df, y='region', x='pay_water',points='all')
				fig.update_layout(yaxis={'title':None},xaxis={'title':'Price of water'})
				col2.plotly_chart(fig,use_container_width=True)
				
			else:
				#st.write(df)
				
				regions['coordinates']=regions['coordinates'].apply(lambda x:eval(x))
				regions['centroid']=regions['centroid'].apply(lambda x:eval(x))
				regions['lat']=regions['centroid'].apply(lambda x:x[0])
				regions['long']=regions['centroid'].apply(lambda x:x[1])
				#st.write(regions)
					
				regions_poly=[pdk.Layer("PolygonLayer",regions,id="geojson",opacity=0.5,stroked=False,get_polygon="coordinates",filled=True,\
    					extruded=True,wireframe=True,get_fill_color="fill_color",get_line_color=[0, 0, 0],pickable=True,)]
				text=[pdk.Layer("TextLayer",data=regions,get_position=['lat','long'],filled=False,billboard=False,get_line_color=[180, 180, 180],\
				get_text="admin",get_size=15,get_color=[0,0,0],line_width_min_pixels=1,)]
					
				st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',\
				initial_view_state=pdk.ViewState(latitude=14.566,longitude=46.5,zoom=6,height=400,pitch=40),\
				layers=text+regions_poly
				))
				
				fig = px.box(df, x='region_origin', y='CSI',points='all')
				fig.update_layout(yaxis={'title':None},xaxis={'title':'Price of water'})
				st.plotly_chart(fig,use_container_width=True,height=300)
				
		
	elif topic=='Display correlations':	
		
		title1.title('Main correlations uncovered from the database')
		continues=pickle.load( open( "cont_feat.p", "rb" ) )
		cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
		
				
		quests=correl[correl['variable_x'].fillna('').apply(lambda x: True if 'region' not in x else False)]
		
		#st.write(quest)
		#st.write(codes)
		#st.write(cat_cols)
		
		#st.write(data['assistancetype'].value_counts())
		
			
							
		for i in quests['variable_x'].unique():
			
			k=0
			st.markdown("""---""")		
			
			quest=quests[quests['variable_x']==i]
			
			#st.write(quest)
			
			if len(quest)>1 or 'bar' in quest['graphtype'].unique():
				col1,col2=st.columns([1,1])
			
			for i in range(len(quest)):
				
				
				
				#st.write(quest.iloc[i]['variable_x']+'##')
				#st.write('in cat cols: ',quest.iloc[i]['variable_x'] in cat_cols)
				
			
								
			
				if quest.iloc[i]['variable_x'] in cat_cols or quest.iloc[i]['variable_y'] in cat_cols:
					
					if quest.iloc[i]['variable_x'] in cat_cols:
						cat,autre=quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']
					else:
						cat,autre=quest.iloc[i]['variable_y'],quest.iloc[i]['variable_x']
					#st.write('cat: ',cat,' et autre: ',autre)
						
					df=pd.DataFrame(columns=[cat,autre])
					
					catcols=[j for j in data.columns if cat in j]
					cats=[' '.join(i.split(' ')[1:])[:57] for i in catcols]
				
					for n in range(len(catcols)):
						ds=data[[catcols[n],autre]].copy()
						ds=ds[ds[catcols[n]]==1]
						ds[catcols[n]]=ds[catcols[n]].apply(lambda x: cats[n])
						ds.columns=[cat,autre]
						df=df.append(ds)
					df['persons']=np.ones(len(df))		
					#st.write(df)		
					
					#st.write(quest.iloc[i]['graphtype'])
						
									
				else:	
					df=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]].copy()
					df['persons']=np.ones(len(df))
				
				if quest.iloc[i]['graphtype']=='sunburst':
					st.subheader(quest.iloc[i]['title'])
					fig = px.sunburst(df.fillna(''), path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], 	values='persons',color=quest.iloc[i]['variable_y'])
					#fig.update_layout(title_text=quest.iloc[i]['variable_x'] + ' and ' +quest.iloc[i]['variable_y'],font=dict(size=20))
					st.plotly_chart(fig,size=1000)
					
					
					
				
				elif quest.iloc[i]['graphtype']=='treemap':
					
					st.subheader(quest.iloc[i]['title'])
					fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons')
					#fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					
					st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					
				
					
				elif quest.iloc[i]['graphtype']=='violin':
					
					
					
					fig = go.Figure()
				
					if quest.iloc[i]['variable_x'].split(' ')[0] in codes['list name'].unique():
						categs = codes[codes['list name']==quest.iloc[i]['variable_x'].split(' ')[0]].sort_values(by='coding')['label'].tolist()				
					
					else:
						categs = df[quest.iloc[i]['variable_x']].unique()
					for categ in categs:
					    fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][df[quest.iloc[i]['variable_x']] == categ],
	                            		y=df[quest.iloc[i]['variable_y']][df[quest.iloc[i]['variable_x']] == categ],
	                            		name=categ,
	                            		box_visible=True,
                           			meanline_visible=True,points="all",))
					fig.update_layout(showlegend=False)
					fig.update_yaxes(range=[-0.1, df[quest.iloc[i]['variable_y']].max()+1],title=quest.iloc[i]['ytitle'])
					k+=1
					
					if len(quest[quest['graphtype']=='violin'])==2:
						
						if k==1:
							col1.subheader(quest.iloc[i]['title'])
							col1.plotly_chart(fig,use_container_width=True)
							col1.write(quest.iloc[i]['description'])
						else:
							col2.subheader(quest.iloc[i]['title'])
							col2.plotly_chart(fig,use_container_width=True)
							col2.write(quest.iloc[i]['description'])
					else:
						st.subheader(quest.iloc[i]['title'])
						st.plotly_chart(fig,use_container_width=True)
						st.write(quest.iloc[i]['description'])
					
									
				elif quest.iloc[i]['graphtype']=='bar':
					
					st.subheader(quest.iloc[i]['title'])
				
					col1,col2=st.columns([1,1])

					fig1=count2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,legendtitle=quest.iloc[i]['legendtitle'],xaxis=quest.iloc[i]['xtitle'])
					
					col1.plotly_chart(fig1,use_container_width=True)
						
					fig2=pourcent2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,legendtitle=quest.iloc[i]['legendtitle'],xaxis=quest.iloc[i]['xtitle'])
					#fig2.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=True,xaxis_tickangle=45)
					col2.plotly_chart(fig2,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					#st.write(df)
						
	
			
	elif topic=='Display Sankey Graphs':
	
		title1.title('Sankey Diagrams')
		st.title('')
		sankey=[i for i in data.columns if data[i].dtype=='object' and i!='nan']
		
		
		sank=data[sankey].fillna('Unknown').copy()
		
		sank['ones']=np.ones(len(sank))
		
		st.title('Main needs identified')
		
		fig=sankey_graph(sank,['main_need','second_need','third_need'],height=600,width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.write(' - '.join(['Main need','Second main need','Third main need']))
		st.plotly_chart(fig,use_container_width=True)
		
		
		
		if st.checkbox('Design my own Sankey Graph'):
			
			st.markdown("""---""")
			selection=st.multiselect('Select features you want to see in the order you want them to appear',\
			 [questions[i] for i in sank.columns if i!='ones'])
			feats=[i for i in questions if questions[i] in selection]
			
			if len(feats)>=2:
				st.write(' - '.join(selection))
				fig3=sankey_graph(sank,feats,height=600,width=1500)
				fig3.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
				st.plotly_chart(fig3,use_container_width=True)	
		
		
		
			
	
	
	

    
 
if __name__== '__main__':
    main()




    
