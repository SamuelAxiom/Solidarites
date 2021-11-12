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
	questions=pd.read_csv('questions.csv',sep='\t').iloc[0].to_dict()
	
	return data,correl,questions

data,correl,questions=load_data()

#st.dataframe(correl)
#st.write(data.columns)
#st.write(correl.shape)

def sankey_graph(data,L,height=600,width=1600):
    """ sankey graph de data pour les catégories dans L dans l'ordre et 
    de hauter et longueur définie éventuellement"""
    
    nodes_colors=["blue","green","grey",'yellow',"coral"]
    link_colors=["lightblue","lightgreen","lightgrey","lightyellow","lightcoral"]
    
    
    labels=[]
    source=[]
    target=[]
    
    for cat in L:
        lab=data[cat].unique().tolist()
        lab.sort()
        labels+=lab
    
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
    #print(color_nodes)
    color_links=[]
    for i in range(len(data[L[0]].unique())):
    	color_links+=[link_colors[i] for couleur in range(iteration)]
    #print(L,len(L),iteration)
    #print(color_links)
   
   
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
		codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
		continues=pickle.load( open( "cont_feat.p", "rb" ) )
		cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
		maps=pd.read_csv('map.csv',sep='\t')
		title1.title('Correlations uncovered from the database in relation to geographic information')
		toshow=correl[correl['variable_x'].fillna('').apply(lambda x: True if 'region' in x else False)]
		st.write(data['region'].unique())
		st.write(toshow)
		st.write(cat_cols)
		
		for i in range(len(toshow)):
			
			st.title(toshow.iloc[i]['variable_y'])
						
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
				region=maps[maps['admin'].isin(data['region'].unique().tolist())].copy()
			st.write(region)
			
			st.write(df)
			
			if toshow.iloc[i]['variable_y'] not in continues:
				
				a=df.groupby([toshow.iloc[i]['variable_x'],toshow.iloc[i]['variable_y']]).aggregate({'persons':'count'}).unstack()
				
				
				a=a.merge(region,how='left',left_index=True,right_on='admin').fillna(0)
				
				
				
				L=[i[1] for i in a if i[0]=='persons']
				st.write(a)
				a.columns=L+a.columns.tolist()[len(L):]
				
				a=a.fillna(0)
				a['centroid']=a['centroid'].apply(lambda x:eval(x))
				a['lat']=a['centroid'].apply(lambda x:x[0])
				a['long']=a['centroid'].apply(lambda x:x[1])
				
				col1,col2=st.columns([2,1])
				
				st.write(a)
				
				col1.subheader('coucou')
						
				col1.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',\
				initial_view_state=pdk.ViewState(latitude=14.566,longitude=49.678,zoom=5.5,height=600,pitch=45),\
				layers=[pdk.Layer('ColumnLayer',data=a,get_position='centroid',get_elevation='+'.join(L[k:]),\
				elevation_scale=500,pickable=True,auto_highlight=True,get_fill_color=[0, k*255/(len(L)+1), 0],radius=5000) for k in range(len(L))]
				))
				
				
				
				
				
				fig = go.Figure(go.Bar(x=x, y=df['Poor'], name='Poor',marker_color='red'))
				fig.add_trace(go.Bar(x=x, y=df['Borderline'], name='Borderline',marker_color='yellow'))
				fig.add_trace(go.Bar(x=x, y=df['Acceptable'], name='Acceptable',marker_color='green'))
				fig.update_layout(barmode='relative', \
	        	          xaxis={'title':'Village'},\
        		          yaxis={'title':'Persons'}, legend_title_text='FCS Score')
				col2.plotly_chart(fig,use_container_width=True)
				somme=df['Poor']+df['Borderline']+df['Acceptable']
				fig2 = go.Figure(go.Bar(x=x, y=df['Poor']/somme, name='Poor',marker_color='red'))
				fig2.add_trace(go.Bar(x=x, y=df['Borderline']/somme, name='Borderline',marker_color='yellow'))
				fig2.add_trace(go.Bar(x=x, y=df['Acceptable']/somme, name='Acceptable',marker_color='green'))
				fig2.update_layout(barmode='relative', \
	        	          xaxis={'title':'Village'},\
        		          yaxis={'title':'Pourcentage'}, legend_title_text='FCS Score')
				col2.plotly_chart(fig2,use_container_width=True)
		
				st.write('PS: Position of villages is not the real one as I did not have the GPS coordinates for the benficiaries. This was just to show what could be done')
				
				
			
			else:
				pass
			
			
		
	elif topic=='Display correlations':	
		st.markdown("""---""")
			
		st.header(questions[var])
							
		k=0
			
		for correlation in correl[correl['Variables']==var]['Correl'].unique():
					
					
					
			if correlation in specific:
						
				df=pd.DataFrame(columns=['Challenge','Acres Owned','Chall'])
				dico={'D17_challenge_roads':'Roads','D17_challenge_Insecurity':'Insecurity',\
      				'D17_challenge_transpCost':'Transportation Costs','D17_challenge_distance':'Distance',\
      				'D17_challenge_notenough':'Not enough goods', 'D17_challenge_Other':'Other'}
				for i in dico:
					a=data[[i,'C2_Acresowned']].copy()
					a=a[a[i].isin(['Yes','No'])]
							
					a.columns=['Challenge','Acres Owned']
					a['Chall']=a['Challenge'].apply(lambda x: dico[i])
					df=df.append(a)
					#st.write(df)
					fig = px.box(df, x="Chall", y="Acres Owned",color='Challenge')
					fig.update_layout(barmode='relative',xaxis={'title':'Challenges for accessing market'},\
										yaxis_title='Acres owned',width=800,height=450)
					if k%2==0:
						col1, col2, col3 = st.columns([4,1,4])
							
						col1.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
						col1.plotly_chart(fig,use_container_width=True)
					else:
							
						col3.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
						col3.plotly_chart(fig,use_container_width=True)
						
					k+=1
						
				else:
						
					df=data[[correlation,var]].copy()
					
					if var in continues:
						if correlation in continues:
							fig = px.scatter(df, x=var, y=correlation)
							fig.update_layout(xaxis={\
							'title':questions[var]},yaxis_title=questions[correlation],width=1500,height=800)
						else:
							fig = px.box(df, x=correlation, y=var,points='all')
							fig.update_traces(marker_color='green')
							fig.update_layout(barmode='relative', \
        	      	  				xaxis={'title':questions[correlation]},\
        	      	 				yaxis_title=questions[var],width=800,height=450)
							
						if k%2==0:
							col1, col2, col3 = st.columns([4,1,4])
								
							col1.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
							col1.plotly_chart(fig,use_container_width=True)
						else:
								
							col3.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
							col3.plotly_chart(fig,use_container_width=True)
						k+=1
							
					else:
						if correlation in continues:
							fig = px.box(df, x=var, y=correlation,points='all')
							fig.update_traces(marker_color='green')
							fig.update_layout(barmode='relative',xaxis={'title':questions[var]},\
									yaxis_title=questions[correlation],width=800,height=450)
							if k%2==0:
								col1, col2, col3 = st.columns([4,1,4])
									
								col1.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
								col1.plotly_chart(fig,use_container_width=True)
							else:
									
								col3.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
								col3.plotly_chart(fig,use_container_width=True)
							k+=1
                 				
						else:
							agg=df[[correlation,var]].groupby(by=[var,correlation]).aggregate({var:'count'}).unstack()
							x=[i for i in agg.index]
							fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
							for i in range(len(agg.columns)-1):
    								fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
							fig.update_layout(barmode='relative', \
                	  				xaxis={'title':questions[var]},\
       	        	  			yaxis={'title':'Persons'}, legend_title_text=None)
							
							agg=df[[correlation,var]].groupby(by=[var,correlation]).aggregate({var:'count'}).unstack()
							agg=agg.T/agg.T.sum()
							agg=agg.T*100
							x=[i for i in agg.index]
							fig2 = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
							for i in range(len(agg.columns)-1):
    								fig2.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
							fig2.update_layout(barmode='relative', \
        	        	  			xaxis={'title':questions[var]},\
        	        	 				yaxis={'title':'Pourcentages'}, legend_title_text=None)
								
							st.write(correl[(correl['Variables']==var) & (correl['Correl']==correlation)]['Description'].iloc[0])
								
							col1, col2, col3 = st.columns([4,1,4])
								
							col1.plotly_chart(fig,use_container_width=True)
							col3.plotly_chart(fig2,use_container_width=True)
							k=0
						
						
	elif topic=='Display Wordclouds':
		title2.title('Wordclouds for open questions')
		df=data[[i for i in data.columns if 'text' in i]].copy()
		#st.write(df)
		feature=st.sidebar.selectbox('Select the question for which you would like to visualize wordclouds of answers',[questions[i] for i in df.columns])	
		var=[i for i in questions if questions[i]==feature][0]
		
		col1, col3 = st.columns([6,3])
		col1.title('Wordcloud from question:')
		col1.title(feature)
				
		x, y = np.ogrid[:300, :300]
		mask = ((x - 150)) ** 2 + ((y - 150)/1.4) ** 2 > 130 ** 2
		mask = 255 * mask.astype(int)
		corpus=' '.join(df[var].apply(lambda x:'' if x=='0' else x))
		corpus=re.sub('[^A-Za-z ]',' ', corpus)
		corpus=re.sub('\s+',' ', corpus)
		corpus=corpus.lower()
		
		col3.title('')
		col3.title('')
		col3.title('')
		sw=col3.multiselect('Select words you would like to remove from the wordcloud', [i[0] for i in Counter(corpus.split(' ')).most_common()[:20] if i[0] not in STOPWORDS])
		
		if corpus==' ':
    			corpus='No_response'
		else:
			corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
		wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
		
		wc.generate(corpus)
		
		col1.image(wc.to_array(),width=400)	
		
		if col1.checkbox('Would you like to filter Wordcloud according to other questions'):
			
			st.markdown("""---""")
			
			feature2=st.selectbox('Select one question to filter the wordcloud (Select one of the last ones for checking some new tools)',[questions[i] for i in data.columns if \
			i!='FCS Score' and (i in continues or len(data[i].unique())<=8)])
			var2=[i for i in questions if questions[i]==feature2][0]
			
			if var2 in continues:
				threshold=st.slider('Select the threshold', min_value=data[var2].fillna(0).min(),max_value=data[var2].fillna(0).max())
				subcol1,subcol2=st.columns([2,2])	
				
				corpus1=' '.join(data[data[var2]<threshold][var].apply(lambda x:'' if x=='0' else x))
				corpus1=re.sub('[^A-Za-z ]',' ', corpus1)
				corpus1=re.sub('\s+',' ', corpus1)
				corpus1=corpus1.lower()
				if corpus1==' 'or corpus1=='':
    					corpus1='No_response'
				else:
					corpus1=' '.join([i for i in corpus.split(' ') if i not in sw])
				wc1 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc1.generate(corpus1)
				corpus2=' '.join(data[data[var2]>=threshold][var].apply(lambda x:'' if x=='0' else x))
				corpus2=re.sub('[^A-Za-z ]',' ', corpus2)
				corpus2=re.sub('\s+',' ', corpus2)
				corpus2=corpus2.lower()
				if corpus2==' ' or corpus2=='':
    					corpus2='No_response'
				else:
					corpus2=' '.join([i for i in corpus.split(' ') if i not in sw])
				wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc2.generate(corpus2)
				subcol1.write('Response under the threshold')
				subcol1.image(wc1.to_array(),width=400)
				subcol2.write('Response over the threshold')
				subcol2.image(wc2.to_array(),width=400)
			else:
				subcol1,subcol2=st.columns([2,2])
				L=data[var2].unique()
				
				corpus1=corpus2=corpus3=corpus4=corpus5=corpus6=corpus7=corpus8=''
				Corpuses=[corpus1,corpus2,corpus3,corpus4,corpus5,corpus6,corpus7,corpus8]
				
				
				for i in range(len(L)):
					Corpuses[i]=' '.join(data[data[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
					Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
					Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
					Corpuses[i]=Corpuses[i].lower()
					if Corpuses[i]==' ':
    						Corpuses[i]='No_response'
					else:
						Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
					wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc2.generate(Corpuses[i])
					if i%2==0:
						subcol1.write('Response : '+str(L[i])+' '+str(len(data[data[var2]==L[i]]))+' '+'repondent')
						subcol1.image(wc2.to_array(),width=400)
					else:
						subcol2.write('Response : '+str(L[i])+' '+str(len(data[data[var2]==L[i]]))+' '+'repondent')
						subcol2.image(wc2.to_array(),width=400)
			
	elif topic=='Display Sankey Graphs':
	
		title2.title('Visuals for questions related to cultures (questions C3 to C17)')
		st.title('')
				
			
		sankey=[i for i in data.columns if i[0]=='C' and 'C1_' not in i and 'C2_' not in i and i!='Clan']
		sankeyseeds=sankey[:65]
		sank=data[sankeyseeds]
		bean=sank[[i for i in sank.columns if 'Bean' in i]].copy()
		sesame=sank[[i for i in sank.columns if 'Sesame' in i]].copy()
		cowpea=sank[[i for i in sank.columns if 'Cowpea' in i]].copy()
		maize=sank[[i for i in sank.columns if 'Maize' in i]].copy()
		other=sank[[i for i in sank.columns if 'Other' in i]].copy()
		colonnes=['Seeds Planted','Type of seeds','Origin of seeds','Area cultivated','Did you have enough seed',\
          'Did you face pest attack','Area affected','Have you done pest management','Origin of fertilizer',\
          'Fertilizer from Wardi','Applied good practices','Used irrigation','Area irrigated']
		for i in [bean,sesame,cowpea,maize,other]:
    			i.columns=colonnes
		bean=bean[bean['Seeds Planted']=='Yes']
		sesame=sesame[sesame['Seeds Planted']=='Yes']
		cowpea=cowpea[cowpea['Seeds Planted']=='Yes']
		maize=maize[maize['Seeds Planted']=='Yes']
		other=other[other['Seeds Planted']=='Yes']
		
		bean['Seeds Planted']=bean['Seeds Planted'].apply(lambda x: 'Beans')
		sesame['Seeds Planted']=sesame['Seeds Planted'].apply(lambda x: 'Sesame')
		cowpea['Seeds Planted']=cowpea['Seeds Planted'].apply(lambda x: 'Cowpeas')
		maize['Seeds Planted']=maize['Seeds Planted'].apply(lambda x: 'Maize')
		other['Seeds Planted']=other['Seeds Planted'].apply(lambda x: 'Other')
		
		sank=pd.DataFrame(columns=colonnes)
		for i in [bean,sesame,cowpea,maize,other]:
		    sank=sank.append(i)
		sank['ones']=np.ones(len(sank))
		
		
		
		
		st.title('Some examples')
		
		st.markdown("""---""")
		st.write('Seeds planted - Origin of Seeds - Type of Seeds - Area Cultivated - Did you have enough seeds?')
		fig=sankey_graph(sank,['Seeds Planted','Origin of seeds','Type of seeds','Area cultivated','Did you have enough seed'],height=600,width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig,use_container_width=True)
		
		st.markdown("""---""")
		st.write('Origin of fertilizer - Did you face pest attack - Applied good practices - Seeds Planted')
		fig1=sankey_graph(sank,['Origin of fertilizer','Did you face pest attack','Applied good practices','Seeds Planted'],height=600,width=1500)
		fig1.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig1,use_container_width=True)
		
		st.markdown("""---""")
		st.write('Area Cultivated - Type of Seeds - Did you face pest attack - Area affected')
		fig2=sankey_graph(sank,['Area cultivated','Type of seeds','Did you face pest attack','Area affected'],height=600,width=1500)
		fig2.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig2,use_container_width=True)
		
		if st.checkbox('Design my own Sankey Graph'):
			
			st.markdown("""---""")
			feats=st.multiselect('Select features you want to see in the order you want them to appear', colonnes)
			
			if len(feats)>=2:
				st.write(' - '.join(feats))
				fig3=sankey_graph(sank,feats,height=600,width=1500)
				fig3.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
				st.plotly_chart(fig3,use_container_width=True)
		
		
		
			
	
	
	

    
 
if __name__== '__main__':
    main()




    
