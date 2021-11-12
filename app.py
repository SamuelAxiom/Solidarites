import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
#import variables


#Fonction de graph
@st.cache		
def sunb(q1,i,main_question,second_question,df2):
	df2['ones']=np.ones(len(df2))
	fig = px.sunburst(df2.fillna(''), path=[q1,i], values='ones')
	fig.update_layout(title_text=main_question + ' et ' +second_question,font=dict(size=20))
	return fig

@st.cache	
def count(q1,q2,main_question,second_question,df):
	agg=df[[q2,q1]].groupby(by=[q1,q2]).aggregate({q1:'count'}).unstack()
	x=[i for i in agg.index]
	fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
	for i in range(len(agg.columns)-1):
    		fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
	fig.update_layout(barmode='relative', \
                  xaxis={'title':main_question},\
                  yaxis={'title':'Persons'}, legend_title_text=None)
	return fig

@st.cache
def count2(abscisse,ordonnée,data):
    
    agg=data[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg2=agg.T/agg.T.sum()
    agg2=agg2.T.round(2)*100
    x=agg.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        
        for i in range(len(labels)):
            if labels[i] in data[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':'<b>'+abscisse+'<b>','title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.1,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    fig.update_layout(title_text='test')
    
    return fig

@st.cache
def pourcent(q1,q2,main_question,second_question,df):
	agg=df[[q2,q1]].groupby(by=[q1,q2]).aggregate({q1:'count'}).unstack()
	agg=agg.T/agg.T.sum()
	agg=agg.T*100
	x=[i for i in agg.index]
	fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
	for i in range(len(agg.columns)-1):
    		fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
	fig.update_layout(barmode='relative', \
                  xaxis={'title':main_question},\
                  yaxis={'title':'Pourcentages'}, legend_title_text=None)
	return fig

@st.cache
def pourcent2(abscisse,ordonnée,data):
    
    agg2=data[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg=agg2.T/agg2.T.sum()
    agg=agg.T.round(2)*100
    x=agg2.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        
        for i in range(len(labels)):
            if labels[i] in data[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} people",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':'<b>'+abscisse+'<b>','title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.1,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    fig.update_layout(title_text='test')
    
    return fig

@st.cache
def box(cont,cat,cont_question,noncont_question,df):
	fig = px.box(df, x=cat, y=cont,points='all')
	fig.update_traces(marker_color='green')
	fig.update_layout(barmode='relative', \
                  xaxis={'title':noncont_question},\
                 yaxis_title=cont_question)
	return fig

@st.cache	
def scatter(q1,q2,main_question,second_question,df):
	fig = px.scatter(df, x=q1, y=q2)
	fig.update_layout(xaxis={'title':main_question},yaxis_title=second_question)

	return fig

@st.cache
def selectdf(q1):
	q2_list=correl[q1].tolist()+[q1]
	features=[]
	for feat in q2_list:
		if feat in cat_cols:
			features+=[k for k in data.columns if feat in k]
		else:
			features.append(feat)
	return data[features]

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1,3,1])
col1.write("")
col2.title('Sol Yemen')
col3.write("")

st.sidebar.title('Questions Selector')


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	correl=pd.read_csv('correlations.csv',sep='\t')
	questions=pd.read_csv('questions.csv',index_col=None,sep='\t')
	codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
	continues=pickle.load( open( "cont_feat.p", "rb" ) )
	cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
	dummy_cols=pickle.load( open( "dummy.p", "rb" ) )	
	questions.set_index('Idquest',inplace=True)
	
	return data,correl,questions[[i for i in questions.columns if 'Unnamed' not in i]],codes,continues,cat_cols,dummy_cols


#Récup des data avec les variables nouvelles
data,correl,questions,codes,continues,cat_cols,dummy_cols=load_data()

######################faudra surement aussi récupérer d'autres trucs sur les types de données des int_cat et int_cat_desc############################

st.write(correl)
st.write(codes)

def main():
	graphs=pd.read_csv('graphs.csv',index_col=0,sep='\t')
	st.write(graphs)
	q1 = st.sidebar.selectbox('Main question:', [None]+[i for i in correl.columns if i not in graphs['Question'].unique()])
	if q1 != None:
		df=selectdf(q1)
		q2_list=correl[q1]	
		# TRAITEMENT PARTICULIER DES DONNÉES DE CAT_COLS
		if q1 in cat_cols:
			
			quests1=[i for i in df.columns if q1+' ' in i]
			
			if q1=='HH':
				fig = go.Figure(data=[go.Box(y=df[i], boxpoints='all', jitter=0.3, pointpos=-1.8,name=i[3:] )\
					 for i in quests1])
			else:
				cats=[' '.join(i.split(' ')[1:]) for i in quests1]
				fig = px.bar(x=cats, y=df[quests1].sum(), labels={'y':'People'})	
			col1, col2 = st.columns([1,3])
			col1.write('Donnée multiple')
			col1.write(q1)
			col1.write(quests1)
			col2.plotly_chart(fig)
			
			
		else:
			st.write(questions.loc['Question'][q1])
			quests1=[q1]
			fig=px.histogram(df, x=q1,color_discrete_sequence=['green'])
			st.plotly_chart(fig)
			
#Visualisation des 6 paramètres le splus importants pour la prédiction			
			
		if st.sidebar.checkbox('Do you want to generate graphs with other potential correlated questions?'):	
			keeps={}
			
			for q2 in q2_list:
				
				if q2 in cat_cols:
					quests2=[k for k in df.columns if q2 in k]
					st.subheader(q2+': '+', '.join(quests2))
					
				else:
					st.subheader(q2+': '+questions.loc['Question'][q2])
					quests2=[q2]
				quest=quests1+quests2
				
				
				#st.dataframe(df2)								
				
				# On commence par le cas HH
				
				if q1=='HH ' or q2=='HH ':
					if q1=='HH ':
						hh,question=q1,q2
					else:
						hh,question=q2,q1
					
					col1, col3,col5 = st.columns([1,1,1])
					HH=['HH women','HH men','HH total','HH girls','HH boys','HH children']
					people=['women','men','total','girls','boys','children below 5']
					
					if question in cat_cols:
						df3=pd.DataFrame(columns=HH+[question])
						#st.dataframe(df3)	
						catcols=[j for j in df if question in j]
						cats=[' '.join(i.split(' ')[1:]) for i in catcols]
						#st.dataframe(df2)
						#st.write(catcols,cat)
						for i in range(len(catcols)):
							ds=df[HH+[catcols[i]]].copy()
							ds=ds[ds[catcols[i]]==1]
							ds[catcols[i]]=ds[catcols[i]].apply(lambda x: cats[i])
							ds.columns=HH+[question]
							df3=df3.append(ds)
						#st.dataframe(df3)
						col1.plotly_chart(box(HH[0],question,people[0],question,df3),use_container_width=True)
						col3.plotly_chart(box(HH[1],question,people[1],question,df3),use_container_width=True)
						col5.plotly_chart(box(HH[2],question,people[2],question,df3),use_container_width=True)
						col1.plotly_chart(box(HH[3],question,people[3],question,df3),use_container_width=True)
						col3.plotly_chart(box(HH[4],question,people[4],question,df3),use_container_width=True)
						col5.plotly_chart(box(HH[5],question,people[5],question,df3),use_container_width=True)
						keeps[q2]=keep(q1,q2)
							
							
					elif question in continues:
						col1.plotly_chart(box(question,HH[0],question,people[0],df),use_container_width=True)
						col3.plotly_chart(box(question,HH[1],question,people[1],df),use_container_width=True)
						col5.plotly_chart(box(question,HH[2],question,people[2],df),use_container_width=True)
						col1.plotly_chart(box(question,HH[3],question,people[3],df),use_container_width=True)
						col3.plotly_chart(box(question,HH[4],question,people[4],df),use_container_width=True)
						col5.plotly_chart(box(question,HH[5],question,people[5],df),use_container_width=True)
						keeps[q2]=keep(q1,q2)
					
					else:
						col1.plotly_chart(box(HH[0],question,people[0],question,df),use_container_width=True)
						col3.plotly_chart(box(HH[1],question,people[1],question,df),use_container_width=True)
						col5.plotly_chart(box(HH[2],question,people[2],question,df),use_container_width=True)
						col1.plotly_chart(box(HH[3],question,people[3],question,df),use_container_width=True)
						col3.plotly_chart(box(HH[4],question,people[4],question,df),use_container_width=True)
						col5.plotly_chart(box(HH[5],question,people[5],question,df),use_container_width=True)
						keeps[q2]=keep(q1,q2)
										
				# On regarde maintenant si les deux sont des données catégorielles
					
				elif q1 in cat_cols and q2 in cat_cols:
					
					df3=pd.DataFrame(columns=[q1,q2])
					catq1=[' '.join(i.split(' ')[1:]) for i in quests1]
					catq2=[' '.join(i.split(' ')[1:]) for i in quests2]

					for i in range(len(quests1)):
						for j in range(len(quests2)):       
							ds=df[[quests1[i],quests1[j]]].copy()
							ds=ds[ds[quests1[i]]==1]
							ds=ds[ds[quests2[j]]==1]      
							ds[quests1[i]]=ds[quests1[i]].apply(lambda x: catq1[i])
							ds[quests2[j]]=ds[quests2[j]].apply(lambda x: catq2[j])
							ds.columns=[q1,q2]
							df3=df3.append(ds)
								
					col1, col3 = st.columns([1,1])
					df3['ones']=np.ones(len(df3))
					figs1 = px.sunburst(df3.fillna('None'), path=[q1,q2], values='ones')
					figs1.update_layout(title_text=q1+ ' et ' +q2,font=dict(size=20))
					figs2 = px.sunburst(df3.fillna('None'), path=[q2,q1], values='ones')
					figs2.update_layout(title_text=q1+ ' et ' +q2,font=dict(size=20))
					col1.plotly_chart(figs1,use_container_width=True)
					col3.plotly_chart(figs2,use_container_width=True)
					keeps[q2]=keep(q1,q2)
						
				# On regarde maintenant si une des deux est catégorielle	
					
				elif q1 in cat_cols or q2 in cat_cols:
					
					if q1 in cat_cols:
						cat,autre=q1,q2
						catcols=quests1
					else:
						cat,autre=q2,q1
						catcols=quests2							
					
					df3=pd.DataFrame(columns=[q1,q2])
					
					cats=[' '.join(i.split(' ')[1:]) for i in catcols]
					
					for i in range(len(catcols)):
						ds=df[[catcols[i],autre]].copy()
						ds=ds[ds[catcols[i]]==1]
						ds[catcols[i]]=ds[catcols[i]].apply(lambda x: cats[i])
						ds.columns=[q1,q2]
						df3=df3.append(ds)				
					
					if autre in dummy_cols:
						
						col1, col3 = st.columns([1,1])
						df3['ones']=np.ones(len(df3))
						figs1 = px.sunburst(df3.fillna('None'), path=[q1,q2], values='ones')
						figs1.update_layout(title_text=q1+ ' et ' +q2,font=dict(size=20))
						figs2 = px.sunburst(df3.fillna('None'), path=[q2,q1], values='ones')
						figs2.update_layout(title_text=q1+ ' et ' +q2,font=dict(size=20))
						col1.plotly_chart(figs1,use_container_width=True)
						col3.plotly_chart(figs2,use_container_width=True)
						keeps[q2]=keep(q1,q2)
							
					elif autre in continues:
						st.plotly_chart(box(autre,cat,autre,cat,df3))
						keeps[q2]=keep(q1,q2)
					
					else:
						col1, col3 = st.columns([4,4])
						df3['ones']=np.ones(len(df3))
						figs1 = px.sunburst(df3.fillna('None'), path=[q1,q2], values='ones')
						figs1.update_layout(title_text=q1+ ' et ' +q2,font=dict(size=20))
						figs2 = px.sunburst(df3.fillna('None'), path=[q2,q1], values='ones')
						figs2.update_layout(title_text=q1+ ' et ' +q2,font=dict(size=20))
						col1.plotly_chart(figs1)
						col3.plotly_chart(figs2)
						keeps[q2]=keep(q1,q2)
				
					
				elif q1 in continues:
					if q2 in continues:
						st.plotly_chart(scatter(q1,q2,q1,q2,df),use_container_width=True)
						keeps[q2]=keep(q1,q2)
					else:
						st.plotly_chart(box(q1,q2,q1,q2,df),use_container_width=True)
						keeps[q2]=keep(q1,q2)
											
				
				elif q2 in continues:
					st.plotly_chart(box(q2,q1,q2,q1,df),use_container_width=True)
					keeps[q2]=keep(q1,q2)
				
				elif q1 in dummy_cols:
					if q2 in dummy_cols:
						col1, col3 = st.columns([4,4])
						col1.plotly_chart(sunb(q1,q2,q1,q2,df),use_container_width=True)
						col3.plotly_chart(sunb(q2,q1,q2,q1,df),use_container_width=True)
						keeps[q2]=keep(q1,q2)
					else:
						if len(df[q1].unique())>4*len(df[q1].unique()):
							col1, col3 = st.columns([5,5])
							col1.plotly_chart(count(q1,q2,q1,q2,df),use_container_width=True)
							col3.plotly_chart(pourcent(q1,q2,q1,q2,df),use_container_width=True)
							col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
							col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
							keeps[q2]=keep(q1,q2,df)
						else:
							col1, col3 = st.columns([5,5])
							col1.plotly_chart(count(q2,q1,q2,q1,df),use_container_width=True)
							col3.plotly_chart(pourcent(q2,q1,q2,q1,df),use_container_width=True)
							col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
							col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
							keeps[q2]=keep(q1,q2)
				else:
					st.write('here')
					col1, col3 = st.columns([5,5])
					col1.plotly_chart(count(q1,q2,q1,q2,df),use_container_width=True)
					col3.plotly_chart(pourcent(q1,q2,q1,q2,df),use_container_width=True)
					col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
					col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
					col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
					col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
					keeps[q2]=keep(q1,q2)
			
			
			
			st.write(graphs)
			st.write(keeps)
			for item in keeps:
				if keeps[item]!={}:
					graphs=graphs.append(keeps[item],ignore_index=True)
			st.write(graphs)
			
			if st.button('Save'):
				graphs.to_csv('graphs.csv',sep='\t')				
		
		
		else:
			st.write('')
			st.write('')
			st.write('')
			col1, col2, col3 = st.columns([1,3,1])
			col2.text('Select something on the left side')
		
		
		
		


def keep(q1,q2):	
	if st.checkbox('Keep '+ q2):
		new_row={}
		new_row['Question']=q1
		if st.selectbox('Abscisse for '+q2, [q1,q2])==q1:
			new_row['variable_x']=q1
			new_row['variable_y']=q2
		else:
			new_row['variable_x']=q2
			new_row['variable_y']=q1

		new_row['description']=st.text_input('Descripe the relation with '+q2)
		new_row['xtitle']=st.text_input('Specify abscisses with '+q2)
		new_row['ytitle']=st.text_input('Specify y-axis name with '+q2)
		new_row['legendtitle']=st.text_input('Specify legend title with '+q2)
		new_row['title']=st.text_input('Specify the title of the graph with '+q2)
		
		return new_row			
		

if __name__== '__main__':
    main()
    
