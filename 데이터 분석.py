#%% 유동인구데이터(SK텔레콤) 전처리 및 분석
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
plt.style.use('ggplot')
import chart_studio.plotly as py
import cufflinks as cf
import plotly.graph_objects as go


import matplotlib.font_manager as fm
path = 'C:\Windows\Fonts\malgunbd.ttf'
font_name = fm.FontProperties(fname=path).get_name()
plt.rc('font', family=font_name)
import warnings
warnings.filterwarnings(action='ignore')
pd.options.display.float_format = '{:.2f}'.format

#데이터 전처리
Data = pd.read_csv('Input/Flow_SK/Four_Region_FLOW_TIME.csv',index_col = [0])
Data = Data.rename(columns={'STD_YM' : "년월", 'STD_YMD':"년월일", 'HDONG_CD':"행정동코드", 'HDONG_NM':"행정동명칭",
                     'TMST_00' :"00시_유동인구", 'TMST_01' : "01시_유동인구",
       'TMST_02':"02시_유동인구", 'TMST_03':"03시_유동인구", 'TMST_04':"04시_유동인구",
                     'TMST_05':"05시_유동인구", 'TMST_06':"06시_유동인구", 'TMST_07':"07시_유동인구",
       'TMST_08':"08시_유동인구", 'TMST_09':"09시_유동인구", 'TMST_10':"10시_유동인구",
                     'TMST_11':"11시_유동인구", 'TMST_12':"12시_유동인구", 'TMST_13':"13시_유동인구",
       'TMST_14':"14시_유동인구", 'TMST_15':"15시_유동인구", 'TMST_16':"16시_유동인구", 'TMST_17':"17시_유동인구",
                     'TMST_18':"18시_유동인구", 'TMST_19':"19시_유동인구",
       'TMST_20':"20시_유동인구", 'TMST_21':"21시_유동인구", 'TMST_22':"22시_유동인구", 'TMST_23':"23시_유동인구"})
Data['연도'] = Data.년월일.apply(lambda x : str(x)[:4])
Data['월'] = Data.년월일.apply(lambda x : str(x)[4:6])
Data['일'] = Data.년월일.apply(lambda x : str(x)[6:])
Data['행정동_head'] = Data.행정동명칭.apply(lambda x : re.compile('[가-힣]+').findall(x)[0])
Data['행정동_tail'] = Data.행정동명칭.apply(lambda x : re.compile('[0-9동]+').findall(x)[0])
Data = Data.reindex(columns=['연도', '월', '일', '행정동_head',
       '행정동_tail','년월', '년월일', '행정동코드', '행정동명칭', '00시_유동인구', '01시_유동인구', '02시_유동인구',
       '03시_유동인구', '04시_유동인구', '05시_유동인구', '06시_유동인구', '07시_유동인구', '08시_유동인구',
       '09시_유동인구', '10시_유동인구', '11시_유동인구', '12시_유동인구', '13시_유동인구', '14시_유동인구',
       '15시_유동인구', '16시_유동인구', '17시_유동인구', '18시_유동인구', '19시_유동인구', '20시_유동인구',
       '21시_유동인구', '22시_유동인구', '23시_유동인구'])
Data = Data.drop(['년월','년월일','행정동명칭','행정동코드'],axis = 1)
# 정제 후 데이터 따로 저장
Data.to_csv('Input/Flow_SK/4개지역_시간별_유동인구(정제후).csv')
Data_head = dict(list(Data.groupby(['행정동_head'])))
# 전체 지역 표시
location_list = list(Data_head.keys())

#데이터 시각화
#고산동 기준, 연도와 월 기준으로 그래프 그림
# year과 month는 string형태로 입력
def Check_data(data,location,year,month):
    test = data[location][(data[location].연도 == year) & (Data_head[location].월 ==month)]
    x_list = list(test.columns[5:])
    x_list = list(pd.Series(x_list).str[:3].values)
    
    Dong_list = list(np.zeros(len(test.행정동_tail.value_counts().index)))
    test2 = test.groupby(['연도','월','행정동_tail']).sum().T
    
    for i in range(len(Dong_list)):
        Dong_list[i] = test2.iloc[:,i]
    
    # 그래프
    fig = go.Figure()
    for i in range(len(Dong_list)):
        fig.add_trace(go.Scatter(x = x_list,y=  Dong_list[i],
                                mode = 'lines+markers',
                                name = '{}동'.format(i+1)))

    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                  xanchor='left', yanchor='bottom',
                                  text='{}년 {}월 {} 시간대별 유동인구'.format(year,month,location),
                                  font=dict(family='Arial',
                                            size=30,
                                            color='rgb(37,37,37)'),
                                  showarrow=False))
    fig.update_layout(annotations =annotations)
    fig.show()

year_list = ['2019','2020']
month_list = ['02','03','04','05'] 
for year in year_list:
    for month in month_list:
        Check_data(Data_head,'공릉',year,month)

#다른 동 적용
for location in location_list[1:4]:
    for year in year_list:
        for month in month_list:
            Check_data(Data_head,location,year,month)
#%% 카드매출데이터(신한카드) 전처리 및 분석
## 신한은행 - 내국인 csv파일 만들기
import pandas as pd
import copy

file_for = pd.read_csv('C:\\Users\\T919\\Desktop\\2020빅콘테스트 문제데이터(혁신아이디어분야)\\02_카드매출데이터(신한카드)\\CARD_SPENDING_FOREIGNER\\CARD_SPENDING_FOREIGNER.txt',sep='\t', encoding='CP949')

card_spending_foreigner = pd.DataFrame(file_for)
# card_spending_foreigner.to_csv('C:\\Users\\T919\\Desktop\\card_spending_foreigner.csv',header=True,index=False, encoding='CP949')
card_spending_foreigner_copy = copy.deepcopy(card_spending_foreigner)

file_res = pd.read_csv('C:\\Users\\T919\\Desktop\\2020빅콘테스트 문제데이터(혁신아이디어분야)\\02_카드매출데이터(신한카드)\\CARD_SPENDING_RESIDENT\\CARD_SPENDING_RESIDENT.txt',sep='\t', encoding='CP949')
card_spending_resident = pd.DataFrame(file_res)

card_spending_resident_copy = copy.deepcopy(card_spending_resident)
card_spending_resident_copy_1 = copy.deepcopy(card_spending_resident)


dong_gu = pd.ExcelFile('C:\\Users\\T919\\Desktop\\2020빅콘테스트 문제데이터 정리 후\\02_카드매출데이터(신한카드)\\02_혁신아이디어분야_카드소비데이터(신한카드)_데이터정의서.xlsx').parse(1).iloc[:,1:]

dong_gu_1 = copy.deepcopy(dong_gu).iloc[1:,:]
dong_gu_1.columns = dong_gu.iloc[0,:]

cd = pd.DataFrame(dong_gu_1.iloc[:,0])
cd['동코드'] = dong_cd

    
gu_dic = {}

gu_cd = list(dong_gu_1.iloc[:,0])
gu_name = list(dong_gu_1["구명"])
    
for j in range(len(gu_cd)):
    gu_dic[gu_cd[j]] = gu_name[j]    
    
gu_code = list(set(list(cd['구코드'])))    
    
    
for i in gu_code:
    card_spending_resident_copy.loc[card_spending_resident_copy.GU_CD==i,'GU_CD'] = gu_dic[i]

card_spending_resident_copy = card_spending_resident_copy

for i in gu_code:
    p = dong_gu_1[dong_gu_1['구명'] == gu_dic[i]]
    for j in range(p.shape[0]):
        card_spending_resident_copy.loc[card_spending_resident_copy.DONG_CD == int(p.iloc[j,1]),'DONG_CD'] = p.iloc[j,3]

up_jong = pd.ExcelFile('C:\\Users\\T919\\Desktop\\2020빅콘테스트 문제데이터 정리 후\\02_카드매출데이터(신한카드)\\02_혁신아이디어분야_카드소비데이터(신한카드)_데이터정의서.xlsx').parse(2).iloc[:,1:]

large_up_jong = list(up_jong.iloc[1:,0].dropna())

classified_up_jong = {}

for i in range(len(large_up_jong)):
    key = int(large_up_jong[i].strip("()").split("(")[1])
    value = large_up_jong[i].strip("()").split("(")[0]
    classified_up_jong[key] = value

list_classified_up_jong = list(classified_up_jong.keys())

for i in list_classified_up_jong:
    card_spending_resident_copy.loc[card_spending_resident_copy.MCT_CAT_CD == i, 'MCT_CAT_CD'] = classified_up_jong[i]

city = card_spending_resident_copy.GU_CD.str.split(' ').str[0].values.reshape(-1,1)

left = card_spending_resident_copy.iloc[:,:1].values
right = card_spending_resident_copy.iloc[:,1:].values

df = np.append(left,city,axis=1)
df = np.append(df,right,axis=1)

card_spending_resident_copy_1.columns

columns_list = ['기준일자', '시', '시-구', '동','업종','성별','나이코드','이용건수(건)','이용금액(천원)']


df = pd.DataFrame(df, columns = columns_list)

# city.str.split(' ').str[0]


df.to_csv('C:\\Users\\T919\\Desktop\\card_spending_resident.csv',header=True,index=False, encoding='CP949')

## 신한은행-외국인 csv파일 만들기
import pandas as pd
import copy

file_for = pd.read_csv('C:\\Users\\T919\\Desktop\\2020빅콘테스트 문제데이터(혁신아이디어분야)\\02_카드매출데이터(신한카드)\\CARD_SPENDING_FOREIGNER\\CARD_SPENDING_FOREIGNER.txt',sep='\t', encoding='CP949')

card_spending_foreigner = pd.DataFrame(file_for)
# card_spending_foreigner.to_csv('C:\\Users\\T919\\Desktop\\card_spending_foreigner.csv',header=True,index=False, encoding='CP949')
card_spending_foreigner_copy = copy.deepcopy(card_spending_foreigner)

file_res = pd.read_csv('C:\\Users\\T919\\Desktop\\2020빅콘테스트 문제데이터(혁신아이디어분야)\\02_카드매출데이터(신한카드)\\CARD_SPENDING_RESIDENT\\CARD_SPENDING_RESIDENT.txt',sep='\t', encoding='CP949')
card_spending_resident = pd.DataFrame(file_res)

card_spending_resident_copy = copy.deepcopy(card_spending_resident)
card_spending_resident_copy_1 = copy.deepcopy(card_spending_resident)


dong_gu = pd.ExcelFile('C:\\Users\\T919\\Desktop\\2020빅콘테스트 문제데이터 정리 후\\02_카드매출데이터(신한카드)\\02_혁신아이디어분야_카드소비데이터(신한카드)_데이터정의서.xlsx').parse(1).iloc[:,1:]

dong_gu_1 = copy.deepcopy(dong_gu).iloc[1:,:]
dong_gu_1.columns = dong_gu.iloc[0,:]

cd = pd.DataFrame(dong_gu_1.iloc[:,0])
cd['동코드'] = dong_cd

    
gu_dic = {}

gu_cd = list(dong_gu_1.iloc[:,0])
gu_name = list(dong_gu_1["구명"])
    
for j in range(len(gu_cd)):
    gu_dic[gu_cd[j]] = gu_name[j]    
    
gu_code = list(set(list(cd['구코드'])))    
    
    
for i in gu_code:
    card_spending_resident_copy.loc[card_spending_resident_copy.GU_CD==i,'GU_CD'] = gu_dic[i]

card_spending_resident_copy = card_spending_resident_copy

for i in gu_code:
    p = dong_gu_1[dong_gu_1['구명'] == gu_dic[i]]
    for j in range(p.shape[0]):
        card_spending_resident_copy.loc[card_spending_resident_copy.DONG_CD == int(p.iloc[j,1]),'DONG_CD'] = p.iloc[j,3]




up_jong = pd.ExcelFile('C:\\Users\\T919\\Desktop\\2020빅콘테스트 문제데이터 정리 후\\02_카드매출데이터(신한카드)\\02_혁신아이디어분야_카드소비데이터(신한카드)_데이터정의서.xlsx').parse(2).iloc[:,1:]

large_up_jong = list(up_jong.iloc[1:,0].dropna())

classified_up_jong = {}

for i in range(len(large_up_jong)):
    key = int(large_up_jong[i].strip("()").split("(")[1])
    value = large_up_jong[i].strip("()").split("(")[0]
    classified_up_jong[key] = value

list_classified_up_jong = list(classified_up_jong.keys())

for i in list_classified_up_jong:
    card_spending_resident_copy.loc[card_spending_resident_copy.MCT_CAT_CD == i, 'MCT_CAT_CD'] = classified_up_jong[i]





city = card_spending_resident_copy.GU_CD.str.split(' ').str[0].values.reshape(-1,1)

left = card_spending_resident_copy.iloc[:,:1].values
right = card_spending_resident_copy.iloc[:,1:].values

df = np.append(left,city,axis=1)
df = np.append(df,right,axis=1)

card_spending_resident_copy_1.columns

columns_list = ['기준일자', '시', '시-구', '동','업종','성별','나이코드','이용건수(건)','이용금액(천원)']


df = pd.DataFrame(df, columns = columns_list)

# city.str.split(' ').str[0]


df.to_csv('C:\\Users\\T919\\Desktop\\card_spending_resident.csv',header=True,index=False, encoding='CP949')


card_spending_foreigner_copy

# csv 파일 작업 완료
# 내외국인 카드 이용건수 분석 작업 진행
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import table
import csv
import time
%matplotlib inline


import seaborn as sns
plt.style.use('ggplot')


import matplotlib.font_manager as fm
path = 'C:\Windows\Fonts\malgunbd.ttf'
font_name = fm.FontProperties(fname=path).get_name()
plt.rc('font', family=font_name)
import warnings
warnings.filterwarnings(action='ignore')
pd.options.display.float_format = '{:.2f}'.format
os. chdir('C:\\Users\\ie gram_08\\Desktop\\빅콘테스트\\물가')

# 내국인 csv파일 호출
resident=pd.read_csv('C:/workspace/Bigcon/card_spending_resident.csv',encoding='cp949')

resident['year']=resident.기준일자.apply(lambda x :str(x)[:4])
resident=resident.reindex(columns=['year','시','시-구', '동', '업종', '성별', '나이코드', '이용건수(건)', '이용금액(천원)'])

# 19년, 20년 지역별 카드 이용건수 파악_내국인 대상
resident1=pd.DataFrame(resident['이용건수(건)'].groupby([resident['year'],resident['업종']]).sum())
resident2=pd.DataFrame(resident['이용금액(천원)'].groupby([resident['year'],resident['업종']]).sum())
resident2.to_excel('금액.xlsx')

# 외국인 csv 파일 호출
foreigner=pd.read_csv('card_spending_foreigner.csv',encoding='cp949')
foreigner['year']=foreigner.기준일자.apply(lambda x :str(x)[:4])
foreigner=foreigner.reindex(columns=['year','시', '시-구', '동', '업종', '국적', '이용건수(건)', '이용금액(천원)'])

# 19년, 20년 업종별 카드 이용건수 분석_외국인 대상
foreigner1=pd.DataFrame(foreigner['이용건수(건)'].groupby([foreigner['year'],foreigner['업종']]).sum())
foreigner2=pd.DataFrame(foreigner['이용금액(천원)'].groupby([foreigner['year'],foreigner['업종']]).sum())

# 품목별 소비자 물가지수 시각화 작업 진행
# 파일 호출
consumer=pd.read_csv('품목별_소비자물가지수.csv',encoding='cp949')
consumer.set_index(consumer['품목별'],inplace=True)
consumer.drop(['시도별','품목별'],axis=1,inplace=True)
consumer=consumer.T

#물가지수 시각화
consumer.plot(kind='line',figsize=(12,10))

#%% SNS데이터(와이즈넛) 전처리 및 분석

import os
import numpy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
from collections import Counter
from wordcloud import WordCloud
warnings.filterwarnings('ignore')
os.chdir('C:/Users/user/Desktop/잡/Desktop/2020빅콘테스트 문제데이터(혁신아이디어분야)/03_SNS데이터(와이즈넛)')
# 폰트 설정
mpl.rc('font', family= 'Malgun Gothic')
# 유니코드에서  음수 부호설정
mpl.rc('axes', unicode_minus=False)

# 데이터 작업 실시
# 1. 데이터 호출
sns_data=pd.read_csv('2020 bigcontest data_wisenut.csv',sep='',encoding='utf-8')

# 2. 데이터 전처리 작업 실시
sns_data['SI']=sns_data['GU_NM(삭제)'].apply(lambda x : x[:2])
col=sns_data.columns
col=col.drop('SI')
col=col.insert(1,'SI')
sns_data=sns_data[col]
col=list(col)
col[-1]=col[-1][:19]
sns_data.columns=col

# 3. 워드클라우드 작업 실시
news_cnt=sns_data.iloc[:,[1,3,5,6,7,8,9,10,11,12,13]]
news_col=list(news_cnt.columns)
for i in range(3,len(news_col)):
    news_col[i]=news_col[i][9:]
news_cnt.columns=news_col

blog_cnt=sns_data.iloc[:,[1,3,5,14,15,16,17,18,19,20,21]]
blog_col=list(blog_cnt.columns)

for i in range(3,len(blog_col)):
    blog_col[i]=blog_col[i][9:]
blog_cnt.columns=blog_col

cafe_cnt=sns_data.iloc[:,[1,3,5,22,23,24,25,26,27,28,29]]
cafe_col=list(cafe_cnt.columns)
for i in range(3,len(cafe_col)):
    cafe_col[i]=cafe_col[i][9:]
cafe_cnt.columns=cafe_col

topic=sns_data.iloc[:,[1,3,5,30,31,32,33,34,35,36,37]] #월별 동별 SNS 단어 데이터
topic_col=list(topic.columns)
for i in range(3,len(topic_col)):
    topic_col[i]=topic_col[i][9:]
topic.columns=topic_col

topic_1=topic+','
topic_2=topic_1.groupby('GU_NM(삭제)').sum().drop(['SI','DONG_NM(삭제)'],axis=1)
topic_2['201902']=topic_2['201902']+topic_2['201903']+topic_2['201904']+topic_2['201905']
topic_2.drop(['201903','201904','201905'],inplace=True,axis=1)
topic_2['202002']=topic_2['202002']+topic_2['202003']+topic_2['202004']+topic_2['202005']
topic_2.drop(['202003','202004','202005'],inplace=True,axis=1)

a1=Counter(topic_2.iloc[0,0].split(',')[:-1])
a2=Counter(topic_2.iloc[1,0].split(',')[:-1])
a3=Counter(topic_2.iloc[2,0].split(',')[:-1])
a4=Counter(topic_2.iloc[3,0].split(',')[:-1])

b1=Counter(topic_2.iloc[0,1].split(',')[:-1])
b2=Counter(topic_2.iloc[1,1].split(',')[:-1])
b3=Counter(topic_2.iloc[2,1].split(',')[:-1])
b4=Counter(topic_2.iloc[3,1].split(',')[:-1])

wordcloud = WordCloud(font_path='C:/Windows/Fonts/맑은 고딕/malgun.ttf',background_color='white')

df=pd.DataFrame(a1.keys(),columns=['word'])
df_1=pd.DataFrame(a1.values(),columns=['count'])
df=pd.concat([df,df_1],axis=1)
count=list(zip(df['word'],df['count']))
wordcloud.generate_from_frequencies(dict(count)).to_image() #대구 수성구 2019

df=pd.DataFrame(a2.keys(),columns=['word'])
df_1=pd.DataFrame(a2.values(),columns=['count'])
df=pd.concat([df,df_1],axis=1)
count=list(zip(df['word'],df['count']))
wordcloud.generate_from_frequencies(dict(count)).to_image() #대구 중구 2019

df=pd.DataFrame(a3.keys(),columns=['word'])
df_1=pd.DataFrame(a3.values(),columns=['count'])
df=pd.concat([df,df_1],axis=1)
count=list(zip(df['word'],df['count']))
wordcloud.generate_from_frequencies(dict(count)).to_image() #서울 중구 2019

df=pd.DataFrame(a4.keys(),columns=['word'])
df_1=pd.DataFrame(a4.values(),columns=['count'])
df=pd.concat([df,df_1],axis=1)
count=list(zip(df['word'],df['count']))
wordcloud.generate_from_frequencies(dict(count)).to_image() #서울 노원구 2019

df=pd.DataFrame(b1.keys(),columns=['word'])
df_1=pd.DataFrame(b1.values(),columns=['count'])
df=pd.concat([df,df_1],axis=1)
count=list(zip(df['word'],df['count']))
wordcloud.generate_from_frequencies(dict(count)).to_image() #대구 수성구 2019

df=pd.DataFrame(b2.keys(),columns=['word'])
df_1=pd.DataFrame(b2.values(),columns=['count'])
df=pd.concat([df,df_1],axis=1)
count=list(zip(df['word'],df['count']))
wordcloud.generate_from_frequencies(dict(count)).to_image() #대구 중구 2019

df=pd.DataFrame(b3.keys(),columns=['word'])
df_1=pd.DataFrame(b3.values(),columns=['count'])
df=pd.concat([df,df_1],axis=1)
count=list(zip(df['word'],df['count']))
wordcloud.generate_from_frequencies(dict(count)).to_image() #서울 중구 2019

df=pd.DataFrame(b4.keys(),columns=['word'])
df_1=pd.DataFrame(b4.values(),columns=['count'])
df=pd.concat([df,df_1],axis=1)
count=list(zip(df['word'],df['count']))
wordcloud.generate_from_frequencies(dict(count)).to_image() #서울 노원구 2019

# 열 이름 변경 작업 진행
code=['숙박','레저업소','문화취미','의료기관','보건위생','요식업소']
positive_cnt=sns_data.iloc[:,[1,3,5]+[i for i in range(38,86)]]
positive_col=list(positive_cnt.columns)
for i in range(3,len(positive_col)):
    positive_col[i]=code[int(positive_col[i][2])-1]+'-'+positive_col[i][13:]
positive_cnt.columns=positive_col

negative_cnt=sns_data.iloc[:,[1,3,5]+[i for i in range(86,134)]]
negative_col=list(negative_cnt.columns)
for i in range(3,len(negative_col)):
    negative_col[i]=code[int(negative_col[i][2])-1]+'-'+negative_col[i][13:]
negative_cnt.columns=negative_col

# 월별 뉴스, 블로그, 카페 게시량_구 총합
fig,axes=plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12,7)
sns.barplot(data=news_cnt,ax=axes[0][0])
axes[0][0].set(title='뉴스 건수')

sns.barplot(data=blog_cnt,ax=axes[0][1])
axes[0][1].set(title='블로그 건수')

sns.barplot(data=cafe_cnt,ax=axes[1][0])
axes[1][0].set(title='카페 건수')

plt.show()

# 구별로 구분해놓은 그래프_"월별 뉴스, 블로그, 카페 게시량 총합"
fig,axes=plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(15,9)

news_cnt_1=news_cnt.iloc[:,1:].drop('DONG_NM(삭제)',axis=1)
news_cnt_1=news_cnt_1.groupby('GU_NM(삭제)').sum().T
news_cnt_1.plot(kind='bar',width=0.75,ax=axes[0][0])
plt.setp(axes[0][0].get_xticklabels(), rotation=0, ha='left')
axes[0][0].set_title('월별 뉴스 건수',size=20)

blog_cnt_1=blog_cnt.iloc[:,1:].drop('DONG_NM(삭제)',axis=1)
blog_cnt_1=blog_cnt_1.groupby('GU_NM(삭제)').sum().T
blog_cnt_1.plot(kind='bar',width=0.75,ax=axes[0][1])
plt.setp(axes[0][1].get_xticklabels(), rotation=0, ha='left')
axes[0][1].set_title('월별 블로그 건수',size=20)

cafe_cnt_1=cafe_cnt.iloc[:,1:].drop('DONG_NM(삭제)',axis=1)
cafe_cnt_1=cafe_cnt_1.groupby('GU_NM(삭제)').sum().T
cafe_cnt_1.plot(kind='bar',width=0.75,ax=axes[1][0])
plt.setp(axes[1][0].get_xticklabels(), rotation=0, ha='left')
axes[1][0].set_title('월별 카페 건수',size=20)

plt.tight_layout()
plt.show()

# "월별 뉴스, 블로그, 카페 게시량 총합"_Area그래프
fig,axes=plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(15,9)

news_cnt_1=news_cnt.iloc[:,1:].drop('DONG_NM(삭제)',axis=1)
news_cnt_1=news_cnt_1.groupby('GU_NM(삭제)').sum().T
news_cnt_1.plot(kind='area',ax=axes[0][0])
plt.setp(axes[0][0].get_xticklabels(), rotation=0, ha='left')
axes[0][0].set_title('날짜별 뉴스 건수',size=20)

blog_cnt_1=blog_cnt.iloc[:,1:].drop('DONG_NM(삭제)',axis=1)
blog_cnt_1=blog_cnt_1.groupby('GU_NM(삭제)').sum().T
blog_cnt_1.plot(kind='area',ax=axes[0][1])
plt.setp(axes[0][1].get_xticklabels(), rotation=0, ha='left')
axes[0][1].set_title('날짜별 블로그 건수',size=20)

cafe_cnt_1=cafe_cnt.iloc[:,1:].drop('DONG_NM(삭제)',axis=1)
cafe_cnt_1=cafe_cnt_1.groupby('GU_NM(삭제)').sum().T
cafe_cnt_1.plot(kind='area',ax=axes[1][0])
plt.setp(axes[1][0].get_xticklabels(), rotation=0, ha='left')
axes[1][0].set_title('날짜별 카페 건수',size=20)

plt.tight_layout()
plt.show()

# 전체 SNS에서 월별, 카테고리별 긍정 및 부정 게시량
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,8)
sns.barplot(data=pd.DataFrame(positive_cnt.sum()[3:]).T,ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=50, ha='right')
axes[0].set(title='긍정 게시량')

sns.barplot(data=pd.DataFrame(negative_cnt.sum()[3:]).T,ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set(title='부정 게시량')

plt.tight_layout()
plt.show()

# 201905 동별 의료기관 긍정 게시량
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,10)
positive_1=positive_cnt[['DONG_NM(삭제)','의료기관-201905']]
positive_1['DONG_NM(삭제)']=positive_cnt['GU_NM(삭제)']+' '+positive_1['DONG_NM(삭제)']
sns.barplot(data=positive_1,x='DONG_NM(삭제)',y='의료기관-201905',ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=50, ha='right')
axes[0].set(title='201905 동별 의료기관 긍정 게시량')

negative_1=negative_cnt[['DONG_NM(삭제)','의료기관-201905']]
negative_1['DONG_NM(삭제)']=negative_cnt['GU_NM(삭제)']+' '+negative_1['DONG_NM(삭제)']
sns.barplot(data=negative_1,x='DONG_NM(삭제)',y='의료기관-201905',ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set(title='201905 동별 의료기관 부정 게시량')

plt.tight_layout()
plt.show()

# 201905 동별 의료기관 부정 게시량
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,10)
positive_1=positive_cnt[['DONG_NM(삭제)','의료기관-202002']]
positive_1['DONG_NM(삭제)']=positive_cnt['GU_NM(삭제)']+' '+positive_1['DONG_NM(삭제)']
sns.barplot(data=positive_1,x='DONG_NM(삭제)',y='의료기관-202002',ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=50, ha='right')
axes[0].set(title='202002 동별 의료기관 긍정 게시량')

negative_1=negative_cnt[['DONG_NM(삭제)','의료기관-202002']]
negative_1['DONG_NM(삭제)']=negative_cnt['GU_NM(삭제)']+' '+negative_1['DONG_NM(삭제)']
sns.barplot(data=negative_1,x='DONG_NM(삭제)',y='의료기관-202002',ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set(title='202002 동별 의료기관 부정 게시량')

plt.tight_layout()
plt.show()

# 2020년 2월 의료기관 동별 긍정 게시량
fig,axes=plt.subplots(nrows=3,ncols=2)
fig.set_size_inches(15,9)

positive_2=positive_cnt.iloc[:,[1]+[i for i in range(3,11)]].groupby('GU_NM(삭제)').sum().T
positive_2.plot(kind='bar',width=0.75,ax=axes[0][0])
plt.setp(axes[0][0].get_xticklabels(), rotation=20, ha='right')
axes[0][0].set_title('월별 숙박 긍정 게시량',size=20)
axes[0][0].legend(loc='upper left', frameon=False,fontsize=9)

positive_3=positive_cnt.iloc[:,[1]+[i for i in range(11,19)]].groupby('GU_NM(삭제)').sum().T
positive_3.plot(kind='bar',width=0.75,ax=axes[0][1])
plt.setp(axes[0][1].get_xticklabels(), rotation=20, ha='right')
axes[0][1].set_title('월별 레저업소 긍정 게시량',size=20)
axes[0][1].legend(loc='upper left', frameon=False,fontsize=9)

positive_4=positive_cnt.iloc[:,[1]+[i for i in range(19,27)]].groupby('GU_NM(삭제)').sum().T
positive_4.plot(kind='bar',width=0.75,ax=axes[1][0])
plt.setp(axes[1][0].get_xticklabels(), rotation=20, ha='right')
axes[1][0].set_title('월별 문화취미 긍정 게시량',size=20)
axes[1][0].legend(loc='upper left', frameon=False,fontsize=9)

positive_5=positive_cnt.iloc[:,[1]+[i for i in range(27,35)]].groupby('GU_NM(삭제)').sum().T
positive_5.plot(kind='bar',width=0.75,ax=axes[1][1])
plt.setp(axes[1][1].get_xticklabels(), rotation=20, ha='right')
axes[1][1].set_title('월별 의료기관 긍정 게시량',size=20)
axes[1][1].legend(loc='upper left', frameon=False,fontsize=9)

positive_6=positive_cnt.iloc[:,[1]+[i for i in range(35,43)]].groupby('GU_NM(삭제)').sum().T
positive_6.plot(kind='bar',width=0.75,ax=axes[2][0])
plt.setp(axes[2][0].get_xticklabels(), rotation=20, ha='right')
axes[2][0].set_title('월별 보건위생 긍정 게시량',size=20)
axes[2][0].legend(loc='upper left', frameon=False,fontsize=9)

positive_7=positive_cnt.iloc[:,[1]+[i for i in range(43,51)]].groupby('GU_NM(삭제)').sum().T
positive_7.plot(kind='bar',width=0.75,ax=axes[2][1])
plt.setp(axes[2][1].get_xticklabels(), rotation=20, ha='right')
axes[2][1].set_title('월별 요식업소 긍정 게시량',size=20)
axes[2][1].legend(loc='upper right', frameon=False,fontsize=9)

plt.tight_layout()
plt.show()

# 2020년 2월 의료기관 동별 부정 게시량
fig,axes=plt.subplots(nrows=3,ncols=2)
fig.set_size_inches(15,9)

negative_2=negative_cnt.iloc[:,[1]+[i for i in range(3,11)]].groupby('GU_NM(삭제)').sum().T
negative_2.plot(kind='bar',width=0.75,ax=axes[0][0])
plt.setp(axes[0][0].get_xticklabels(), rotation=20, ha='right')
axes[0][0].set_title('월별 숙박 부정 게시량',size=20)
axes[0][0].legend(loc='upper left', frameon=False,fontsize=9)

negative_3=negative_cnt.iloc[:,[1]+[i for i in range(11,19)]].groupby('GU_NM(삭제)').sum().T
negative_3.plot(kind='bar',width=0.75,ax=axes[0][1])
plt.setp(axes[0][1].get_xticklabels(), rotation=20, ha='right')
axes[0][1].set_title('월별 레저업소 부정 게시량',size=20)
axes[0][1].legend(loc='upper left', frameon=False,fontsize=9)

negative_4=negative_cnt.iloc[:,[1]+[i for i in range(19,27)]].groupby('GU_NM(삭제)').sum().T
negative_4.plot(kind='bar',width=0.75,ax=axes[1][0])
plt.setp(axes[1][0].get_xticklabels(), rotation=20, ha='right')
axes[1][0].set_title('월별 문화취미 부정 게시량',size=20)
axes[1][0].legend(loc='upper left', frameon=False,fontsize=9)

negative_5=negative_cnt.iloc[:,[1]+[i for i in range(27,35)]].groupby('GU_NM(삭제)').sum().T
negative_5.plot(kind='bar',width=0.75,ax=axes[1][1])
plt.setp(axes[1][1].get_xticklabels(), rotation=20, ha='right')
axes[1][1].set_title('월별 의료기관 부정 게시량',size=20)
axes[1][1].legend(loc='upper left', frameon=False,fontsize=9)

negative_6=negative_cnt.iloc[:,[1]+[i for i in range(35,43)]].groupby('GU_NM(삭제)').sum().T
negative_6.plot(kind='bar',width=0.75,ax=axes[2][0])
plt.setp(axes[2][0].get_xticklabels(), rotation=20, ha='right')
axes[2][0].set_title('월별 보건위생 부정 게시량',size=20)
axes[2][0].legend(loc='upper left', frameon=False,fontsize=9)

negative_7=negative_cnt.iloc[:,[1]+[i for i in range(43,51)]].groupby('GU_NM(삭제)').sum().T
negative_7.plot(kind='bar',width=0.75,ax=axes[2][1])
plt.setp(axes[2][1].get_xticklabels(), rotation=20, ha='right')
axes[2][1].set_title('월별 요식업소 부정 게시량',size=20)
axes[2][1].legend(loc='upper right', frameon=False,fontsize=9)

plt.tight_layout()
plt.show()
#%% 유통데이터(GS리테일) 전처리 및 분석

import os
import numpy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')
os.chdir('C:/Users/user/Desktop/잡/Desktop/2020빅콘테스트 문제데이터(혁신아이디어분야)/04_유통데이터(GS리테일)')
# 폰트 설정
mpl.rc('font', family= 'Malgun Gothic')
# 유니코드에서  음수 부호설정
mpl.rc('axes', unicode_minus=False)

# 데이터 작업 시작
# 파일 호출
dong_data=pd.ExcelFile('04_혁신아이디어분야_유통데이터(GS리테일)_데이터정의서 및 문제 데이터.xlsx').parse(3)
cate_data=pd.ExcelFile('04_혁신아이디어분야_유통데이터(GS리테일)_데이터정의서 및 문제 데이터.xlsx').parse(4)
total_data=pd.ExcelFile('04_혁신아이디어분야_유통데이터(GS리테일)_데이터정의서 및 문제 데이터.xlsx').parse(5)

# 데이터 전처리 작업 진행
dong_data=dong_data.iloc[2:,:]
dong_col=dong_data.loc[2]
for i in range(3,len(dong_col)):
    dong_col[i]=int(dong_col[i])
dong_col=list(map(str,dong_col))
dong_data.columns=dong_col
dong_data=dong_data.iloc[1:,:]

#단위 변환
dong_data.iloc[:,3:]=dong_data.iloc[:,3:]*10000

#결측치 삭제
dong_data_1=dong_data.dropna().reset_index(drop=True)

# 구 기준으로 업종별 구분 작업 진행
cate_data.fillna(0,inplace=True)
for i in range(len(dong_data_1)):
    idx=cate_data[cate_data['ADMD']==dong_data_1.iloc[i,2]].index
    cate_data.iloc[idx,5:]=cate_data.iloc[idx,5:]*dong_data_1.iloc[i,3:]

# 지역 이름을 구 이름 앞에 붙이는 작업 진행
# 중구의 경우 서울과 대구 중복
cate_data['BOR']=cate_data['PVN'].apply(lambda x : x[:2])+' '+cate_data['BOR']

#월별 매출 합계 작업을 위한 전처리 작업
cate_data_1=cate_data.iloc[:,5:]
cate_month=pd.DataFrame()
cate_col=cate_data.columns[5:]
k=cate_col[0]
a=0
col=[]
for i in range(len(cate_col)):
    if k[:6]==cate_col[i][:6]:
        continue
    else:
        cate_month=pd.concat([cate_month,cate_data.iloc[:,a:i].sum(axis=1)],axis=1)
        col.append(k[:6])
        a=i
        k=cate_col[i]
cate_month.columns=col
cate_month=pd.concat([cate_data.BOR,cate_data.ANTC_ITEM_LCLS_NM,cate_month],axis=1)

# 월별 전체 매출 합계 시각화
cate_data_1=cate_data.iloc[:,5:]
cate_month=pd.DataFrame()
cate_col=cate_data.columns[5:]
k=cate_col[0]
a=0
col=[]
for i in range(len(cate_col)):
    if k[:6]==cate_col[i][:6]:
        continue
    else:
        cate_month=pd.concat([cate_month,cate_data.iloc[:,a:i].sum(axis=1)],axis=1)
        col.append(k[:6])
        a=i
        k=cate_col[i]
cate_month.columns=col
cate_month=pd.concat([cate_data.BOR,cate_data.ANTC_ITEM_LCLS_NM,cate_month],axis=1)

# 전체 카테고리별 매출 합계
fig,axes=plt.subplots(nrows=2)
fig.set_size_inches(18,12)
cate_month.groupby(['ANTC_ITEM_LCLS_NM']).sum().T.plot(kind='bar',width=0.75,ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=0, ha='left')
axes[0].set_title('전체 카테고리별 매출 합계',size=20)
axes[0].legend(loc='upper right', frameon=False,fontsize=9,ncol=2)
plt.show()

# 지역마다 업종별 매출 합계 그래프 시각화
fig,axes=plt.subplots(nrows=3,ncols=2)
fig.set_size_inches(18,12)


cate_month.groupby(['BOR']).sum().T.plot(kind='bar',width=0.75,ax=axes[0][0])
plt.setp(axes[0][0].get_xticklabels(), rotation=0, ha='left')
axes[0][0].set_title('구별 매출 합계',size=20)
axes[0][0].legend(loc='upper right', frameon=False,fontsize=9,ncol=2)

cate_month.groupby(['BOR','ANTC_ITEM_LCLS_NM']).sum().T['서울 중구'].plot(kind='bar',width=0.75,ax=axes[1][0])
plt.setp(axes[1][0].get_xticklabels(), rotation=0, ha='left')
axes[1][0].set_title('서울 중구 카테고리별 매출 월 합계',size=20)
axes[1][0].legend(loc='upper right', frameon=False,fontsize=9,ncol=3)
axes[1][0].set_ylim(0,2500000)

cate_month.groupby(['BOR','ANTC_ITEM_LCLS_NM']).sum().T['서울 노원구'].plot(kind='bar',width=0.75,ax=axes[1][1])
plt.setp(axes[1][1].get_xticklabels(), rotation=0, ha='left')
axes[1][1].set_title('서울 노원구 카테고리별 매출 월 합계',size=20)
axes[1][1].legend(loc='upper right', frameon=False,fontsize=9,ncol=3)
axes[1][1].set_ylim(0,2500000)

cate_month.groupby(['BOR','ANTC_ITEM_LCLS_NM']).sum().T['대구 중구'].plot(kind='bar',width=0.75,ax=axes[2][0])
plt.setp(axes[2][0].get_xticklabels(), rotation=0, ha='left')
axes[2][0].set_title('대구 중구 카테고리별 매출 월 합계',size=20)
axes[2][0].legend(loc='upper right', frameon=False,fontsize=9,ncol=3)
axes[2][0].set_ylim(0,2500000)

cate_month.groupby(['BOR','ANTC_ITEM_LCLS_NM']).sum().T['대구 수성구'].plot(kind='bar',width=0.75,ax=axes[2][1])
plt.setp(axes[2][1].get_xticklabels(), rotation=0, ha='left')
axes[2][1].set_title('대구 수성구 카테고리별 매출 월 합계',size=20)
axes[2][1].legend(loc='upper right', frameon=False,fontsize=9,ncol=3)
axes[2][1].set_ylim(0,2500000)


plt.tight_layout()
plt.show()

# 시 단위로 매출 합계 시각화 작업을 위한 전처리 작업
cate_month_1=pd.concat([cate_data.PVN,cate_month],axis=1)
# 시 단위를 기준으로 업종별 매출 합계 그래프 시각화
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,9)

cate_month_1.groupby(['PVN','ANTC_ITEM_LCLS_NM']).sum().T['서울특별시'].plot(kind='bar',width=0.75,ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=0, ha='left')
axes[0].set_title('서울시 카테고리별 매출 합계',size=20)
axes[0].legend(loc='upper right', frameon=False,fontsize=13,ncol=2)
axes[0].set_ylim(0,6000000)

cate_month_1.groupby(['PVN','ANTC_ITEM_LCLS_NM']).sum().T['대구광역시'].plot(kind='bar',width=0.75,ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=0, ha='left')
axes[1].set_title('대구시 카테고리별 매출 합계',size=20)
axes[1].legend(loc='upper right', frameon=False,fontsize=13,ncol=2)
axes[1].set_ylim(0,6000000)

plt.tight_layout()
plt.show()

fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,9)

cate_month_1.groupby(['PVN','ANTC_ITEM_LCLS_NM']).sum().T['서울특별시'].plot(kind='line',ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=0, ha='left')
axes[0].set_title('서울시 카테고리별 매출 합계',size=20)
axes[0].legend(loc='upper right', frameon=False,fontsize=13,ncol=2)
axes[0].set_ylim(0,6000000)

cate_month_1.groupby(['PVN','ANTC_ITEM_LCLS_NM']).sum().T['대구광역시'].plot(kind='line',ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=0, ha='left')
axes[1].set_title('대구시 카테고리별 매출 합계',size=20)
axes[1].legend(loc='upper right', frameon=False,fontsize=13,ncol=2)
axes[1].set_ylim(0,6000000)

plt.tight_layout()
plt.show()

fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,9)

cate_month_1.groupby('PVN').sum().T['서울특별시'].plot(kind='bar',width=0.75,ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=0, ha='left')
axes[0].set_title('서울시 월별 매출 합계',size=20)
axes[0].legend(loc='upper right', frameon=False,fontsize=13,ncol=2)
axes[0].set_ylim(0,14000000)

cate_month_1.groupby('PVN').sum().T['대구광역시'].plot(kind='bar',width=0.75,ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=0, ha='left')
axes[1].set_title('대구시 월별 매출 합계',size=20)
axes[1].legend(loc='upper right', frameon=False,fontsize=13,ncol=2)
axes[1].set_ylim(0,14000000)

plt.tight_layout()
plt.show()

fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,9)

cate_month_1.groupby('PVN').sum().T['서울특별시'].plot(kind='line',ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=0, ha='left')
axes[0].set_title('서울시 월별 매출 합계',size=20)
axes[0].legend(loc='upper right', frameon=False,fontsize=13,ncol=2)
axes[0].set_ylim(0,14000000)

cate_month_1.groupby('PVN').sum().T['대구광역시'].plot(kind='line',ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=0, ha='left')
axes[1].set_title('대구시 월별 매출 합계',size=20)
axes[1].legend(loc='upper right', frameon=False,fontsize=13,ncol=2)
axes[1].set_ylim(0,14000000)

plt.tight_layout()
plt.show()

#%% 물류데이터(CJ올리브네트웍스) 전처리 및 분석

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import table
import csv
import time
%matplotlib inline


import seaborn as sns
plt.style.use('ggplot')


import matplotlib.font_manager as fm
path = 'C:\Windows\Fonts\malgunbd.ttf'
font_name = fm.FontProperties(fname=path).get_name()
plt.rc('font', family=font_name)
import warnings
warnings.filterwarnings(action='ignore')
pd.options.display.float_format = '{:.2f}'.format

# 데이터 분석 작업 시작
# 파일 호출
delivery=pd.read_excel('2020 빅콘테스트_CJ올리브네트웍스_제공DB.xlsx',encoding='utf-8')
d=[]
for i in range(156571):
    d.append(delivery['CTPV_NM'][i]+' '+delivery['CTGG_NM'][i])


delivery1=pd.DataFrame({'배송년월':list(delivery['DL_YMD']),
                          '상품': list(delivery['DL_GD_LCLS_NM']),
                         '시도': list(delivery['CTPV_NM']),
                         '시군구':d,
                         '송장건수':list(delivery['INVC_CONT'])})
b=[]
for i in range(156571):
    k=str(delivery1['배송년월'][i])[0:4]
    b.append(k)

delivery1['배송년월']=b
#20년도 데이터는 76892번째 부터
k=0
for i in range(156571):
    a=str(delivery['DL_YMD'][i])[0:2]
    if a=='20':
        print(k)
        break
    else:
        k+=1

#연도별로 데이터 나누기
year_19 = delivery[0:76892]
year_20 = delivery[76892:156572].reset_index(drop=True)

#배송 년월일 -> 년월로 수정
a=[]
for i in range(76892):
    k=str(year_19['DL_YMD'][i])[0:4]
    a.append(k)
    
year_19['DL_YMD']=a

# 시도 시군구 합치기 
c=[]
for i in range(76892):
    c.append(year_19['CTPV_NM'][i]+' '+year_19['CTGG_NM'][i])


delivery_19=pd.DataFrame({'배송년월':list(year_19['DL_YMD']),
                          '상품': list(year_19['DL_GD_LCLS_NM']),
                         '시도': list(year_19['CTPV_NM']),
                         '시군구':c,
                         '송장건수':list(year_19['INVC_CONT'])})

b=[]
for i in range(79679):
    k=str(year_20['DL_YMD'][i])[0:4]
    b.append(k)

year_20['DL_YMD']=b
d=[]
for i in range(79679):
    d.append(year_20['CTPV_NM'][i]+' '+year_20['CTGG_NM'][i])

delivery_20=pd.DataFrame({'배송년월':list(year_20['DL_YMD']),
                          '상품': list(year_20['DL_GD_LCLS_NM']),
                         '시도': list(year_20['CTPV_NM']),
                         '시군구':d,
                         '송장건수':list(year_20['INVC_CONT'])})

# 연도별 송장 건수 비교
# 20년도에 송장 건수가 증가한 것을 알 수 있다.
#연도 날짜별 각 품목 송장건수 비교
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,15)

all_19=pd.DataFrame(delivery_19['송장건수'].groupby([delivery_19['배송년월'],delivery_19['상품']]).sum())
all_19.plot(kind='bar',width=0.75,ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=90, ha='left')
axes[0].set_ylim(0,800000)
axes[0].set_title('19년도 날짜별 전체 송장건수',size=20)


all_20=pd.DataFrame(delivery_20['송장건수'].groupby([delivery_20['배송년월'],delivery_20['상품']]).sum())
all_20.plot(kind='bar',width=0.75,ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=90, ha='left')
axes[1].set_ylim(0,800000)
axes[1].set_title('20년도 날짜별 전체 송장건수',size=20)

plt.tight_layout()
plt.show()

#19년도 전체 품목별 송장빈도
count_item19 = pd.DataFrame(year_19['INVC_CONT'].groupby(year_19['DL_GD_LCLS_NM']).sum())
count_item19=count_item19.sort_values(by='INVC_CONT' ,ascending=False)
item19=[]
for i in range(10):
    item19.append(count_item19.index[i])

#20년도 전체 품목별 송장빈도
count_item20 = pd.DataFrame(year_20['INVC_CONT'].groupby(year_20['DL_GD_LCLS_NM']).sum())
count_item20=count_item20.sort_values(by='INVC_CONT' ,ascending=False)
item20=[]
for i in range(10):
    item20.append(count_item20.index[i])

sum_item19=0
sum_item20=0
for i in range(10):
    sum_item19=sum_item19+count_item19['INVC_CONT'][i]
    sum_item20=sum_item20+count_item20['INVC_CONT'][i]
print('19년도 전체 송장 수:',sum_item19)
print('20년도 전체 송장 수:',sum_item20)

per_invc_19=[]
per_invc_20=[]
item=[]
for i in range(10):
    a=round((count_item19['INVC_CONT'][i]/sum_item19)*100,1)
    b=round((count_item20['INVC_CONT'][i]/sum_item20)*100,1)
    per_invc_19.append(a)
    per_invc_20.append(b)

    
df1=pd.DataFrame({'상품':item19,
                 '19년도 비율':per_invc_19})

df2=pd.DataFrame({'상품':item20,
                 '20년도 비율':per_invc_20})
#count_item19['per_invc']=per_invc_19
#count_item20['per_invc']=per_invc_20
#del count_item19['INVC_CONT']
#del count_item20['INVC_CONT']
#count_item19.reset_index()
#count_item20.reset_index()
df3=pd.merge(df1,df2, on='상품')

## 송장빈도 연도별 비율
# 1. 식품은 전년대비 4.2% 증가
# 2. 생활건강은 전년대비 2.6% 증가
# 3. 화장품/미용이 전년대비 1.7% 증가
# 4. 패션의류는 전년대비 3.4% 감소
# 5. 나머지 지역은 전체적으로 비슷하다.

plt.figure(figsize=(28,10))
color1=['orangered','deepskyblue','peachpuff','lightgreen','lightyellow','lightgray','mediumpurple','orchid','lightcoral','lightpink']
color2=['deepskyblue','peachpuff','orangered','lightgreen','lightgray','mediumpurple','lightyellow','orchid','lightcoral','lightpink']
# plot chart
ax1 = plt.subplot(1,3,1, aspect='equal')
df1.plot(kind='pie', y = '19년도 비율', ax=ax1, 
 startangle=90, shadow=False, labels=None,legend = False, fontsize=18,colors=color1)
plt.title("19년도 각 품목 송장건수 비율",fontsize=20)

ax2 = plt.subplot(1,3,2, aspect='equal')
df2.plot(kind='pie', y = '20년도 비율', ax=ax2, 
 startangle=90, shadow=False, labels=None,legend = False, fontsize=18,colors=color2)
plt.title("20년도 각 품목 송장건수 비율",fontsize=20)

# plot table
ax3 = plt.subplot(1,3,3)
plt.axis('off')
tbl = table(ax3, df3, loc='center')
for i in range(10):
    tbl._cells[(i+1, 0)].set_facecolor(color1[i])
tbl.scale(1.5, 3.0)
tbl.auto_set_font_size(False)
tbl.set_fontsize(25)
plt.show()

## 지역별 식품 송장건수 추이
#20년도에 전체적으로 급격하게 증가한걸 알 수 있다.

fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,15)

food_19=delivery_19[(delivery_19['상품'] == '식품')& (delivery_19['시도']=='서울특별시')]
seoul_food=pd.DataFrame(food_19['송장건수'].groupby([food_19['배송년월'],food_19['시군구']]).sum())
seoul_food.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
seoul_food.reset_index(inplace=True)

food_20 = delivery_20[(delivery_20['상품'] == '식품')& (delivery_20['시도']=='서울특별시')]
fo20=pd.DataFrame(food_20['송장건수'].groupby([food_20['배송년월'],food_20['시군구']]).sum())
fo20.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
fo20.reset_index(inplace=True)
seoul_food['20년도']=list(fo20['송장건수'])
seoul_food.rename(columns = {"송장건수": "19년도"}, inplace = True)
seoul_food.set_index(seoul_food['시군구'],inplace=True)

seoul_food.plot(kind='bar',width=0.75,ax=axes[0])
axes[0].set_ylim(0,320000)
plt.setp(axes[0].get_xticklabels(), rotation=90, ha='left')

axes[0].set_title('2~5월별 서울특별시 식품 송장건수',size=20)

food_19=delivery_19[(delivery_19['상품'] == '식품')& (delivery_19['시도']=='대구광역시')]
daegu_food=pd.DataFrame(food_19['송장건수'].groupby([food_19['배송년월'],food_19['시군구']]).sum())
daegu_food.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
daegu_food.reset_index(inplace=True)

food_20 = delivery_20[(delivery_20['상품'] == '식품')& (delivery_20['시도']=='대구광역시')]
fo20=pd.DataFrame(food_20['송장건수'].groupby([food_20['배송년월'],food_20['시군구']]).sum())
fo20.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
fo20.reset_index(inplace=True)
daegu_food['20년도']=list(fo20['송장건수'])
daegu_food.rename(columns = {"송장건수": "19년도"}, inplace = True)
daegu_food.set_index(daegu_food['시군구'],inplace=True)

daegu_food.plot(kind='bar',width=0.75,ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=90, ha='left')
axes[1].set_ylim(0,320000)
axes[1].set_title('2~5월별 대구광역시 식품 송장건수',size=20)
         
plt.tight_layout()
plt.show()

## 지역별 생활건강 송장건수
#20년도에 전체적으로 증가한 것을 알 수 있다.
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,15)

food_19=delivery_19[(delivery_19['상품'] == '생활건강')& (delivery_19['시도']=='서울특별시')]
seoul_food=pd.DataFrame(food_19['송장건수'].groupby([food_19['배송년월'],food_19['시군구']]).sum())
seoul_food.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
seoul_food.reset_index(inplace=True)

food_20 = delivery_20[(delivery_20['상품'] == '생활건강')& (delivery_20['시도']=='서울특별시')]
fo20=pd.DataFrame(food_20['송장건수'].groupby([food_20['배송년월'],food_20['시군구']]).sum())
fo20.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
fo20.reset_index(inplace=True)
seoul_food['20년도']=list(fo20['송장건수'])
seoul_food.rename(columns = {"송장건수": "19년도"}, inplace = True)
seoul_food.set_index(seoul_food['시군구'],inplace=True)

seoul_food.plot(kind='bar',width=0.75,ax=axes[0])
axes[0].set_ylim(0,320000)
plt.setp(axes[0].get_xticklabels(), rotation=90, ha='left')

axes[0].set_title('2~5월별 서울특별시 생활건강 송장건수',size=20)

food_19=delivery_19[(delivery_19['상품'] == '생활건강')& (delivery_19['시도']=='대구광역시')]
daegu_food=pd.DataFrame(food_19['송장건수'].groupby([food_19['배송년월'],food_19['시군구']]).sum())
daegu_food.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
daegu_food.reset_index(inplace=True)

food_20 = delivery_20[(delivery_20['상품'] == '생활건강')& (delivery_20['시도']=='대구광역시')]
fo20=pd.DataFrame(food_20['송장건수'].groupby([food_20['배송년월'],food_20['시군구']]).sum())
fo20.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
fo20.reset_index(inplace=True)
daegu_food['20년도']=list(fo20['송장건수'])
daegu_food.rename(columns = {"송장건수": "19년도"}, inplace = True)
daegu_food.set_index(daegu_food['시군구'],inplace=True)

daegu_food.plot(kind='bar',width=0.75,ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=90, ha='left')
axes[1].set_ylim(0,320000)
axes[1].set_title('2~5월별 대구광역시 생활건강 송장건수',size=20)
         
plt.tight_layout()
plt.show()

## 지역별 화장품/미용 송장건수
#20년도에 전체적으로 증가한 것을 알 수 있다.
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,15)

food_19=delivery_19[(delivery_19['상품'] == '화장품/미용')& (delivery_19['시도']=='서울특별시')]
seoul_food=pd.DataFrame(food_19['송장건수'].groupby([food_19['배송년월'],food_19['시군구']]).sum())
seoul_food.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
seoul_food.reset_index(inplace=True)

food_20 = delivery_20[(delivery_20['상품'] == '화장품/미용')& (delivery_20['시도']=='서울특별시')]
fo20=pd.DataFrame(food_20['송장건수'].groupby([food_20['배송년월'],food_20['시군구']]).sum())
fo20.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
fo20.reset_index(inplace=True)
seoul_food['20년도']=list(fo20['송장건수'])
seoul_food.rename(columns = {"송장건수": "19년도"}, inplace = True)
seoul_food.set_index(seoul_food['시군구'],inplace=True)

seoul_food.plot(kind='bar',width=0.75,ax=axes[0])
plt.setp(axes[0].get_xticklabels(), rotation=90, ha='left')
axes[0].set_ylim(0,320000)
axes[0].set_title('2~5월별 서울특별시 화장품/미용 송장건수',size=20)

food_19=delivery_19[(delivery_19['상품'] == '화장품/미용')& (delivery_19['시도']=='대구광역시')]
daegu_food=pd.DataFrame(food_19['송장건수'].groupby([food_19['배송년월'],food_19['시군구']]).sum())
daegu_food.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
daegu_food.reset_index(inplace=True)

food_20 = delivery_20[(delivery_20['상품'] == '화장품/미용')& (delivery_20['시도']=='대구광역시')]
fo20=pd.DataFrame(food_20['송장건수'].groupby([food_20['배송년월'],food_20['시군구']]).sum())
fo20.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
fo20.reset_index(inplace=True)
daegu_food['20년도']=list(fo20['송장건수'])
daegu_food.rename(columns = {"송장건수": "19년도"}, inplace = True)
daegu_food.set_index(daegu_food['시군구'],inplace=True)

daegu_food.plot(kind='bar',width=0.75,ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=90, ha='left')
axes[1].set_ylim(0,320000)
axes[1].set_title('2~5월별 대구광역시 화장품/미용 송장건수',size=20)
         
plt.tight_layout()
plt.show()

# 지역별 패션의류 송장건수
#3월, 4월 서울특별시 중구와 대구광역시 중구 패션의류 송장건수가 감소한 것을 알 수 있다.
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,15)

food_19=delivery_19[(delivery_19['상품'] == '패션의류')& (delivery_19['시도']=='서울특별시')]
seoul_food=pd.DataFrame(food_19['송장건수'].groupby([food_19['배송년월'],food_19['시군구']]).sum())
seoul_food.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
seoul_food.reset_index(inplace=True)

food_20 = delivery_20[(delivery_20['상품'] == '패션의류')& (delivery_20['시도']=='서울특별시')]
fo20=pd.DataFrame(food_20['송장건수'].groupby([food_20['배송년월'],food_20['시군구']]).sum())
fo20.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
fo20.reset_index(inplace=True)
seoul_food['20년도']=list(fo20['송장건수'])
seoul_food.rename(columns = {"송장건수": "19년도"}, inplace = True)
seoul_food.set_index(seoul_food['시군구'],inplace=True)

seoul_food.plot(kind='bar',width=0.75,ax=axes[0])
axes[0].set_ylim(0,320000)
plt.setp(axes[0].get_xticklabels(), rotation=90, ha='left')
axes[0].set_title('2~5월별 서울 패션의류 송장건수',size=20)

food_19=delivery_19[(delivery_19['상품'] == '패션의류')& (delivery_19['시도']=='대구광역시')]
daegu_food=pd.DataFrame(food_19['송장건수'].groupby([food_19['배송년월'],food_19['시군구']]).sum())
daegu_food.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
daegu_food.reset_index(inplace=True)

food_20 = delivery_20[(delivery_20['상품'] == '패션의류')& (delivery_20['시도']=='대구광역시')]
fo20=pd.DataFrame(food_20['송장건수'].groupby([food_20['배송년월'],food_20['시군구']]).sum())
fo20.sort_values(by=['시군구'],axis=0,ascending=False,inplace=True)
fo20.reset_index(inplace=True)
daegu_food['20년도']=list(fo20['송장건수'])
daegu_food.rename(columns = {"송장건수": "19년도"}, inplace = True)
daegu_food.set_index(daegu_food['시군구'],inplace=True)

daegu_food.plot(kind='bar',width=0.75,ax=axes[1])
plt.setp(axes[1].get_xticklabels(), rotation=90, ha='left')
axes[1].set_ylim(0,320000)
axes[1].set_title('2~5월별 대구 패션의류 송장건수',size=20)
         
plt.tight_layout()
plt.show()

# 시군구별 송장건수 합계
#송장건수가 점점 증가하고 있는 추세이다.
total = pd.DataFrame(delivery1['송장건수'].groupby([delivery1['배송년월'],delivery1['시군구']]).sum().unstack())

# 특별시별 송장건수 합계
#전체적으로 송장건수가 증가하는 것을 볼 수 있다.
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15,15)

seoul_t=delivery1[delivery1['시도']=='서울특별시']
seoul_total=pd.DataFrame(seoul_t['송장건수'].groupby(seoul_t['배송년월']).sum())
seoul_total.plot(ax=axes[0])
axes[0].set_ylim(0,1600000)
axes[0].set_title('서울특별시 송장건수 합계',size=20)

deagu_t=delivery1[delivery1['시도']=='대구광역시']
deagu_total=pd.DataFrame(deagu_t['송장건수'].groupby(deagu_t['배송년월']).sum())
deagu_total.plot(ax=axes[1])
axes[1].set_ylim(0,1600000)
axes[1].set_title('대구광역시 송장건수 합계',size=20)

total.plot(figsize=(12,12),title='구별 송장건수 합계',fontsize=12)

# 월별 송장건수 합계
#20년도 3월에 급격하게 증가했다가 그 이후로 비슷한 추이로 가고 있다.
total1=pd.DataFrame(delivery1['송장건수'].groupby(delivery1['배송년월']).sum())
total1.plot(figsize=(12,12),title='월별 송장건수 합계',fontsize=12)

# 시, 상품별 송장건수 합계
#식품 송장건수가 20년도에 들어 압도적으로 1위하였다.
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(20,20)

seoul12=delivery1[delivery1['시도']=='서울특별시']
seoul12_item=pd.DataFrame(seoul12['송장건수'].groupby([seoul12['배송년월'],seoul1['상품']]).sum().unstack())
seoul12_item.plot(ax=axes[0],fontsize=15)
plt.setp(axes[0].get_xticklabels(), rotation=0, ha='left')
axes[0].set_ylim(0,380000)
axes[0].set_title('서울특별시 상품별 송장건수',size=20)

daegu1=delivery1[delivery1['시도']=='대구광역시']
daegu1_item=pd.DataFrame(daegu1['송장건수'].groupby([daegu1['배송년월'],daegu1['상품']]).sum().unstack())
daegu1_item.plot(ax=axes[1],fontsize=15)
plt.setp(axes[1].get_xticklabels(), rotation=0, ha='left')
axes[1].set_ylim(0,380000)
axes[1].set_title('대구광역시  상품별 송장건수',size=20)

# 시군구 상품별 송장건수
#식품 송장건수가 급격하게 증가하여 20년도에는 모든지역에서 송장건수 1위를 차지하고 있다.
fig,axes=plt.subplots(nrows=4,ncols=1)
fig.set_size_inches(25,25)

seoul1=delivery1[delivery1['시군구']=='서울특별시 노원구']
seoul1_item=pd.DataFrame(seoul1['송장건수'].groupby([seoul1['배송년월'],seoul1['상품']]).sum().unstack())
seoul1_item.plot(ax=axes[0],fontsize=15)
plt.setp(axes[0].get_xticklabels(), rotation=0, ha='left')
axes[0].set_ylim(0,300000)
axes[0].set_title('서울특별시 노원구 상품별 송장건수',size=20)

seoul2=delivery1[delivery1['시군구']=='서울특별시 중구']
seoul2_item=pd.DataFrame(seoul2['송장건수'].groupby([seoul2['배송년월'],seoul2['상품']]).sum().unstack())
seoul2_item.plot(ax=axes[1],fontsize=15)
plt.setp(axes[1].get_xticklabels(), rotation=0, ha='left')
axes[1].set_ylim(0,300000)
axes[1].set_title('서울특별시 중구 상품별 송장건수',size=20)

daegu1=delivery1[delivery1['시군구']=='대구광역시 수성구']
daegu1_item=pd.DataFrame(daegu1['송장건수'].groupby([daegu1['배송년월'],daegu1['상품']]).sum().unstack())
daegu1_item.plot(ax=axes[2],fontsize=15)
plt.setp(axes[2].get_xticklabels(), rotation=0, ha='left')
axes[2].set_ylim(0,300000)
axes[2].set_title('대구광역시 수성구 상품별 송장건수',size=20)

daegu2=delivery1[delivery1['시군구']=='대구광역시 중구']
daegu2_item=pd.DataFrame(daegu2['송장건수'].groupby([daegu2['배송년월'],daegu2['상품']]).sum().unstack())
daegu2_item.plot(ax=axes[3],fontsize=15)
plt.setp(axes[3].get_xticklabels(), rotation=0, ha='left')
axes[3].set_ylim(0,300000)
axes[3].set_title('대구광역시 중구 상품별 송장건수',size=20)

# 성남시 확진자 추이
##출처: http://www.gidcc.or.kr/%EC%BD%94%EB%A1%9C%EB%82%98covid-19-%ED%98%84%ED%99%A9/
nam=pd.read_csv('1-1_data.csv',encoding='utf-8')
nam=nam.rename(columns={'기준일(발병일, 확진일 선택)' : "년월일", '레코드 수': '확진자'})
nam=nam.sort_index(ascending=False)
nam['월'] = nam.년월일.apply(lambda x : str(x)[6])
nam
nam1=pd.DataFrame(nam['확진자'].groupby([nam['월']]).sum())
nam1.plot(figsize=(15,15),fontsize=15,title='성남시 확진자 추이')

#%%


