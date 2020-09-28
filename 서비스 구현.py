#%% Alarm standard(알림 서비스)
import json
import folium
import os
import re
import sys
mod = sys.modules[__name__]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


import seaborn as sns
plt.style.use('ggplot')


import geopandas as gpd
import matplotlib.font_manager as fm
path = 'C:\Windows\Fonts\malgunbd.ttf'
font_name = fm.FontProperties(fname=path).get_name()
plt.rc('font', family=font_name)
import warnings
warnings.filterwarnings(action='ignore')
pd.options.display.float_format = '{:.2f}'.format
import chart_studio.plotly as py
import cufflinks as cf
%config InlineBackend.figure_format = 'retina'
cf.go_offline()

# 인구 단계 표현
# 1. 파일 불러오기 및 정제
Flow = pd.read_csv('Input/Flow_SK/Four_Region_Flow_TIme.csv',index_col = [0])
Nowon_list_2 = ['월계1동',
 '월계2동',
 '월계3동',
 '공릉1동',
 '공릉2동',
 '하계1동',
 '하계2동',
 '중계본동',
 '중계1동',
 '중계4동',
 '중계2.3동',
 '상계1동',
 '상계2동',
 '상계3.4동',
 '상계5동',
 '상계6.7동',
 '상계8동',
 '상계9동',
 '상계10동']
Flow_list = []
for i in Nowon_list_2:
    Flow_list.append(Flow[Flow.HDONG_NM == i])
Nowon_Flow = pd.concat(Flow_list,axis = 0)
Nowon_Flow = Nowon_Flow.drop(['HDONG_CD'],axis = 1)

# EDA 작업 진행
# 동 단위로 데이터 세분화 작업 진행
Kind = dict(list(Nowon_Flow.groupby('HDONG_NM')))
# 0~18 숫자 확인용
desc = dict(zip(pd.Series(Kind.keys(), dtype='category').cat.codes,list(Kind.keys())))

# 동적 변수 할당 작업 진행
for i in range(len(Kind.keys())):
    setattr(mod,'Nowon_{}'.format(i),Kind[list(Kind.keys())[i]])
Nowon_0.STD_YM = Nowon_0.STD_YM.apply(lambda x : str(x))

# "공릉 1동" 을 예시로 시간대별 유동인구 추이 시각화
plt.figure(figsize = (10,40))
Nowon_0[Nowon_0.STD_YM.str.startswith('2020')].iloc[:,3:].plot(subplots = True,figsize=(10,70))
plt.show()

# 2020년 데이터만 추출
Test_copy = Nowon_0.reset_index().drop('index',axis = 1 )

# 20년 4월 18일 15시 노원구 지도 그리기
data_dir_dong= 'Data/bnd_dong_00_2019_2019'
Dong = gpd.read_file(os.path.join(data_dir_dong,'bnd_dong_00_2019_2019_2Q.shp'),encoding = 'cp949')
A = '월계1동,월계2동,월계3동,공릉1동,공릉2동,하계1동,하계2동,중계본동,중계1동,중계4동,중계2·3동,상계1동,상계2동,상계3·4동,상계5동,상계6·7동,상계8동,상계9동,상계10동'
Nowon_list = A.split(',')
data_list= []
for i in Nowon_list:
    data_list.append(Dong[Dong.adm_dr_nm == i])
# 노원구 파일 정리
Nowon_data = pd.concat(data_list,axis = 0)
Flow = pd.read_csv('Input/Flow_SK/Four_Region_Flow_TIme.csv',index_col = [0])
Nowon_list_2 = ['월계1동',
 '월계2동',
 '월계3동',
 '공릉1동',
 '공릉2동',
 '하계1동',
 '하계2동',
 '중계본동',
 '중계1동',
 '중계4동',
 '중계2.3동',
 '상계1동',
 '상계2동',
 '상계3.4동',
 '상계5동',
 '상계6.7동',
 '상계8동',
 '상계9동',
 '상계10동']
Flow_list = []
for i in Nowon_list_2:
    Flow_list.append(Flow[Flow.HDONG_NM == i])
Nowon_Flow = pd.concat(Flow_list,axis = 0)
Nowon_Flow.STD_YMD = Nowon_Flow.STD_YMD.apply(lambda x : str(x))
Nowon_Flow_2020 = Nowon_Flow[Nowon_Flow.STD_YMD.str.startswith('2020')]
# 2020년 4월 18일 15시 노원구 인구수
Nowon_04_18 = Nowon_Flow_2020[Nowon_Flow_2020.STD_YMD == '20200418'][['HDONG_NM','TMST_15']]
Nowon_list = Target_data.HDONG_NM.values

# 지도 그리기 작업 순서 시작
# 1. 행정동 코드 및 유동인구수를 표현하는 데이터 작업
Nowon_04_18_map =Nowon_Flow_2020[Nowon_Flow_2020.STD_YMD == '20200418'][['HDONG_CD',"TMST_15"]]
Nowon_04_18_map.HDONG_CD = Nowon_data.adm_dr_cd.values

# 2. 노원구 지역만 나오게 Json파일 수정
geo_path = 'Data/skorea-submunicipalities-2018-geo.json'
geo_str = json.load(open(geo_path,encoding = 'utf-8'))

# Raw data 보존하고자 geo_test_str라는 이름으로 데이터 복사
geo_test_str = geo_str
geo_list= []
want_data = Nowon_data[['adm_dr_cd','adm_dr_nm']]
geo_copy = geo_str['features']
for i in range(3504):
    for j in want_data['adm_dr_nm'].values:
        if geo_copy[i]['properties']['name'] == j:
            geo_list.append(geo_copy[i])

# 기존 데이터에 대입
geo_test_str['features'] = geo_list
seoul_map = folium.Map(location = [37.650641, 127.074564],zoom_start = 12,tiles = 'cartodbpositron')
seoul_map.choropleth(geo_data = geo_test_str,
                    data = Nowon_17_data,
                    columns = list(Nowon_17_data.columns),
                    fill_color = 'PuRd',
                    key_on = 'properties.code',
                    highlight = True,
                    fill_opacity = 0.5, line_opacity = 1,
                    legend_name = 'Population per time')

# 밀집도에 대한 기준 설정 코드 시작
## 1. 공휴일,주말 제외 작업
#- 공휴일
# ```
# 4/15 국회의원 선거 수요일
# 4/30 부처님 오신날 목요일
# 5/1 근로자의 날
# 5/5 어린이날 화요일
# ```
# - 이유 : 상대적으로 아주 낮은 유동인구 데이터를 보여주었으므로 제거합니다.
# 이유 설명 - 1가지 예시(공휴일)
Nowon_Flow_2020[Nowon_Flow_2020.HDONG_NM == '공릉1동'].TMST_15.plot(figsize = (20,10))
# 공휴일 확인_3월 1일
# 공휴일인 3월 1일이 가장 낮은 값을 지닌다. 이 값은 앞으로 인구밀집도에 대한 기준을 보수적으로 평가하도록 유도하기 때문에 제거합니다.
Nowon_Flow_2020 = Nowon_Flow_2020.drop('STD_YM',axis = 1)
Nowon_Flow_2020.STD_YMD = pd.to_datetime(Nowon_Flow_2020.STD_YMD,format = '%Y-%m-%d')
Nowon_Flow_2020 = Nowon_Flow_2020.drop('HDONG_CD',axis = 1)
Nowon_Flow_2020_except = Nowon_Flow_2020.drop([13404,14439,14508,14784],axis = 0)
Nowon_Flow_2020_except = Nowon_Flow_2020_except[Nowon_Flow_2020_except.STD_YMD.apply(lambda x : x.weekday()) <5]
# 주말, 공휴일 제외 완료
HDONG_list = Nowon_Flow_2020_except.HDONG_NM.unique() # 특수문자  : 점
number_list = np.arange(len(HDONG_list))
set_dict = dict(zip(number_list,HDONG_list))
HDONG_dict = dict(list(Nowon_Flow_2020_except.groupby('HDONG_NM')))
mod = sys.modules[__name__]
for i, name in enumerate(HDONG_list):
    setattr(mod,"Nowon_{}".format(i),HDONG_dict[name])

Target_data_04_18 = Nowon_Flow_2020[Nowon_Flow_2020.STD_YMD == '20200418'][['HDONG_NM','TMST_15']]
# 인구 / 면적 연산 작업 실시
# 면적 당 인구
HDONG_list = list(map(lambda x : x.replace('.',','),HDONG_list))
HDONG_area = []
for name in HDONG_list:
    Dong_1 = Nowon_data.loc[Nowon_data.adm_dr_nm == name, "geometry"].squeeze()
    Dong_1_area = Dong_1.area / 10**6 # 미터(m)단위이므로 km제곱으로 변경하려면 10^6으로 나눈다.
    HDONG_area.append(Dong_1_area)

# 2. 연산 작업 진행
Target_data_04_18.TMST_15 = Target_data_04_18.TMST_15 / HDONG_area

# 밀집도 기준 설정
# 3월 2일부터 4월 17일 15시 유동인구 데이터 활용
def Part_divide_except_04(data,name): # 4월달만
    Test_1 = data.reset_index().drop('index',axis = 1)
    
    # 특수문자 기호를 콤마(',')
    Test_1.HDONG_NM = Test_1.HDONG_NM.str.replace('.',',')
    
    ## 면적을 구하는 코드
    # 1.shp파일 불러오기
    data_dir_dong= 'Data/bnd_dong_00_2019_2019'
    Dong = gpd.read_file(os.path.join(data_dir_dong,'bnd_dong_00_2019_2019_2Q.shp'),encoding = 'cp949')
    
    # 특수문자 변환 점('.')으로
    Dong.adm_dr_nm= Dong.adm_dr_nm.str.replace('·','.')
    
    # 2.노원구 데이터만 추출, 현재 특수기호 문자 점('.')으로 표현
    A = '월계1동,월계2동,월계3동,공릉1동,공릉2동,하계1동,하계2동,중계본동,중계1동,중계4동,중계2.3동,상계1동,상계2동,상계3.4동,상계5동,상계6.7동,상계8동,상계9동,상계10동'
    Nowon_list = A.split(',')
    data_list= []

    for i in Nowon_list:
        data_list.append(Dong[Dong.adm_dr_nm == i])

    # 3.노원구 파일 정리
    Nowon_data = pd.concat(data_list,axis = 0)

    # 특수문자 기호를 콤마(,)로 변경
    Nowon_data.adm_dr_nm = Nowon_data.adm_dr_nm.str.replace('.',',')
    
    # 4.면적 구하기
    Dong_1 = Nowon_data.loc[Nowon_data.adm_dr_nm == name, "geometry"].squeeze()
    Dong_1_area = Dong_1.area / 10**6 # 미터(m)단위이므로 km제곱으로 변경하려면 10^6으로 나눈다.
    
    
    ## 5.1km제곱당 면적
    Population = Test_1.loc[20:53].TMST_18 / Dong_1_area
    
    
    # 6. 과거 3 분야로 
    Standard = pd.qcut(Population,5).cat.categories
    
    return Standard

#구간 저장
Nowon_standard_15 = []
Nowon_standard_15.append(Part_divide_except_04(Nowon_0,'월계1동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_1,'월계2동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_2,'월계3동'))

Nowon_standard_15.append(Part_divide_except_04(Nowon_3,'공릉1동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_4,'공릉2동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_5,'하계1동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_6,'하계2동'))

Nowon_standard_15.append(Part_divide_except_04(Nowon_7,'중계본동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_8,'중계1동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_9,'중계4동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_10,'중계2,3동'))


Nowon_standard_15.append(Part_divide_except_04(Nowon_11,'상계1동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_12,'상계2동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_13,'상계3,4동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_14,'상계5동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_15,'상계6,7동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_16,'상계8동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_17,'상계9동'))
Nowon_standard_15.append(Part_divide_except_04(Nowon_18,'상계10동'))

Target_data_04_18.HDONG_NM = Target_data_04_18.HDONG_NM.str.replace('.',',')
Target_data_04_18['Standard'] = Nowon_standard_15
Target_data_04_18.columns = ['HDONG_NM','TMST_15','Standard']

# 데이터 저장
Target_data_04_18.to_csv("Data/Target_data_04_18.csv")
#%% 최적 경로 찾기 서비스
import copy
from haversine import haversine
import pandas as pd
from tkinter import *


root = Tk()
root.title('흠형과 아이들')
root.geometry('975x775+5+5')


text = ""
count = 0
departure = ""
destination = ""
crowded = 0

def clear():
    global count, text, result, departure, destination, crowded
    text = ""
    count = 0
    crowded = 0
    result = ""
    departure = ""
    destination = ""
    textEntry.set("")
    textEntry1.set("")
    textEntry2.set("")
    
    
def comfoo():
    global count
    count=1

    
def comfoo1():
    global count, result
    count=2

    
def setTextInput(text):
    global departure, destination
    if(count== 0):
        textEntry.set(text)
        departure = text
        
    elif(count == 1):
        textEntry1.set(text)
        destination = text


def start():
    global result, crowded
    result = ""
    crowded = 0
# 서울시 노원구 동별 인접 노드 초기화
    landscape={
        '상계1동' : {'상계3,4동':0,'상계9동':0,'상계8동':0,'상계5동':0},
        '상계3,4동':{'중계4동':0, '상계5동':0,'상계1동':0},
        '상계5동':{'상계9동':0,'상계1동':0,'상계3,4동':0,'중계4동':0,'상계2동':0},
        '상계9동':{'상계1동':0,'상계5동':0,'상계8동':0,'상계10동':0},
        '상계8동':{'상계1동':0,'상계9동':0,'상계10동':0},
        '상계10동':{'상계8동':0,'상계9동':0,'상계2동':0,'상계6,7동':0},
        '상계2동':{'상계5동':0,'상계10동':0,'상계6,7동':0,'중계4동':0,'중계1동':0},
        '중계4동':{'상계3,4동':0,'상계5동':0,'상계2동':0,'중계1동':0,'중계본동':0},
        '상계6,7동':{'상계10동':0,'상계2동':0,'중계2,3동':0,'월계2동':0,'하계2동':0},
        '중계2,3동':{'상계6,7동':0,'중계1동':0,'하계1동':0,'하계2동':0},
        '중계1동':{'상계2동':0,'중계4동':0,'중계본동':0,'하계1동':0,'중계2,3동':0},
        '중계본동':{'중계4동':0,'중계1동':0,'하계1동':0,'공릉2동':0},
        '하계1동':{'중계1동':0,'중계본동':0,'중계2,3동':0,'하계2동':0,'공릉2동':0},
        '하계2동':{'중계2,3동':0,'상계6,7동':0,'하계1동':0,'월계3동':0,'공릉1동':0},
        '월계2동':{'상계6,7동':0,'월계3동':0,'월계1동':0},
        '월계1동':{'월계2동':0,'월계3동':0},
        '월계3동':{'월계1동':0,'월계2동':0,'공릉1동':0,'하계2동':0},
        '공릉1동':{'월계3동':0,'하계2동':0,'공릉2동':0},
        '공릉2동':{'공릉1동':0,'하계1동':0,'중계본동':0}
        }

    # 동별 위도, 경도
    Dong_dict = {'월계1동': (37.63019940184156, 127.06595931151917),
     '월계2동': (37.64490995356445, 127.0605321871607),
     '월계3동': (37.63498486169752, 127.07081648041262),
     '공릉2동': (37.64548852468367, 127.11248319287002),
     '하계1동': (37.64630116459204, 127.08373317141721),
     '하계2동': (37.63957602617072, 127.07035648321087),
     '중계본동': (37.65777243662645, 127.09456996970314),
     '중계1동': (37.65517736301124, 127.07817360846009),
     '중계4동': (37.66461204792735, 127.09541063032314),
     '상계1동': (37.69613745772456, 127.08517719487207),
     '상계2동': (37.66156826026975, 127.07202312964317),
     '상계5동': (37.674099521678116, 127.07569521533226),
     '상계8동': (37.67118835605322, 127.05836444631622),
     '상계9동': (37.67273947471108, 127.0685854756317),
     '상계10동': (37.6651921548431, 127.065651391478),
     '상계3,4동': (37.69038984836544, 127.09644725709585),
     '상계6,7동': (37.65574831890184, 127.06877717329392),
     '중계2,3동': (37.65048531321757, 127.07183122777153),
     '공릉1동': (37.63152127844566, 127.08166602486317)
     }
    
        # 동별 좌표값
    xy_dict = {'월계1동': (184, 626),
     '월계2동': (145, 545),
     '월계3동': (246, 628),
     '공릉2동': (390, 550),
     '하계1동': (295,500),
     '하계2동': (238, 525),
     '중계본동': (355,425),
     '중계1동': (292, 410),
     '중계4동': (325, 345),
     '상계1동': (185, 150),
     '상계2동': (248, 356),
     '상계5동': (270, 310),
     '상계8동': (168, 278),
     '상계9동': (215, 270),
     '상계10동': (188, 340),
     '상계3,4동': (325, 200),
     '상계6,7동': (200, 405),
     '중계2,3동': (240, 463),
     '공릉1동': (290, 610)
     }
    
    
    land=copy.deepcopy(landscape)
    #  동별 인접 노드 간 거리 계산
    for i in land.keys():
        a=Dong_dict[i]
        for j in land[i].keys():
            b=Dong_dict[j]
            dis=haversine(a,b, unit = 'km')
            land[i][j]=dis


    # 동별 인구 밀집 단계 데이터
    step_data=pd.read_csv('./Target_data_04_18.csv',index_col=0)
    # 단계 문자열 처리
    step_data.Standard.values[0]
    step_data=step_data.reset_index(drop=True)
    step_data
    step_inter=[]
    for i in range(len(step_data)):
        interval=[]
        ch=''
        p=0
        for j in step_data['Standard'][i]:
            if j=="\n":
                break
            if j=='[' or j=='(':
                p=1
                continue
            if j==']' or j==')':
                p=0
                if len(ch)>0:
                    interval.append(ch.strip(' '))
                    ch=''
            if (p==1) and (j!='[' or j!='(' or j!=' '):
                if j!=',':
                    ch+=j
                else:
                    interval.append(ch.strip(' '))
                    ch=''
        step_inter.append(interval)
    # 인구 밀집 단계 구간 type 변환
    for i in range(len(step_inter)):
        step_inter[i]=list(map(float,step_inter[i]))
    # 구간에 맞는 인구 밀집 단계 계산
    TMST=list(step_data['TMST_15'])
    dong=list(step_data['HDONG_NM'])
    step={}
    for i in range(len(TMST)):
        for j in range(len(step_inter[i])):
            if step_inter[i][j]>TMST[i]:
                if j==0:
                    step[dong[i]]=1
                else:
                    step[dong[i]]=(j+1)//2
                break
            if j==9:
                step[dong[i]]=5

    # 인접 노드 최단 거리 계산 및 방문
    def visitDong(routing,visit,n):
        routing[visit]['visited'] = 1 #방문
        for go, dis in land[visit].items(): # 거리 재계산
            todist = routing[visit]['sdis'] + dis
            if ((routing[go]['sdis'] >= todist) or len(routing[go]['route'])==0) and step[go]<=n:
                routing[go]['sdis'] = todist # 거리가 더 짧은 것이 있다면 다시 넣기
                routing[go]['route'] = copy.deepcopy(routing[visit]['route'])
                routing[go]['route'].append(visit)

    # 출발지부터 도착지까지의 최단 경로 및 거리
    # n : 최대 허용 인구 밀집 단계
    def short_route(departure,destination,n):
        global routing
        routing={}
        for place in land.keys():
            routing[place] = {'sdis':0, 'route':[], 'visited':0}

        visitDong(routing,departure,n)
        while True:
            dis_min = max(routing.values(), key=lambda x : x['sdis'])['sdis'] #최대로 초기화
            visit = False
            for Dong, info in routing.items():
                if 0 < info['sdis'] <= dis_min and info['visited']==0: #방문 안한 곳 중에서 최소거리 동
                    dis_min = info['sdis']
                    visit = Dong  # 방문

            if visit == False: # 방문한 곳이 없으면 종료
                break

            visitDong(routing,visit,n)

        if routing[destination]['sdis']==0:
            result = "\n"+"\n"+ "출발 :"+ departure+'   '+'도착 :'+ destination+"\n"+'경로가 없습니다.'
            RUT = Label(frame1, text = result, heigh =5 ).grid(row= 8,columnspan=8)
            textEntry2.set(result)
        
        
        else:
            StrA = "-".join(routing[destination]['route'])
            StrB = "".join(str(routing[destination]['sdis']))
            result = "\n"+"\n"+ "출발 :"+ departure+'   '+'도착 :'+ destination + "\n"+"\n"+"경로 : "+ StrA+"\n"+"최단거리 : "+StrB
            RUT = Label(frame1, textvariable = textEntry2 ).grid(row= 8,columnspan=20)
            textEntry2.set(result)
            
    short_route(departure,destination,int(Getvalue.get()))   
        
             

frame = Frame(root)
frame.place(x=0, y=0)
imge = PhotoImage(file = './지도.png')
imgLabel = Label(frame)
imgLabel.config(image=imge)
imgLabel.pack()
san1 = Button(frame, text = "상계1동",bg='#dcc3db' ,width=8, height =2, relief = FLAT, command=lambda:setTextInput("상계1동"),font='14px').place(x=185,y=150)
san3 = Button(frame, text = "상계3,4동", width=7, height =2, relief = FLAT, command=lambda:setTextInput("상계3,4동"),font='18px' ).place(x=325,y=200)
san5 = Button(frame, text = "상계5동", width=4, height =1, relief = FLAT, command=lambda:setTextInput("상계5동") ).place(x=270,y=310)
san9 = Button(frame, text = "상계9동", width=5, height =1, relief = FLAT, command=lambda:setTextInput("상계9동") ).place(x=215,y=270)
san8 = Button(frame, text = "상계8동", width=4, height =1, relief = FLAT, command=lambda:setTextInput("상계8동") ).place(x=168,y=278)
san10 = Button(frame, text = "상계10동", width=6, height =1, relief = FLAT, command=lambda:setTextInput("상계10동") ).place(x=188,y=340)
san2 = Button(frame, text = "상계2동",bg='#e2d5e5', width=5, height =1, relief = FLAT, command=lambda:setTextInput("상계2동") ).place(x=248,y=356)
san6 = Button(frame, text = "상계6,7동",bg='#c5799a', width=7, height =1, relief = FLAT, command=lambda:setTextInput("상계6,7동") ).place(x=200,y=405)

jon6 = Button(frame, text = "중계4동", width=7, height =1, relief = FLAT, command=lambda:setTextInput("중계4동") ).place(x=325,y=345)
jonbone = Button(frame, text = "중계본동", width=7, height =1, relief = FLAT, command=lambda:setTextInput("중계본동") ).place(x=355,y=425)
jon1 = Button(frame, text = "중계1동", width=4, height =1, relief = FLAT, command=lambda:setTextInput("중계1동") ).place(x=292,y=410)
jon2 = Button(frame, text = "중계2,3동",bg='#e2d5e5', width=6, height =1, relief = FLAT, command=lambda:setTextInput("중계2,3동") ).place(x=240,y=463)

ha1 = Button(frame, text = "하계1동", width=7, height =1, relief = FLAT, command=lambda:setTextInput("하계1동") ).place(x=295,y=500)
ha2 = Button(frame, text = "하계2동", width=4,bg='#e2d5e5', height =1, relief = FLAT, command=lambda:setTextInput("하계2동") ).place(x=238,y=525)

wal2 = Button(frame, text = "월계2동", width=7, height =1, relief = FLAT, command=lambda:setTextInput("월계2동") ).place(x=145,y=545)
wal1 = Button(frame, text = "월계1동", width=5, height =1, relief = FLAT, command=lambda:setTextInput("월계1동") ).place(x=184,y=626)
wal3 = Button(frame, text = "월계3동", width=4, bg='#e2d5e5',height =1, relief = FLAT, command=lambda:setTextInput("월계3동") ).place(x=246,y=628)

gong1 = Button(frame, text = "공릉1동", bg='#e685b4',width=4, height =1, relief = FLAT, command=lambda:setTextInput("공릉1동") ).place(x=290,y=610)
gong2 = Button(frame, text = "공릉2동", bg='#e889b5',width=6, height =2, relief = FLAT, command=lambda:setTextInput("공릉2동"),font='14px' ).place(x=390,y=550)
frame1 = Frame(root)
frame1.place(x=650, y=300)



global textEntry,textEntry1,textEntry2
textEntry=StringVar()
textEntry1=StringVar()
textEntry.set(str(text))
textEntry1.set(str(text))
textEntry2=StringVar()
textEntry2.set(str(text))


StartLabel = Label(frame1, text = "출발지").grid(row=1, column =0)
Startwind = Entry(frame1, justify='right',textvariable = textEntry ).grid(row =1, column =1, columnspan=4)
comfo = Button(frame1, text = "선택", width=6, height =1, command = comfoo).grid(row=1, column =5)
    
TargetLabel = Label(frame1, text = "도착지").grid(row=2, column =0)
Startwind = Entry(frame1, justify='right',textvariable = textEntry1).grid(row =2, column =1, columnspan=4)
comfo1 = Button(frame1, text = "선택", width=6, height =1,command = comfoo1).grid(row=2, column =5)


crowd = Label(frame1, text = "최대밀집단계").grid(row=4, column =0)

Getvalue = IntVar()
one = Radiobutton(frame1, text = '1',value = '1',variable =  Getvalue).grid(row=4, column =1)
two = Radiobutton(frame1, text = '2',value = '2',variable =  Getvalue).grid(row=4, column =2)
three = Radiobutton(frame1, text = '3',value = '3',variable =  Getvalue).grid(row=4, column =3)
four = Radiobutton(frame1, text = '4',value = '4',variable =  Getvalue).grid(row=4, column =4)
five = Radiobutton(frame1, text = '5',value = '5',variable =  Getvalue).grid(row=4, column =5)

Sub = Button(frame1, text = "길 찾기", width=6, height =1, command = start).grid(row=5, column =2)
Del = Button(frame1, text = "초기화", width=6, height =1, command = clear).grid(row=5, column =3)

   

root.mainloop()
 