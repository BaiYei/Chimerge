#ChiMerge 是监督的、自底向上的(即基于合并的)数据离散化方法。它依赖于卡方分析：具有最小卡方值的相邻区间合并在一起，直到满足确定的停止准则。
#ChiMerge算法包括2部分：1、初始化，2、自底向上合并，当满足停止条件的时候，区间合并停止。


import math

import numpy as np
import pandas as pd

iris = pd.read_csv('iris.csv', header=None)
iris.columns = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'target_class']

#将最小卡方值添加进DataFrame
def merge_rows(df,feature):

    tdf = df[:-1]
    distinct_values = sorted(set(tdf['chi2']), reverse=False)

    col_names =  [feature,'Iris-setosa', 'Iris-versicolor', 
                  'Iris-virginica','chi2']
    updated_df  = pd.DataFrame(columns = col_names)  #加入卡方值，生成新的数组
    
    updated_df_index=0
    for index, row in df.iterrows(): 
        if(index==0):
            updated_df.loc[len(updated_df)] = df.loc[index]
            updated_df_index+=1
        else:
            if(df.loc[index-1]['chi2']==distinct_values[0]):
                updated_df.loc[updated_df_index-1]['Iris-setosa']+=df.loc[index]['Iris-setosa']
                updated_df.loc[updated_df_index-1]['Iris-versicolor']+=df.loc[index]['Iris-versicolor']
                updated_df.loc[updated_df_index-1]['Iris-virginica']+=df.loc[index]['Iris-virginica']
            else:
                updated_df.loc[len(updated_df)] = df.loc[index]
                updated_df_index+=1
                
    updated_df['chi2'] = 0.  

    return updated_df
        

#计算卡方值
def calc_chi2(array):
    shape = array.shape
    n = float(array.sum())
    row={}
    column={}
    
    #计算每行的和
    for i in range(shape[0]):
        row[i] = array[i].sum()
    
    #计算每列的和
    for j in range(shape[1]):
        column[j] = array[:,j].sum()

    chi2 = 0
    
    #卡方计算公式
    for i in range(shape[0]):
        for j in range(shape[1]):
            eij = row[i]*column[j] / n 
            oij = array[i,j]  
            if eij==0.:
                chi2 += 0.  #确保不存在NaN值
            else:
                chi2 += math.pow((oij - eij),2) / float(eij)
  
    return chi2
    
#计算每一个类别的卡方值
def update_chi2_column(contingency_table,feature):
    for index, row in contingency_table.iterrows():
        if(index!=contingency_table.shape[0]-1):
            list1=[]
            list2=[]
            list1.append(contingency_table.loc[index]['Iris-setosa'])
            list1.append(contingency_table.loc[index]['Iris-versicolor'])
            list1.append(contingency_table.loc[index]['Iris-virginica'])
            list2.append(contingency_table.loc[index+1]['Iris-setosa'])
            list2.append(contingency_table.loc[index+1]['Iris-versicolor'])
            list2.append(contingency_table.loc[index+1]['Iris-virginica'])
            prep_chi2 = np.array([np.array(list1),np.array(list2)])

            c2 = calc_chi2(prep_chi2)

            contingency_table.loc[index]['chi2'] = c2
    return contingency_table


#计算频次表
def create_contingency_table(dataframe,feature):
    distinct_values = sorted(set(dataframe[feature]), reverse=False)
    col_names =  [feature,'Iris-setosa', 'Iris-versicolor','Iris-virginica','chi2']
    my_contingency  = pd.DataFrame(columns = col_names)
    
    #计算唯一值
    for i in range(len(distinct_values)): 
        temp_df=dataframe.loc[dataframe[feature]==distinct_values[i]]
        count_dict = temp_df["target_class"].value_counts().to_dict()

        setosa_count = 0
        versicolor_count = 0
        virginica_count = 0

        if 'Iris-setosa' in count_dict:
            setosa_count = count_dict['Iris-setosa']
        if 'Iris-versicolor' in count_dict:
            versicolor_count = count_dict['Iris-versicolor']
        if 'Iris-virginica' in count_dict:
            virginica_count = count_dict['Iris-virginica']

        new_row = [distinct_values[i],setosa_count,versicolor_count,virginica_count,0]
        my_contingency.loc[len(my_contingency)] = new_row

    return my_contingency


#ChiMerge
def chimerge(feature, data, max_interval):
    df = data.sort_values(by=[feature],ascending=True).reset_index()
    
    #传入频次表
    contingency_table = create_contingency_table(df,feature)

    #计算初始间隔值
    num_intervals= contingency_table.shape[0] 

    #是否满足最大间隔
    while num_intervals > max_interval: 
        #相邻列的卡方值
        chi2_df = update_chi2_column(contingency_table,feature) 
        contingency_table = merge_rows(chi2_df,feature)
        num_intervals= contingency_table.shape[0]               

    #得出结果
    print('The split points for '+feature+' are:')
    for index, row in contingency_table.iterrows():
        print(contingency_table.loc[index][feature])
    
    print('The final intervals for '+feature+' are:')
    for index, row in contingency_table.iterrows():
        if(index!=contingency_table.shape[0]-1):
            for index2, row2 in df.iterrows():
                if df.loc[index2][feature]<contingency_table.loc[index+1][feature]:
                    temp = df.loc[index2][feature]
        else:
            temp = df[feature].iloc[-1]
        print("["+str(contingency_table.loc[index][feature])+","+str(temp)+"]")
    print(" ")

    

if __name__=='__main__':
    for feature in ['sepal_length', 'sepal_width', 'petal_length','petal_width']:
        chimerge(feature=feature, data=iris, max_interval=6)
