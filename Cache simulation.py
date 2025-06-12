import pickle
import numpy as np
import datetime
import random
import matplotlib.pyplot
import pandas as pd
import time
import os
import sys
import joblib
import json
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import scipy.stats

class data():
    
    def __init__(self):

        self.df = self.download_pickle("/data/cms/scratch/jvilalta/Notebooks_ML/df_final_xCache.pkl")
        perdf = self.download_pickle("per.pkl")

        self.df['date'] = pd.to_datetime(self.df['date']).dt.date

        le = LabelEncoder()

        d_label = self.df['d_label']
        d_label_encoded = le.fit_transform(d_label)
        self.df['d_label_encoded'] = d_label_encoded

        
        self.data_dist =  np.array(perdf[(perdf['arbol'] == '/store/data') & (perdf['type'] == 'RAW')]['Porcentaje_bajado'])/100
        self.mc_dist = np.array(perdf[(perdf['arbol'] == '/store/mc') & (perdf['N_accesses'] == 1)]['Porcentaje_bajado'])/100
        self.user_dist = np.array(perdf[(perdf['arbol'] == '/store/user') & (perdf['N_accesses'] == 1)]['Porcentaje_bajado'])/100

    def download_pickle(self,file_name):
        try:
            with open(file_name, "rb") as file:
                obj = pickle.load(file)
        except:
            obj = pd.read_pickle(file_name)
        return obj
    
    def get_rates(self):
        return self.data_dist, self.mc_dist, self.user_dist
        
    def get_df(self):
        return self.df
        
    def get_tree(self, directory, del_time):
        for root, dirs, files in os.walk(directory):
            files.sort()
            for file in files:
                unixtime = self.get_unixtime(file)
                if unixtime < (del_time - 1296000) and (unixtime + 1296000)  > (del_time - 1296000):
                    return str(file)
        return None
        
    @staticmethod
    def get_unixtime(filename):
        parts = filename.rsplit('_')
        if len(parts) > 1:
            parts = parts[-1].rsplit('.')
            return int(parts[0])
        return None

class cache():
    
    def __init__(self, df, data_dist, mc_dist, user_dist):
        self.data_instance = data()
        self.df = df
        self.data_dist = data_dist
        self.mc_dist = mc_dist
        self.user_dist = user_dist
        self.last_model_path = None
        self.last_model = None

    def apply_distribution(self, workflow, acceso, per_anterior=0):
        if workflow == '/store/data/.../RAW': #nos fijamos
            per = np.random.choice(self.data_dist)
            if per > per_anterior:
                return per
            return per_anterior
        elif workflow == '/store/mc' :
            if acceso == 1:
                return np.random.choice(self.mc_dist)

        elif workflow == '/store/user':
            if acceso == 1:
                return np.random.choice(self.user_dist)

        return 1.

    def run(self ,start_date ,end_date , path, tree, option, size = 200, pct_delete = 0.9, pct_max = 0.95):      
        _HistoricoAdditions = []
        _HistoricoDeletions = {}
        _HistoricoStaticDia = []

        pmiss = 0
        phits = 0

        SIZE = size * 1024 #180 teras pasados a gigas
        cache_restart = SIZE * pct_delete
        max_cache = SIZE * pct_max
        
        mask = ((self.df['date'] >= start_date) & (self.df['date'] < end_date))
        df = self.df.loc[mask]
        
        CACHE = {}
        cache_size = 0
        
        dels = 0
        ndels = 0
        
        if os.path.exists(path+"/"+str(size)+"_deletions.txt"):
          os.remove(path+"/"+str(size)+"_deletions.txt")

        for dia, dia_df in df.groupby('date'):
            start_dia = set([(key,values['tiempo']) for key,values in CACHE.items()])
            
            for row in dia_df.itertuples():
                tamano = row.filesize
                archivo = row.filename
                tiempounix = row.time
                workflow = row.workflow
                d_label_encoded = row.d_label_encoded

                if not CACHE.get(archivo): #Miss
                    
                    per = self.apply_distribution(arbol,acceso=1)
                    tamano = per*tamano
                    
                    if cache_size > max_cache: #Borrado
                        size_a_eliminar = cache_size - cache_restart - tamano
                
                        CACHE, cache_size, new_deletions, dels, ndels = self.XGB_deletion(size_a_eliminar, CACHE, size, dels, ndels, path, tree, option)

                        _HistoricoDeletions[(tiempounix,dia)] =  new_deletions
                        
                    CACHE[archivo] = {'Size': tamano,
                    'N_accesses': 1,
                    'tiempo':tiempounix,
                    'd_label_encoded': d_label_encoded,
                    'accesos_anteriores':[tiempounix],
                    'accesos_anteriores_dias':[dia],
                    'workflow': workflow,}

                    _HistoricoAdditions.append([archivo,tamano,1,True,tiempounix,cache_size,dia]) #archivo, size, n-access, per, miss, tiempounix

                    cache_size += tamano

                else: #Hit

                    n_accesses = CACHE[archivo]['N_accesses'] + 1
                    accesos_anteriores = CACHE[archivo]['accesos_anteriores'] + [tiempounix]
                    accesos_anteriores_dias = CACHE[archivo]['accesos_anteriores_dias'] + [dia]
                    
                    porcentaje_anterior = CACHE[archivo]['porcentaje_bajado']
                    
                    per = self.apply_distribution(arbol,acceso=n_accesses,per_anterior=porcentaje_anterior)
                    tamano = per*tamano
                    
                    del CACHE[archivo]

                    CACHE[archivo]= {'Size': tamano,
                    'N_accesses': n_accesses,
                    'tiempo':tiempounix, 
                    'd_label_encoded': d_label_encoded,
                    'accesos_anteriores':accesos_anteriores,
                    'accesos_anteriores_dias':accesos_anteriores_dias,
                    'workflow': workflow}

                    _HistoricoAdditions.append([archivo,tamano,n_accesses,False,tiempounix,cache_size,dia])
            
            archivos_invariables_hoy = list(start_dia & set([(key,values['tiempo']) for key,values in CACHE.items()])) 
            
            #archivos_invariables_hoy = list(start_dia - set(list(CACHE.items())))
            if archivos_invariables_hoy: #Hay dias en los que no hay ninguno static
                _HistoricoStaticDia.append([(nombre, dia, CACHE[nombre]) for nombre,_ in archivos_invariables_hoy])
        
        ADDITIONSDF = self.create_ADDITIONSDF(_HistoricoAdditions)
        # COMMENTED TO SAVE TIME
        # DELETIONSDF = self.create_DELETIONSDF(_HistoricoDeletions)
        # STATICDIA =  self.create_STATICDIA(_HistoricoStaticDia)
        
        # print(f"\nTotal hits: {phits}")
        # print(f"Total misses: {pmiss}")
        # print(f"Total hits + miss: {phits + pmiss}\n")
        
        return ADDITIONSDF
    
    def XGB_deletion(self,size_a_eliminar, CACHE, size, dels, ndels, path, tree, option):

        f = open(path+"/"+str(size)+"_deletions.txt", "a")
        dels+=1
        del_time=list(CACHE.items())[-1][1]['tiempo']


        model_name = self.data_instance.get_tree(tree, del_time)

        ##LRU mechanism
        if model_name == None: 
            cache_tamanos = np.cumsum([i['Size'] for i in CACHE.values()])
            
            for i in range(1, len(CACHE)):
                if cache_tamanos[i] >= size_a_eliminar:
                    archivos_cache = list(CACHE.items())
                    new_deletions = archivos_cache[:i+1]
                    CACHE = dict(archivos_cache[i+1:])
                    break
                
            cache_size = sum([i['Size'] for i in CACHE.values()])
            ndels+=len(new_deletions)
            del_time=archivos_cache[-1][1]['tiempo']

            
            return CACHE, cache_size, new_deletions, dels, ndels
        
        ## XGB mechanism
        else:
            filesize, d_label, total_accesses, insertion_time = [], [], [], []
            last_time, last_2time, last_3time, last_4time, last_5time = [], [], [], [], []
            deltaT_1_last, recency_1st, times, d_label_encoded = [], [], [], []
            workflow, filename = [], []
            
            
            for key in CACHE:
                times.append(CACHE[key]['tiempo'])

            evict_time=np.max(times)

            
            for key in CACHE:
                workflow.append(CACHE[key]['workflow'])
                
                filename.append(key)
                filesize.append(CACHE[key]['Size'])

                d_label_encoded.append(CACHE[key]['d_label_encoded'])
               
                total_accesses.append(CACHE[key]['N_accesses'])
            
                insertion_time.append(CACHE[key]['accesos_anteriores'][-CACHE[key]['N_accesses']])

                last_time.append(np.nan)
                last_2time.append(np.nan)
                last_3time.append(np.nan)
                last_4time.append(np.nan)
                last_5time.append(np.nan)
            
                for ac_a in range(0,len(CACHE[key]['accesos_anteriores'])):
                    if ac_a == 0: last_time[-1]=(evict_time - CACHE[key]['accesos_anteriores'][-1])
                    if ac_a == 1: last_2time[-1]=(evict_time - CACHE[key]['accesos_anteriores'][-2])
                    if ac_a == 2: last_3time[-1]=(evict_time - CACHE[key]['accesos_anteriores'][-3])
                    if ac_a == 3: last_4time[-1]=(evict_time - CACHE[key]['accesos_anteriores'][-4])
                    if ac_a == 4: last_5time[-1]=(evict_time - CACHE[key]['accesos_anteriores'][-5])

            
                deltaT_1_last.append(CACHE[key]['accesos_anteriores'][-1]-CACHE[key]['accesos_anteriores'][-CACHE[key]['N_accesses']])
                recency_1st.append(evict_time - CACHE[key]['accesos_anteriores'][-CACHE[key]['N_accesses']])


            df1 = pd.DataFrame({
                        "filename": filename,
                      "filesize": filesize,
                      "d_label_encoded": d_label_encoded,
                      "workflow": workflow,
                      "Total Accesses":total_accesses,
                      "last read access": last_time,
                      "2nd last read access": last_2time,
                      "3rd last read access": last_3time,
                      "4th last read access": last_4time,
                      "5th last read access": last_5time,
                      "deltaT_1_last": deltaT_1_last,
                      "recency_1st": recency_1st,
                      })

            
            df1_copy = df1.copy()

            log_vars = ['Total Accesses',
                        '5th last read access',
                        '4th last read access',
                        '3rd last read access',
                        '2nd last read access',
                        'last read access',
                        'recency_1st',
                        'deltaT_1_last']
    
            for var in log_vars:
                df1_copy[var] = np.log1p(df1_copy[var])
        
            


            thevars = ['last read access','recency_1st', 'deltaT_1_last', '2nd last read access', '3rd last read access',
                       '4th last read access','5th last read access', 'Total Accesses', 'd_label_encoded']

        
            dtest = np.array(df1_copy[thevars].values)
            dtest = dtest.astype(float)

            m = os.path.join(tree, model_name) if model_name else None

            if m != self.last_model_path:
                self.last_model = XGBClassifier()
                self.last_model.load_model(m)
                self.last_model_path = m
        
            pred = self.last_model.predict_proba(dtest)[:, 1]

            df1_copy.loc[:, 'pred'] = pred
            df1_copy.sort_values(by=['pred'], ascending=[True], inplace = True)
            df1_copy['cumulative_size'] = df1_copy['filesize'].cumsum()

            
            keys_to_include = df1_copy['filename'][df1_copy['cumulative_size'] < size_a_eliminar]
            new_deletions = {key: CACHE[key] for key in keys_to_include if key in CACHE}

            for fi in df1_copy['filename'][df1_copy['cumulative_size'] < size_a_eliminar]:
                del CACHE[fi]

            cache_size = sum([i['Size'] for i in CACHE.values()])
            ndels+=len(df1_copy['filename'][df1_copy['cumulative_size'] < size_a_eliminar])
            # print(dels, cache_size, size_a_eliminar, ndels)
            del_time=evict_time


            for index, row in df1_copy[df1_copy['cumulative_size'] < size_a_eliminar].iterrows():
                str_to_file = str(dels) + "," + str(row['insertion_time']) + "," + str(del_time) + "," + row['filename'] + "\n"
                f.write(str_to_file)

            del df1_copy, df1
            return CACHE, cache_size, new_deletions, dels, ndels
    
    def create_ADDITIONSDF(self,_HistoricoAdditions):
            try:
                ADDITIONSDF = pd.DataFrame(_HistoricoAdditions,columns=['Archivo','Tamano','NAccesses','Miss','TiempoUnix','CacheSize','Dia','Per'])
            except:
                ADDITIONSDF = pd.DataFrame(_HistoricoAdditions,columns=['Archivo','Tamano','NAccesses','Miss','TiempoUnix','CacheSize','Dia'])

            return ADDITIONSDF

    def create_DELETIONSDF(self,_HistoricoDeletions):
        deletions_list = [(data[0], data[1]['Size'], data[1]['N_accesses'], time, dia) for (time,dia), data_tuple in _HistoricoDeletions.items() for data in data_tuple]
        DELETIONSDF = pd.DataFrame(deletions_list,columns=['Archivo','Tamano','NAccesses','TiempoUnix','Dia'])

        return DELETIONSDF

    def create_STATICDIA(self,_HistoricoStaticDia):
        static_list = [(dia, nombre, info_cache["tiempo"], info_cache['N_accesses'], info_cache["Size"], info_cache['accesos_anteriores'], info_cache['accesos_anteriores_dias']) 
                    for elementos_en_un_dia in _HistoricoStaticDia
                    for nombre, dia, info_cache in elementos_en_un_dia]
        STATICDIA = pd.DataFrame(static_list,columns = ["Dia","Archivo","TiempoUnix",'N_accesses','Size','Access_List','Access_List_Dias'])
        STATICDIA['Diferencia_Dias'] = STATICDIA.apply(lambda row: (row['Dia'] - row['Access_List_Dias'][-1]).days, axis=1)

        return STATICDIA

def get_max_mean(obj):
    return obj.max(), obj.mean()

def calculate_parameters(ADDITIONSDF, all_dates):
    
    # TO GET DF SAMPLE
    # print(ADDITIONSDF.columns)
    # print([row.tolist() for (index, row),i in zip(ADDITIONSDF.iterrows(),range(4))])

    hits_byday = ADDITIONSDF[ADDITIONSDF["Miss"] == False].groupby('Dia')
    n_hits = hits_byday.size().reindex(all_dates, fill_value=0)
    size_hits = hits_byday["Tamano"].sum().reindex(all_dates, fill_value=0)/(3600*24)

    miss_byday = ADDITIONSDF[ADDITIONSDF["Miss"]].groupby('Dia')
    n_miss = miss_byday.size().reindex(all_dates, fill_value=0)
    size_miss = miss_byday["Tamano"].sum().reindex(all_dates, fill_value=0)/(3600*24)

    n_hits_miss = n_hits + n_miss
    size_hits_miss = size_hits + size_miss

    cum_hits = (n_hits.cumsum() * 100) / n_hits_miss.cumsum()
    cum_miss = (n_miss.cumsum() * 100) / n_hits_miss.cumsum()

    return_list = []
    for param in [n_hits,size_hits,n_miss,size_miss,n_hits_miss]:
        return_list.append(get_max_mean(param))
    
    return_list += [cum_hits.iloc[-1],cum_miss.iloc[-1]]
    
    return return_list

def main_XGB(path, tree, option):
    d = data()

    if not os.path.exists(path):
        os.makedirs(path)
    
    # get rates and df, and pass it to simulation, then i'll want it with a handler class
    df = d.get_df()

    # Filter 
    data_dist, mc_dist, user_dist = d.get_rates()
    
    cache_sizes_1 = list(range(0,301,20))[1:]
    cache_sizes_2 = list(range(350,501,50))
    cache_sizes_3 = list(range(600,801,100))

    cache_sizes = cache_sizes_1 + cache_sizes_2 + cache_sizes_3

    start_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2024, 5, 1)
  
    c = cache(df, data_dist, mc_dist, user_dist)
            
    # get all possible dates between the two starting dates
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max()).date
    for cs in cache_sizes:


        # check if its already saved
        if os.path.exists(f"{path}/{start_date}_{end_date}_{cs}.pkl"):
            continue
            
        print("Cache",start_date,end_date,cs)
        
        start = time.time()
        ADDITIONSDF = c.run(start_date ,end_date ,path , tree, option ,  size = cs)
        params =  calculate_parameters(ADDITIONSDF, all_dates)
        end = time.time()
            
        print("Cache run time:",end-start)
        print(params)
            
        with open(f"{path}/{start_date}_{end_date}_{cs}.pkl", 'wb') as archivo_pickle:
            pickle.dump((params), archivo_pickle)

