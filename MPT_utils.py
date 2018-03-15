import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import cvxpy as cvx
import numpy as np
from math import *
from time import sleep
import fix_yahoo_finance as yf

pricing_path = "./pricing/"
component_path = "./sector_components/"
min_gross=0.5; max_gross=1; min_w=0; max_w=0.05
sleep_time = 5


def load_companies(benchmark,skiprows,columns): ## el input es el nombre del benchmark, si ignora o no el primer row y el nombre de las columnas que son de interes, primero ticker y luego nombre de la compañia## que ETF SECTOR tiene el benchmark?: calcularlo:
    resultados_benchmark = {}
    resultados_companies = {}
    resultados= pd.DataFrame({})
    directory=component_path+benchmark ## directorio donde se buscará los archivos
    file_list=os.listdir(directory)    ## lista de archivos en el directorio
    files = [f for f in file_list if bool(re.search("-"+benchmark+"-", f))] ## lista de archivos .etf sectors
    for key in files: ## loop que recorre todos los etf sectors
        aux = pd.read_csv(directory+"/"+key,skiprows=skiprows,index_col=columns[0])[columns[1]] ## load companies
        aux=aux.to_frame()
        aux.columns=["company_name"]
        aux.index.name = "ticker"
        aux = aux.reset_index().dropna().set_index('ticker')
        aux["benchmark"]=benchmark ## agrega columna con nombre del benchmark
        aux["etf_sector"]=re.search("\-([a-z]*?)\.csv",key).groups()[0]  # agrega columna con nonmbre del etf sector
        print("loaded: "+ re.search("\-([a-z]*?)\.csv",key).groups()[0])
        resultados=resultados.append(aux)
        resultados_companies[re.search("\-([a-z]*?)\.csv",key).groups()[0]] = aux.index.tolist()
    resultados_benchmark[benchmark]=resultados.etf_sector.unique().tolist()
    return resultados,resultados_benchmark,resultados_companies



def get_pricing(output_name,ticker_list,start_date):
    try:
        #ticker_list=np.array(ticker_list)
        if isinstance(ticker_list, list)==False:
            ticker_list=[ticker_list]
            #ticker_list=np.array([ticker_list])
        px = yf.download(ticker_list, start=start_date)['Adj Close']
        if isinstance(px, pd.DataFrame)==False: 
            px=px.to_frame()
            px.columns=ticker_list
        px.sort_index(ascending=True, inplace=True)
        px.to_csv(output_name)
        print(output_name)
        #return px
    except Exception as err:
        print("Error: {0}, waiting to try again in {1}".format(err, sleep_time))
        print(output_name)
        sleep(sleep_time)


def load_consol_px(universe):
    consol_px=pd.DataFrame()
    for i in universe:
        print(i)
        directory=pricing_path+i ## directorio donde se buscará los archivos
        file_list=os.listdir(directory)    ## lista de archivos en el directorio
        files = [f for f in file_list if bool(re.search("-hold-pricing.csv", f))] 
        for f in files:
            aux=pd.read_csv(directory+"/"+f,index_col="Date")
            #aux.index=aux["Date"]
            ccols = set(consol_px.columns.tolist())
            newcols = set(aux.columns.tolist())
            consol_px = consol_px.merge(aux[list(newcols.difference(ccols))], 
            left_index=True, 
            right_index=True, 
            how='outer')
    return consol_px


def clean_nas(df):
    cols = df.count().sort_values()[df.count().sort_values() < 1].index.tolist()  ## precios con toda la serie nula
    cols1=df.isnull().sum()[(df.isnull().sum()>1)&(df.isnull().sum()<df.shape[0])]## numero de nulos por compañia
    for idx,i in cols1.items(): ## toma la ultimos i renglones y verifica que haya mas de un nulo pero menor que el numero de nulos i
        if(df[idx].tail(i).isnull().sum()>=2): ## se verifica que el tail haya mas de un nulo
            cols=cols+[idx]
    cols = list(set(cols))## precios con al menos las ultimas 3 dias sin precios
    df1 = df.drop(cols, axis=1) ## quitar empresas que tienen valores nulos en todos sus precios
    df1.fillna(method='pad', inplace=True) ## Rellenar los nulos con valores de arriba y abajo
    df1.fillna(method='bfill', inplace=True)
    df1 = df1.applymap(lambda x: max(float(x), 1)) ## si el precio es menor a 1, ponerle 1
    cols = df[cols].isnull().sum()
    return df1,cols

### grafico de comportamientos de returns
def plot_returns(return_vec):
    returns_hist=return_vec.T.values
    plt.plot(returns_hist.T, alpha=.4);
    plt.xlabel('time')
    plt.ylabel('returns')
    plt.title("Comportamiento de returns para cada compañia")
    plt.show()


def Optimization(return_vec,gamma_val):
    weights = np.array([0 for _ in range(len(return_vec.columns))])
    mu = return_vec.mean().values ## vector de return mean
    n = len(mu) ## numero de compañias
    name = return_vec.mean().index.values.tolist() ## nombre de las compañias
    Sigma =  return_vec.cov().values  ## covarianza de los retornos
    # Long only portfolio optimization.
    w = cvx.Variable(n) ## variable a optimizar
    gamma = cvx.Parameter(sign='positive') ## aversion al riesgo
    ret = mu.T*w   # returns
    risk = cvx.quad_form(w, Sigma) # risk
    prob = cvx.Problem(cvx.Maximize(ret - gamma*risk),[cvx.sum_entries(w) >= min_gross,cvx.sum_entries(w) <= max_gross, w > min_w, w < max_w])# ,
    gamma.value = gamma_val
    prob.solve()
    if prob.status == 'optimal': 
        weights =[i[0] for i in w.value.tolist()]
    return sqrt(risk.value),ret.value,weights,mu,Sigma,name



def plot_Optimization(risk_data,ret_data,gamma_vals,sharpe,name,Sigma,mu):
    opt_gamma = [sharpe.tolist().index(i) for i in sorted(sharpe)[-1:]][0]
    n = len(mu)
    #markers_on=[sharpe.tolist().index(i) for i in top]
    #markers_on = [sharpe.argmax()]#+[63]
    #markers_on = np.random.randint(100, size=10).tolist()+[sharpe.argmax()]
    #markers_on= markers_on.tolist()
    a=list(range(0, len(gamma_vals))) 
    markers_on=a[0::10]+[opt_gamma] ## toma muestra cada posicion 10 de la gamma junto con la gamma optima
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    plt.plot(risk_data, ret_data, 'g-') #g-
    for marker in markers_on:
        plt.plot(risk_data[marker], ret_data[marker], 'bs')
        ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker], ret_data[marker]-.0009))
        ax.annotate("s="+str(round(sharpe[marker],2)), xy=(risk_data[marker], ret_data[marker]))
        #ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
    #fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(n):
        plt.plot(sqrt(Sigma[i,i]), mu[i], 'ro')
        ax.annotate(name[i], xy=(sqrt(Sigma[i,i]), mu[i]-.0003))
    #plt.suptitle('Opt con lb=20')
    plt.xlabel('Standard deviation')
    plt.ylabel('Return')
    plt.title(r"$\gamma = %.2f$" % gamma_vals[opt_gamma] + ";"+ "s="+str(round(sharpe[opt_gamma],2)))
    plt.show()
