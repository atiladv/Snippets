def import_fund_quota_cvm():
    
    #webscrap
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    import time; import sys
    
    import pandas as pd
    from datetime import datetime
    from funcao_milos import insert_banco
    from c_conection_sql import Con_sql
    import os
    call_engine = Con_sql()
    conn, engine = call_engine.connection()
   
    months = ['12/2006',
              '01/2007', '02/2007', '03/2007', '04/2007', '05/2007', '06/2007', '07/2007', '08/2007', '09/2007', '10/2007', '11/2007', '12/2007',
              '01/2008', '02/2008', '03/2008', '04/2008', '05/2008', '06/2008', '07/2008', '08/2008', '09/2008', '10/2008', '11/2008', '12/2008',
              '01/2009', '02/2009', '03/2009', '04/2009', '05/2009', '06/2009', '07/2009', '08/2009', '09/2009', '10/2009', '11/2009', '12/2009',
              '01/2010', '02/2010', '03/2010', '04/2010', '05/2010', '06/2010', '07/2010', '08/2010', '09/2010', '10/2010', '11/2010', '12/2010',
              '01/2011', '02/2011', '03/2011', '04/2011', '05/2011', '06/2011', '07/2011', '08/2011', '09/2011', '10/2011', '11/2011', '12/2011',
              '01/2012', '02/2012', '03/2012', '04/2012', '05/2012', '06/2012', '07/2012', '08/2012', '09/2012', '10/2012', '11/2012', '12/2012',
              '01/2013', '02/2013', '03/2013', '04/2013', '05/2013', '06/2013', '07/2013', '08/2013', '09/2013', '10/2013', '11/2013', '12/2013',
              '01/2014', '02/2014', '03/2014', '04/2014', '05/2014', '06/2014', '07/2014', '08/2014', '09/2014', '10/2014', '11/2014', '12/2014',
              '01/2015', '02/2015', '03/2015', '04/2015', '05/2015', '06/2015', '07/2015', '08/2015', '09/2015', '10/2015', '11/2015', '12/2015',
              '01/2016', '02/2016', '03/2016', '04/2016', '05/2016', '06/2016', '07/2016', '08/2016', '09/2016', '10/2016', '11/2016', '12/2016',
              '01/2017', '02/2017', '03/2017', '04/2017', '05/2017', '06/2017', '07/2017', '08/2017', '09/2017', '10/2017', '11/2017', '12/2017',
              '01/2018', '02/2018', '03/2018', '04/2018', '05/2018', '06/2018', '07/2018', '08/2018', '09/2018', '10/2018', '11/2018', '12/2018',
              '01/2019', '02/2019', '03/2019', '04/2019', '05/2019', '06/2019', '07/2019', '08/2019', '09/2019', '10/2019', '11/2019', '12/2019',
              '01/2020', '02/2020', '03/2020', '04/2020', '05/2020', '06/2020', '07/2020', '08/2020', '09/2020', '10/2020', '11/2020', '12/2020',
              '01/2021', '02/2021', '03/2021', '04/2021', '05/2021']

    months = ['06/2021', '07/2021', '08/2021']
    
    dados = pd.DataFrame(columns=['client', 'url_fund', 'months'])
    dados.loc[len(dados)+1] = ['GOLD FIELDS FIA IE', 'https://cvmweb.cvm.gov.br/SWB/Sistemas/SCW/CPublica/InfDiario/CPublicaInfDiario.aspx?PK_PARTIC=192324&COMPTC=', ['02/2021']]
    dados.loc[len(dados)+1] = ['ALPINE FIM', 'https://cvmweb.cvm.gov.br/SWB/Sistemas/SCW/CPublica/InfDiario/CPublicaInfDiario.aspx?PK_PARTIC=178380&COMPTC=', ['02/2021']]
    # dados.loc[len(dados)+1] = ['ESH THETA', 'https://cvmweb.cvm.gov.br/SWB/Sistemas/SCW/CPublica/InfDiario/CPublicaInfDiario.aspx?PK_PARTIC=146985&COMPTC=', months]
    # dados.loc[len(dados)+1] = ['CENTAURO', 'https://cvmweb.cvm.gov.br/SWB/Sistemas/SCW/CPublica/InfDiario/CPublicaInfDiario.aspx?PK_PARTIC=67059&COMPTC=', months]
    
    for index, row in dados.iterrows():
    
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            chr_d = dir_path+r'/chromedriver.exe'
            chr_d = r'/Users/RR/chromedriver/chromedriver'
            browser = webdriver.Chrome(executable_path = chr_d)
        except:
            browser = webdriver.Chrome()
        
        fund = row['client']
        url_fund = row['url_fund']
    
        # initial webscrap
        url = 'https://cvmweb.cvm.gov.br/SWB/Sistemas/SCW/CPublica/FormBuscaPartic.aspx?TpConsulta=3'
        browser.get(url)
        WebDriverWait(browser, 1)
        time.sleep(1)
        browser.find_element_by_xpath('//*[@id="txtCNPJNome"]').send_keys('esh')
        browser.find_element_by_id('btnContinuar').click()
        browser.get(url_fund)    
        
        for month in months:
            print(fund, ' ',month)
            
            browser.find_element_by_xpath("//select[@name='ddComptc']/option[text()='" + month + "']").click()
            time.sleep(3)
            source = browser.page_source    
            tables = pd.read_html(source, header = 0, decimal=',', thousands='.')[1]
            tables = tables[tables['Quota(R$)'].notna()]
            tables = tables.rename(columns={'Dia' : 'timestamp','Quota(R$)': 'quota', 'Captação no Dia(R$)' : 'aum_in','Resgate no Dia(R$)' : 'aum_out','Total da Carteira(R$)':'aum','N°. Total deCotistas' : 'shareholders'})    
            del tables['Patrimônio Líquido(R$)']
            del tables['Data da próximainformação do PL']
            tables['timestamp']=tables['timestamp'].astype(str)+'/'+month
            tables['timestamp']=tables['timestamp'].apply(lambda x: datetime.strptime(str(x), '%d/%m/%Y').strftime('%Y-%m-%d %H:%M'))
            tables['fund'] = fund
        
            conn.execute("delete from fund_quota where fund = {} and timestamp in({})".format(repr(fund), ",".join([repr(r) for r in tables['timestamp']])))
            query = insert_banco(tables, 'fund_quota')
            conn.execute(query)
    
        sys.stdout.write('\r' + 'got fund data from cvm:' + fund)
        browser.close()
    conn.close()

def fund_risk_measures():

    import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    
    from ffn import PerformanceStats
    from ffn import GroupStats
    import numpy as np
    import pandas as pd
    import os
    from datetime import datetime
    import seaborn as sns; sns.set()
    
    #linear regression
    from sklearn.linear_model import LinearRegression
    
    #database connection
    from c_conection_sql import Con_sql
    con = Con_sql(); conn,execute=con.connection()
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    def plot_heatmap(data, title='', show_legend=True,
                     show_labels=True, label_fmt='.2f',
                     vmin=None, vmax=None,
                     figsize=None, label_color='k',
                     cmap='RdBu', **kwargs):
        """
        Plot a heatmap using matplotlib's pcolor. Args: * data (DataFrame): DataFrame to plot. Usually small matrix, * title (string): Plot title, * show_legend (bool): Show color legend, * show_labels (bool): Show value labels, * label_fmt (str): Label format string, * vmin (float): Min value for scale, * vmax (float): Max value for scale, * cmap (string): Color map, * kwargs: Passed to matplotlib's pcolor
        """
        figsize=(18, 10)
        fig, ax = plt.subplots(figsize=figsize)
    
        heatmap = ax.pcolor(data, vmin=vmin, vmax=vmax, cmap=cmap)
        # for some reason heatmap has the y values backwards....
        ax.invert_yaxis()
    
        if title is not None:
            plt.title(title)
    
        if show_legend:
            fig.colorbar(heatmap)
    
        if show_labels:
            vals = data.values
            for x in range(data.shape[0]):
                for y in range(data.shape[1]):
                    plt.text(x + 0.5, y + 0.5, format(vals[y, x], label_fmt),
                             horizontalalignment='center',
                             verticalalignment='center',
                             color=label_color)
    
        plt.yticks(np.arange(0.25, len(data.index), 1), data.index)
        plt.xticks(np.arange(0.5, len(data.columns), 1), data.columns, rotation=0)
    
        name = 'correlation' + '.jpg'
        fig.savefig(dir_path + '/2_DB/9_DailyFiles/' + name, bbox_inches='tight')
    
        return plt
    
    def plot_corr_heatmap_2(data, **kwargs):
        """
        Plots the correlation heatmap for a given DataFrame.
        """
        return plot_heatmap(data.corr(), vmin=-1, vmax=1, **kwargs, figsize=None)
    
    def plot_correlation(self, freq=None, title=None,
                             figsize=(20, 18), **kwargs):
            """
            Utility function to plot correlations. Args:
                * freq (str): Pandas data frequency alias string, * title (str): Plot title, * figsize (tuple (x,y)): figure size, * kwargs: passed to Pandas' plot_corr_heatmap function
            """
            if title is None:
                title = self._get_default_plot_title(
                    freq, 'Return Correlation Matrix')
    
            rets = self._get_series(freq).to_returns().dropna()
            return rets.plot_corr_heatmap(title=title, figsize=figsize, **kwargs)
    
    def to_drawdown_series(prices):
        """
        When prices are below high water marks, the drawdown series = current / hwm - 1. The max drawdown can be obtained by simply calling .min() on the result (since the drawdown series is negative)
        """
        # make a copy so that we don't modify original data
        drawdown = prices.copy()
    
        # Fill NaN's with previous values
        drawdown = drawdown.fillna(method='ffill')
    
        # Ignore problems with NaN's in the beginning
        drawdown[np.isnan(drawdown)] = -np.Inf
    
        # Rolling maximum
        roll_max = np.maximum.accumulate(drawdown)
        drawdown = drawdown / roll_max - 1.
        return drawdown
    
    # ====
    # Path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)  
    
    end_date = int(datetime.now().strftime("%Y%m%d"))
    end_date_stmp = datetime.now().strftime("%Y-%m-%d")
    
    # funds = ['GOLD FIELDS FIA IE', 'ALPINE FIM', 'ESH THETA', 'CENTAURO']
    funds = ['ALPINE FIM']
    
    # Get fund quota
    qry=conn.execute('SELECT timestamp, fund, quota FROM fund_quota WHERE fund in ({})'.format(",".join([repr(r) for r in funds])))
    df_fund= pd.DataFrame(qry, columns = qry.keys())
    df_fund['timestamp']=df_fund['timestamp'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
    
    df_fund_adj = df_fund.drop_duplicates(subset='timestamp')['timestamp']
    df_fund_adj = pd.DataFrame(df_fund_adj, columns = ['timestamp'])
    for fund in df_fund['fund'].drop_duplicates():
        df_fund_adj_temp = df_fund.loc[(df_fund['fund'] == fund)][['timestamp', 'quota']]
        df_fund_adj_temp.columns = ['timestamp', fund]
        df_fund_adj = pd.merge (df_fund_adj, df_fund_adj_temp, how= 'inner', on = 'timestamp')
    df_fund_adj.index = df_fund_adj['timestamp']
    del df_fund_adj['timestamp']
    df_fund_adj = df_fund_adj.dropna(axis='columns')
    
    df_fund.index = df_fund['timestamp']
    del df_fund['timestamp']
    
    #dates
    start_date = int(pd.to_datetime(df_fund.index.min()).strftime('%Y%m%d'))
    start_date_stmp = str(pd.to_datetime(df_fund.index.min()).strftime('%Y-%m-%d'))
    
    # Get BOV    
    stocks = 'BOVA11'
    #conexão com o banco
    table = 'b3_vis_fixing_hist'
    qry2='SELECT tradedate as timestamp, avg_adj as BOVA11 FROM {} WHERE symbol = {} and (tradedate >= {} and tradedate <= {}) order by tradedate'.format(table, repr(stocks), start_date, end_date)
    qry=conn.execute(qry2)
    df_bova11= pd.DataFrame(qry)
    if len(df_bova11)>0:
        df_bova11.columns = [i for i in qry.keys()]
        #formatação da data para tratamento
        df_bova11['timestamp']=df_bova11['timestamp'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d'))
        df_bova11.index = df_bova11['timestamp']
        del df_bova11['timestamp']
        df_bova11.columns = ['BOVA11']
            
    # Get CDI
    table = 'b3_cdi_hist'
    primary_key = 'tradedate'
    qry=conn.execute('SELECT tradedate as timestamp, yield FROM {} WHERE {} between {} and {} ORDER BY {} ASC'.format(table, primary_key, start_date, end_date, primary_key))
    df_CDI= pd.DataFrame(qry)
    if len(df_CDI)>0:
        df_CDI.columns = [i for i in qry.keys()]
        df_CDI['timestamp']=df_CDI['timestamp'].astype(int)
        df_CDI['yield'] = 1 + df_CDI['yield']/100
        df_CDI['cdi_factor'] = df_CDI['yield']**(1/252)
        df_CDI['CDI'] = df_CDI['cdi_factor'].cumprod()
        df_CDI['timestamp']=df_CDI['timestamp'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d'))
        df_CDI.index = df_CDI['timestamp']
        del df_CDI['timestamp']
        del df_CDI['yield']
        del df_CDI['cdi_factor']
        rfr = df_CDI['CDI'].max()**(252/len(df_CDI['CDI']))-1
        
    # Get benchmark
    qry=conn.execute('SELECT timestamp, value, index FROM benchmarks WHERE timestamp between {} and {}'.format(repr(start_date_stmp), repr(end_date_stmp)))
    df_bench = pd.DataFrame(qry, columns = qry.keys())
    df_bench['timestamp']=df_bench['timestamp'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
    df_bench = df_bench[df_bench['index'] == 'IHFA']
    df_bench_adj = df_bench.drop_duplicates(subset='timestamp')['timestamp']
    df_bench_adj = pd.DataFrame(df_bench_adj, columns = ['timestamp'])
    for index in df_bench['index'].drop_duplicates():
        df_bench_adj_temp = df_bench.loc[(df_bench['index'] == index)][['timestamp', 'value']]
        df_bench_adj_temp.columns = ['timestamp', index]
        df_bench_adj = pd.merge (df_bench_adj, df_bench_adj_temp, how= 'inner', on = 'timestamp')
    df_bench_adj.index = df_bench_adj['timestamp']
    del df_bench_adj['timestamp']
    df_bench_adj = df_bench_adj.dropna(axis='columns')
        
    conn.close()
        
    #merge Alpine and CDI
    prices = pd.merge(df_fund_adj, df_CDI, how='outer', left_index=True, right_index=True).fillna(method='ffill')
    prices = prices.rename(columns = {'quota':fund})
    
    #merge Alpine and BOVA11
    prices = pd.merge(prices, df_bova11, how='outer', left_index=True, right_index=True).fillna(method='ffill')
    
    #merge Alpine and benchs
    prices = pd.merge(prices, df_bench_adj, how='outer', left_index=True, right_index=True).fillna(method='ffill')
    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(axis='rows')
    prices.index.name = 'Date'
    
    returns = np.log(prices / prices.shift(1)).dropna()
    
    #stats
    perf = prices.calc_stats()
    perf.set_riskfree_rate(round(float(rfr), 2))
    perf.display()
    print()
    perf[fund].display_monthly_returns()
    
    # =====
    # Plots
    
    # Hist Performance
    fig = plt.figure(figsize=(18, 10))
    rebase = prices / prices.iloc[0] * 100
    plt.plot(rebase.index, rebase)
    plt.xlabel('')
    plt.ylabel('Rebase 100')
    plt.legend(loc = "lower right")
    import matplotlib.lines as mlines
    
    blue_line = mlines.Line2D([], [], color='blue', label=rebase.columns[0])
    oranges_line = mlines.Line2D([], [], color='orange', label=rebase.columns[1])
    greens_line = mlines.Line2D([], [], color='green', label=rebase.columns[2])
    reds_line = mlines.Line2D([], [], color='red', label=rebase.columns[3])
    plt.legend(handles=[blue_line, oranges_line, greens_line, reds_line])
    
    name = 'hist_performance' + '.jpg'
    plt.savefig(dir_path + '/2_DB/9_DailyFiles/' + name, bbox_inches='tight')
    
    ### Plot Drawdown
    drawdown_series = to_drawdown_series(prices)*100
    fig = plt.figure(figsize=(18, 10))
    
    color_y_n = 'Y'
    
    # uncolored drawdown
    if color_y_n == 'N':        
        for column in drawdown_series.drop(funds, axis=1):
            plt.plot(drawdown_series.index, drawdown_series[column], marker='', color='grey', linewidth=1, alpha=0.4)
    else:
        plt.plot(drawdown_series.index, drawdown_series['CDI'], marker='', color='orange', linewidth=1, alpha=0.4)
        plt.plot(drawdown_series.index, drawdown_series['BOVA11'], marker='', color='green', linewidth=1, alpha=0.4)
        plt.plot(drawdown_series.index, drawdown_series['IHFA'], marker='', color='red', linewidth=1, alpha=0.4)
     
    # Now re do the interesting curve, but biger with distinct color
    plt.legend(handles=[blue_line, oranges_line, greens_line, reds_line])
    plt.plot(drawdown_series.index, drawdown_series[funds], marker='', color='blue', linewidth=2, alpha=0.7)       
    plt.ylabel("Drawdown (%)")
    name = 'drawdown' + '.jpg'
    fig.savefig(dir_path + '/2_DB/9_DailyFiles/' + name, bbox_inches='tight')
        
    ### plot sharp / efficient frontier
    df_sharpe = pd.DataFrame(perf.stats)
    sharpe_arr = df_sharpe.loc[df_sharpe.index=='daily_sharpe'].astype(float).to_numpy().T
    ret_arr = 100*df_sharpe.loc[df_sharpe.index=='daily_mean'].astype(float).to_numpy().T
    vol_arr = 100*df_sharpe.loc[df_sharpe.index=='daily_vol'].astype(float).to_numpy().T
    names_arr = df_sharpe.columns.to_numpy().T
    
    fig = plt.figure(figsize=(18, 8))
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Annual Return (%)')
    
    for x in  df_sharpe.columns:
        #outline one fund
        fund_sr_ret = 100*df_sharpe.at['daily_mean', x]
        fund_sr_vol = 100*df_sharpe.at['daily_vol', x]
        plt.annotate(x, # this is the text
                      (fund_sr_vol, fund_sr_ret), # this is the point to label
                      textcoords="offset points", # how to position the text
                      xytext=(0,10), # distance from text to points (x,y)
                      ha='center') # horizontal alignment can be left, right or center
    plt.show()
    
    name = 'sharpe' + '.jpg'
    fig.savefig(dir_path + '/2_DB/9_DailyFiles/' + name, bbox_inches='tight')
    
    #plot correlations
    fig = plt.figure(figsize=(25, 25))
    plot_corr_heatmap_2(returns)
    
    # Correlation
    corr = returns.corr().as_format('.2f')
    corr_list = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr_list.columns = ['x', 'y', 'correlation']
    
    # Calculate sum of correlations
    global correl_sum
    corr_list['correlation'] = corr_list['correlation'].astype(float)
    correl_sum = corr_list
    correl_sum = correl_sum.drop(correl_sum[(correl_sum.x == correl_sum.y)].index)
    correl_sum = correl_sum.groupby(['y'])['correlation'].mean().reset_index()
    correl_sum['correlation'] = correl_sum['correlation'].as_format('.2f')
    # rename_funds = pd.DataFrame()
    # rename_funds['y'] = old_columns
    # rename_funds['adjusted'] = new_columns
    # correl_sum = pd.merge (correl_sum, rename_funds, how='left', on = 'y')
    # del correl_sum['y']
    
    # #hide funds names
    # old_columns = returns.columns
    # i=0
    # new_columns = []
    # for x in old_columns:
    #     if x not in ['ALPINE', 'CDI', 'BOVA11', 'IHFA']:
    #       x = 'Fund ' + str(i)
    #     new_columns.append(x)
    #     i = i + 1
    # returns.columns = new_columns
    # returns.columns = old_columns
    
    # #Plot histogram
    # from matplotlib.ticker import PercentFormatter
    # min_range = returns.min().min()
    # max_range = returns.max().max()
    # ax = returns.hist(figsize=(12, 10), range = [-.1, .1], weights=np.ones(len(returns)) / len(returns))
    # ax = returns[['ALPINE']].hist(figsize=(12, 5), weights=np.ones(len(returns)) / len(returns))
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
    #export to excel
    df_excel = pd.DataFrame(perf.stats)
    cur_path=dir_path+'/2_DB/4_xls'
    file = cur_path + '/Funds_Comparison.xlsx'
    
    from openpyxl import load_workbook
    
    #read the existing sheets so that openpyxl won't create a new one later
    book = load_workbook(file)
    file = cur_path + '/Funds_Comparison.xlsx'
    writer = pd.ExcelWriter(file, engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    #update without overwrites
    df_excel.to_excel(writer, "Table", startrow=1, startcol=1, header=True, index = True)
    writer.save()
    
if __name__ == '__main__':
     import_fund_quota_cvm()
     fund_risk_measures()