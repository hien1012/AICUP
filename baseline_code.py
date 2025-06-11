from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
import xgboost as xgb
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
# 給一組實部、虛部訊號，計算FFT
# 回傳 (n, xreal, ximag)：n為長度，xreal/ximag為經過FFT的數列
def FFT(xreal, ximag):
    n = 2
    while(n*2 <= len(xreal)):
        n *= 2
    
    p = int(math.log(n, 2))
    
    for i in range(0, n):
        a = i
        b = 0
        for j in range(0, p):
            b = int(b*2 + a%2)
            a = a/2
        if(b > i):
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]
            
    wreal = []
    wimag = []
        
    arg = float(-2 * math.pi / n)
    treal = float(math.cos(arg))
    timag = float(math.sin(arg))
    
    wreal.append(float(1.0))
    wimag.append(float(0.0))
    
    for j in range(1, int(n/2)):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)
        
    m = 2
    while(m < n + 1):
        for k in range(0, n, m):
            for j in range(0, int(m/2), 1):
                index1 = k + j
                index2 = int(index1 + m / 2)
                t = int(n * j / m)
                treal = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal
                ximag[index1] = uimag + timag
                xreal[index2] = ureal - treal
                ximag[index2] = uimag - timag
        m *= 2
        
    return n, xreal, ximag   
    
# 把原始加速度/陀螺儀資料切成每個揮拍區間
# 分別算出該區間的平均值
# 回傳兩組平均值list
def FFT_data(input_data, swinging_times):   
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength
       
    for num in range(len(swinging_times)-1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num+1]):
            a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2)))

        a_mean[num] = (sum(a) / len(a))
        g_mean[num] = (sum(a) / len(a))
    
    return a_mean, g_mean


# 對單一揮拍區間（單一段input_data）計算所有特徵
# 包含均值、方差、均方根、最大/最小值、FFT、PSD、峭度、偏態、熵等
# 算完寫一行到 csv（writer.writerow(output)）
def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    allsum = []
    mean = []
    var = []
    rms = []
    XYZmean_a = 0
    a = []
    g = []
    a_s1 = 0
    a_s2 = 0
    g_s1 = 0
    g_s2 = 0
    a_k1 = 0
    a_k2 = 0
    g_k1 = 0
    g_k2 = 0
    
    for i in range(len(input_data)):
        if i==0:
            allsum = input_data[i]
            a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
            continue
        
        a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
        g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
       
        allsum = [allsum[feature_index] + input_data[i][feature_index] for feature_index in range(len(input_data[i]))]
        
    mean = [allsum[feature_index] / len(input_data) for feature_index in range(len(input_data[i]))]
    
    for i in range(len(input_data)):
        if i==0:
            #var = input_data[i]
            #rms = input_data[i]
            var = [0] * len(input_data[0])
            rms = [0] * len(input_data[0])
            continue

        var = [var[feature_index] + math.pow((input_data[i][feature_index] - mean[feature_index]), 2) for feature_index in range(len(input_data[i]))]
        rms = [rms[feature_index] + math.pow(input_data[i][feature_index], 2) for feature_index in range(len(input_data[i]))]
        
    var = [math.sqrt((var[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    rms = [math.sqrt((rms[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    
    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a)]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g)]
    
    a_var = math.sqrt(math.pow((var[0] + var[1] + var[2]), 2))
    
    for i in range(len(input_data)):
        a_s1 = a_s1 + math.pow((a[i] - a_mean[0]), 4)
        a_s2 = a_s2 + math.pow((a[i] - a_mean[0]), 2)
        g_s1 = g_s1 + math.pow((g[i] - g_mean[0]), 4)
        g_s2 = g_s2 + math.pow((g[i] - g_mean[0]), 2)
        a_k1 = a_k1 + math.pow((a[i] - a_mean[0]), 3)
        g_k1 = g_k1 + math.pow((g[i] - g_mean[0]), 3)
    
    a_s1 = a_s1 / len(input_data)
    a_s2 = a_s2 / len(input_data)
    g_s1 = g_s1 / len(input_data)
    g_s2 = g_s2 / len(input_data)
    a_k2 = math.pow(a_s2, 1.5)
    g_k2 = math.pow(g_s2, 1.5)
    a_s2 = a_s2 * a_s2
    g_s2 = g_s2 * g_s2
    
    a_kurtosis = [a_s1 / a_s2]
    g_kurtosis = [g_s1 / g_s2]
    a_skewness = [a_k1 / a_k2]
    g_skewness = [g_k1 / g_k2]
    
    a_fft_mean = 0
    g_fft_mean = 0
    cut = int(n_fft / swinging_times)
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = 0
    e4 = 0
    
    for i in range(cut * swinging_now, cut * (swinging_now + 1)):
        a_fft_mean += a_fft[i]
        g_fft_mean += g_fft[i]
        a_psd.append(math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2))
        g_psd.append(math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2))
        e1.append(math.pow(a_psd[-1], 0.5))
        e3.append(math.pow(g_psd[-1], 0.5))
        
    a_fft_mean = a_fft_mean / cut
    g_fft_mean = g_fft_mean / cut
    
    a_psd_mean = sum(a_psd) / len(a_psd)
    g_psd_mean = sum(g_psd) / len(g_psd)
    
    for i in range(cut):
        e2 += math.pow(a_psd[i], 0.5)
        e4 += math.pow(g_psd[i], 0.5)
    
    for i in range(cut):
        entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2))
        entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
    
    a_entropy_mean = sum(entropy_a) / len(entropy_a)
    g_entropy_mean = sum(entropy_g) / len(entropy_g)       
        
    
    output = mean + var + rms + a_max + a_mean + a_min + g_max + g_mean + g_min + [a_fft_mean] + [g_fft_mean] + [a_psd_mean] + [g_psd_mean] + a_kurtosis + g_kurtosis + a_skewness + g_skewness + [a_entropy_mean] + [g_entropy_mean]
    writer.writerow(output)

# 對每個 .txt
# 讀入資料，分揮拍段，呼叫 feature 寫出特徵
# 存成一個csv（每一段一行，一檔27行）
def data_generate():
    datapath = './test_data'
    tar_dir = 'tabular_data_test'
    pathlist_txt = Path(datapath).glob('**/*.txt')

    
    for file in pathlist_txt:
        f = open(file)

        All_data = []

        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)
        
        f.close()

        swing_index = np.linspace(0, len(All_data), 28, dtype = int)
        # filename.append(int(Path(file).stem))
        # all_swing.append([swing_index])

        headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']                
        

        with open('./{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i==0:
                        continue
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
            except:
                #print(Path(file).stem)
                print(f"Error in file: {Path(file).stem}")
                print(traceback.format_exc())
                continue

def force_sum_one(arr, ndigits=4):
    arr = np.round(arr, ndigits)  # 先四捨五入
    arr_new = arr.copy()
    for i, row in enumerate(arr):
        s = np.sum(arr[i, :-1])
        # 修正最後一項，使總和恰好是1，且小數位不超過ndigits
        arr_new[i, -1] = np.round(1 - s, ndigits)
    return arr_new

def main():
    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    data_generate()
    
    # --- 讀取 train_info 與 tabular_data_train ---
    ###
    exclude_ids = {2174, 2182, 2242, 2306, 2428, 2703, 2728, 2761, 2804, 2877, 2951, 2965, 3107, 3277}
    all_ids = np.arange(1968, 3412)
    valid_ids = [uid for uid in all_ids if uid not in exclude_ids]
    submission = pd.DataFrame({
        'unique_id': valid_ids,
        'gender': np.nan,
        'hold racket handed': np.nan,
        'play years_0': np.nan,
        'play years_1': np.nan,
        'play years_2': np.nan,
        'level_2': np.nan,
        'level_3': np.nan,
        'level_4': np.nan,
        'level_5': np.nan
    })
    submission.to_csv('submission.csv', index=False)
    ###
    
    info = pd.read_csv('train_info.csv')
    unique_players = info['player_id'].unique()#從 info 裡擷取出所有不重複的 player_id，確保每個人只出現一次。
    unique_gender = info['gender'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)  #把所有 player_id 隨機分成 8:2 訓練/測試兩組。
                                                                                                    #防止同一位玩家同時出現在訓練和測試裡，random_state 固定隨機種子，讓每次切分都一樣。
    train_players_gender, test_players_gender = train_test_split(unique_players, test_size=0.3, random_state=42)
    train_players_hand, test_players_hand = train_test_split(unique_players, test_size=0.2, random_state=50)
    train_players_year, test_players_year = train_test_split(unique_players, test_size=0.2, random_state=1)
    
    datapath = './tabular_data_train'
    datalist = list(Path(datapath).glob('*.csv'))                                                   #所有特徵csv檔案的清單 tabular_data_train/*.csv
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']                           #任務欄位清單
    
    x_train = pd.DataFrame()                                                                        #初始化訓練/測試集的 feature 和 label dataframe
    y_train = pd.DataFrame(columns=target_mask)                                                     #後續將所有資料依 player_id 累加進去
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)

    x_train_gender = pd.DataFrame()                                                                        #初始化訓練/測試集的 feature 和 label dataframe
    y_train_gender = pd.DataFrame(columns=target_mask)                                                     #後續將所有資料依 player_id 累加進去
    x_test_gender = pd.DataFrame()
    y_test_gender = pd.DataFrame(columns=target_mask)
    
    x_train_hand = pd.DataFrame()                                                                        #初始化訓練/測試集的 feature 和 label dataframe
    y_train_hand = pd.DataFrame(columns=target_mask)                                                     #後續將所有資料依 player_id 累加進去
    x_test_hand = pd.DataFrame()
    y_test_hand = pd.DataFrame(columns=target_mask)
    
    x_train_year = pd.DataFrame()                                                                        #初始化訓練/測試集的 feature 和 label dataframe
    y_train_year = pd.DataFrame(columns=target_mask)                                                     #後續將所有資料依 player_id 累加進去
    x_test_year = pd.DataFrame()
    y_test_year = pd.DataFrame(columns=target_mask)
    
    for file in datalist:                                                       #   依序處理每個特徵檔（每個 unique_id 一個檔）
        unique_id = int(Path(file).stem)                                        #   抓取這個特徵檔對應的 unique_id   
        row = info[info['unique_id'] == unique_id]                              #   依 unique_id 取對應那筆 info，裡面有 player_id 跟四個標籤 (一行train_info.csv) 
        if row.empty:
            print(f"unique_id {unique_id} not found in train_info.csv")
            continue
        player_id = row['player_id'].iloc[0]                                    #   抓該檔案的 player_id
        player_id_2 = row['player_id'].iloc[0]
        player_id_gender = row['player_id'].iloc[0]
        player_id_hand = row['player_id'].iloc[0]
        
        data = pd.read_csv(file)                                                #   data : 讀入這個 unique_id 的 27*34 特徵表 (每個特徵檔案) 
        
        target = row[target_mask]                                               #   取這個 unique_id train_info 的四個任務標籤
        target_repeated = pd.concat([target] * len(data))                       #   這四個標籤 要重複27次（因為每檔27行） ==>27筆['gender', 'hold racket handed', 'play years', 'level'] 資料
                                                                                #   相當於data的27筆資料的 label
        if player_id in train_players:                                             
            x_train = pd.concat([x_train, data], ignore_index=True)             #   特徵
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)  #   對應的四個標籤
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)

        #-----------------------------------------------------------#
        if player_id_2 in train_players_year:                                             
            x_train_year = pd.concat([x_train_year, data], ignore_index=True)             #   特徵
            y_train_year = pd.concat([y_train_year, target_repeated], ignore_index=True)  #   對應的四個標籤
        elif player_id_2 in test_players_year:
            x_test_year = pd.concat([x_test_year, data], ignore_index=True)
            y_test_year = pd.concat([y_test_year, target_repeated], ignore_index=True)
            
        if player_id_gender in train_players_gender:                                             
            x_train_gender = pd.concat([x_train_gender, data], ignore_index=True)             #   特徵
            y_train_gender = pd.concat([y_train_gender, target_repeated], ignore_index=True)  #   對應的四個標籤
        elif player_id_gender in test_players_gender:
            x_test_gender = pd.concat([x_test_gender, data], ignore_index=True)
            y_test_gender = pd.concat([y_test_gender, target_repeated], ignore_index=True)
            
        if player_id_hand in train_players_hand:
            x_train_hand = pd.concat([x_train_hand, data], ignore_index=True)             #   特徵
            y_train_hand = pd.concat([y_train_hand, target_repeated], ignore_index=True)
        elif player_id_hand in test_players_hand:
            x_test_hand = pd.concat([x_test_hand, data], ignore_index=True)
            y_test_hand = pd.concat([y_test_hand, target_repeated], ignore_index=True)                       
        #-----------------------------------------------------------#    
    ##-------------------------------------------------------------------##
    datapath_1 = './tabular_data_test'
    datalist_1 = list(Path(datapath_1).glob('*.csv'))
    row_counter = []
    test_1 = pd.DataFrame() 
    for file in datalist_1:                                                     #   依序處理每個特徵檔（每個 unique_id 一個檔）
        unique_id = int(Path(file).stem)
        data = pd.read_csv(file)                                                #   data : 讀入這個 unique_id 的 27*34 特徵表 (每個特徵檔案)
        row_counter.append(data.shape[0]) 
        if data.empty:
            print(f"unique_id {unique_id} not found in test_data.csv")
            continue
        test_1 = pd.concat([test_1, data], ignore_index=True)
    ##--------------------------------------------------------------------##
    
    # --- 標準化與編碼 ---
    scaler = MinMaxScaler()
    #scaler_2 = StandardScaler()
    scaler_2 = MinMaxScaler()
    X_train_scaled_binary = scaler.fit_transform(x_train)
    X_test_scaled_binary = scaler.transform(x_test)
    
    X_train_scaled_year = scaler_2.fit_transform(x_train_year)
    X_test_scaled_year = scaler_2.transform(x_test_year)
    
    X_train_scaled_gender = scaler_2.fit_transform(x_train_gender)
    X_test_scaled_gender = scaler_2.transform(x_test_gender)
    
    X_train_scaled_hand = scaler_2.fit_transform(x_train_hand)
    X_test_scaled_hand = scaler_2.transform(x_test_hand)
    ###
    test_scale_binary = scaler.transform(test_1)
    test_scale_multi = scaler_2.transform(test_1)
    ###
    group_size = 27  # 每個 unique_id 有27筆

    # --- 二元分類 ---
    def model_binary(X_train, y_train, X_test, y_test, target):
        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            device='cuda',
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=10000,        # 樹的數量，預設100，可調大
            max_depth=3,             # 樹的最大深度，預設6
            learning_rate=0.01,      # 學習率，預設0.3，通常設小一點
            subsample=1,           # 每棵樹隨機採樣比例
            reg_lambda=0.7,          # L2 正則化，預設1
            reg_alpha=0.5,           # L1 正則化，預設0
            colsample_bytree=1,    # 每棵樹隨機採樣特徵比例
            random_state=42
        )
        #clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        #clf = LogisticRegression(max_iter=50, random_state=42)
        clf.fit(X_train, y_train)                   #   訓練，clf 變成一個已經訓練好的模型，可以做預測
        y_prob = clf.predict_proba(X_test)[:, 1]    #   預測，y_prob 是一個二維陣列，shape=(n, 2)，取第二類(1)的機率
        y_prob_2 = clf.predict_proba(test_scale_binary)[:, 0]  ####       
        num_groups = len(y_prob) // group_size  #有幾組結果
        num_groups_2 = len(y_prob_2) // group_size  ####
        # group voting
        #if sum(y_prob[:group_size]) / group_size > 0.5: #總和取平均 > 0.5
        #    y_pred = [max(y_prob[i*group_size:(i+1)*group_size]) for i in range(num_groups)] #取27個裡面最大的
        #else:
        #    y_pred = [min(y_prob[i*group_size:(i+1)*group_size]) for i in range(num_groups)] #取27個裡面最小的
        y_pred = [np.mean(y_prob[i*group_size:(i+1)*group_size]) for i in range(num_groups)] #取27個裡面平均的 
        ##--------------------------------------------------------------------------------------------##    
        #if sum(y_prob_2[:group_size]) / group_size > 0.5:
        #    y_pred_2 = [max(y_prob_2[i*group_size:(i+1)*group_size]) for i in range(num_groups_2)]
        #else:
        #    y_pred_2 = [min(y_prob_2[i*group_size:(i+1)*group_size]) for i in range(num_groups_2)]
        y_pred_2 = [np.mean(y_prob_2[i*group_size:(i+1)*group_size])for i in range(num_groups_2)]
        arr = np.round(y_pred_2, 4)
        submission[target] = arr      
        submission.to_csv('submission.csv', index=False, float_format='%.4f')
        ##---------------------------------------------------------------------------------------------##      
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print(f"Binary AUC: {auc_score:.3f}")

    # --- 多元分類 ---#
    def model_multiary(X_train, y_train, X_test, y_test, target):
        num_class = len(np.unique(y_train))
        clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_class,
            tree_method='hist',
            device='cuda',
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=1,
            #reg_lambda=0.5,
            #reg_alpha=0.5,
            colsample_bytree=1,
            random_state=42
        )
        clf.fit(X_train, y_train) #已經訓練完成
        y_prob = clf.predict_proba(X_test)  # shape=(n, num_class)
#--------------------------------------------------------------------------------------#
        y_prob_2 = clf.predict_proba(test_scale_multi)
#--------------------------------------------------------------------------------------#
        num_groups = len(y_prob) // group_size  #分成 num_groups 組，每組27筆
        num_groups_2 = len(y_prob_2) // group_size
        y_prob_agg = []
        y_test_agg = []
        y_prob_agg_2 = []
        for i in range(num_groups):
            group_pred = y_prob[i*group_size:(i+1)*group_size] # i~i+27
            group_true = y_test[i*group_size:(i+1)*group_size]        
            # 取該 group 內最大總和的類別，再選該類別機率最高的那行
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_class)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_prob_agg.append(group_pred[best_instance])
            y_test_agg.append(group_true[0])
              
        # y_prob_agg: shape=(n_groups, num_class)

        #-----------------------------------------------------------------------------------------------------#
        for i in range(num_groups_2):
            group_pred_2 = y_prob_2[i*group_size:(i+1)*group_size]
            class_sums_2 = [sum([group_pred_2[k][j] for k in range(group_size)]) for j in range(num_class)]
            chosen_class_2 = np.argmax(class_sums_2)
            candidate_probs_2 = [group_pred_2[k][chosen_class_2] for k in range(group_size)]
            best_instance_2 = np.argmax(candidate_probs_2)
            y_prob_agg_2.append(group_pred_2[best_instance_2])
                        
        y_prob_agg_2 = np.array(y_prob_agg_2)  # 轉成 numpy array
        y_prob_agg_2 = force_sum_one(y_prob_agg_2, 4)        
        
        if target == 'play years':
            row_sums = y_prob_agg_2.sum(axis=1, keepdims=True)
            y_prob_agg_2 = y_prob_agg_2 / row_sums
            submission['play years_0'] = y_prob_agg_2[:,0]
            submission['play years_1'] = y_prob_agg_2[:,1]
            submission['play years_2'] = y_prob_agg_2[:,2]
            submission.to_csv('submission.csv', index=False, float_format='%.4f')
        elif target == 'level':
            row_sums = y_prob_agg_2.sum(axis=1, keepdims=True)
            y_prob_agg_2 = y_prob_agg_2 / row_sums
            submission['level_2'] = y_prob_agg_2[:,0]
            submission['level_3'] = y_prob_agg_2[:,1]
            submission['level_4'] = y_prob_agg_2[:,2]
            submission['level_5'] = y_prob_agg_2[:,3]    
            submission.to_csv('submission.csv', index=False, float_format='%.4f')        
        #------------------------------------------------------------------------------------------------------#      
        auc_score = roc_auc_score(y_test_agg, y_prob_agg, average='micro', multi_class='ovr')
        print(f"Multiary AUC: {auc_score:.3f}")
        
    # --- 分別處理每個 target ---
    le = LabelEncoder()

    y_train_le_gender = le.fit_transform(y_train_gender['gender'])
    y_test_le_gender = le.transform(y_test_gender['gender'])
    model_binary(X_train_scaled_gender, y_train_le_gender, X_test_scaled_gender, y_test_le_gender,'gender')
    
    y_train_le_hold = le.fit_transform(y_train_hand['hold racket handed'])
    y_test_le_hold = le.transform(y_test_hand['hold racket handed'])
    model_binary(X_train_scaled_hand, y_train_le_hold, X_test_scaled_hand, y_test_le_hold,'hold racket handed')

    #加強與否?#
    y_train_le_years = le.fit_transform(y_train_year['play years'])
    y_test_le_years = le.transform(y_test_year['play years'])
    model_multiary(X_train_scaled_year, y_train_le_years, X_test_scaled_year, y_test_le_years, 'play years')

    y_train_le_level = le.fit_transform(y_train['level'])
    y_test_le_level = le.transform(y_test['level'])
    model_multiary(X_train_scaled_binary, y_train_le_level, X_test_scaled_binary, y_test_le_level, 'level')
    #AUC SCORE: 0.792(gender) + 0.998(hold) + 0.660(years) + 0.822(levels)        
    
if __name__ == '__main__':
    main()