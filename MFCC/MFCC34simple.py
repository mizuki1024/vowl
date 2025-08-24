# MFCC4,5のみを使用した母音認識システム
# === ライブラリのインポート ===
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from time import sleep
import queue

# === フォント設定（Mac対応・IPA記号対応） ===
# IPA記号が表示できるフォントを設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Lucida Grande', 'AppleGothic']

# === 音声処理設定 ===
RATE = 16000  # サンプリングレート（16kHz）
DURATION = 1.0  # 録音時間（秒）
N_MFCC = 13  # MFCCの次元数
RECORDINGS_DIR = "recordings_formant"  # 録音ファイルの保存先
JAPANESE_VOWELS = ['a', 'i', 'u', 'e', 'o']  # 日本語母音
ENGLISH_VOWELS = ['æ', 'ɪ', 'ʊ', 'ɛ', 'ɔ', 'ʌ', 'ɑ']  # 英語母音（cat, sit, put, get, caught, but, father）
VOWELS = JAPANESE_VOWELS + ENGLISH_VOWELS  # 全ての対象母音
SAMPLES_PER_VOWEL = 3  # 母音ごとのサンプル数

# 類似度判定のパラメータ
VERY_SIMILAR_THRESHOLD = 0.85  # 「すごく近い」と判断する類似度閾値

# === 母音の色マッピング ===
COLOR_MAP = {
    # 日本語母音
    'a': 'red',
    'i': 'blue', 
    'u': 'green',
    'e': 'purple',
    'o': 'orange',
    # 英語母音
    'æ': 'darkred',     # cat
    'ɪ': 'darkblue',    # sit
    'ʊ': 'darkgreen',   # put
    'ɛ': 'darkorchid',  # get
    'ɔ': 'darkorange',  # caught
    'ʌ': 'brown',       # but
    'ɑ': 'maroon',      # father
    'ə': 'gray'         # 曖昧母音（シュワー）
}



# === 母音ごとの発音アドバイス ===
ADVICE_MAP = {
    # 日本語母音
    'a': "口を大きく縦に開け、舌は下に落としましょう。",
    'i': "口を横に引いて、舌は前に出すようにしましょう。",
    'u': "唇をすぼめて、舌を後ろに引きましょう。",
    'e': "口角を少し上げ、舌をやや前に出しましょう。",
    'o': "唇を丸く突き出し、舌を後ろに引きましょう。",
    # 英語母音
    'æ': "口を横に大きく開き、舌を低く前に。'cat'の音。",
    'ɪ': "口をやや開き、舌を中央に。'sit'の短い音。",
    'ʊ': "唇を軽く丸め、舌を中央後方に。'put'の音。",
    'ɛ': "口を中程度に開き、舌を中央前方に。'get'の音。",
    'ɔ': "唇を丸く開き、舌を低く後方に。'caught'の音。",
    'ʌ': "口をリラックスして開き、舌を中央に。'but'の音。",
    'ɑ': "口を大きく開き、舌を低く後方に。'father'の音。",
    'ə': "口と舌をリラックスさせ、力を抜いて発音しましょう。"
}

# === フォルマント抽出機能 ===
def extract_formants(y, sr, n_formants=3):
    """音声波形からフォルマント周波数を抽出する（LPC分析使用）"""
    from scipy.signal import freqz
    from scipy.signal.windows import hamming
    
    # 音声信号の前処理
    # プリエンファシス（高周波成分の強調）
    pre_emphasis = 0.97
    emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    
    # フレーム分割のパラメータ
    frame_size = int(0.025 * sr)  # 25msのフレーム
    frame_shift = int(0.010 * sr)  # 10msのシフト
    
    formants_list = []
    
    # フレームごとに処理
    for start in range(0, len(emphasized) - frame_size, frame_shift):
        frame = emphasized[start:start + frame_size]
        
        # ハミング窓を適用
        windowed = frame * hamming(len(frame))
        
        # LPC分析
        # LPC次数は (サンプリングレート/1000) + 2 が目安
        lpc_order = int(sr / 1000) + 4
        
        try:
            # 自己相関法によるLPC係数の計算
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Levinson-Durbin再帰でLPC係数を計算
            lpc_coeffs = solve_lpc(autocorr, lpc_order)
            
            # LPCスペクトル包絡を計算
            w, h = freqz([1], lpc_coeffs, worN=8192, fs=sr)
            
            # スペクトル包絡からピークを検出
            magnitude = 20 * np.log10(np.abs(h) + 1e-15)
            
            # ピーク検出（フォルマント候補）
            peaks, _ = find_peaks(magnitude, distance=int(300/(sr/len(w))))
            
            # ピークの周波数を取得
            peak_freqs = w[peaks]
            
            # 有効なフォルマント範囲でフィルタリング（200Hz-5000Hz）
            valid_peaks = [(f, magnitude[peaks[i]]) for i, f in enumerate(peak_freqs) 
                          if 200 < f < 5000]
            
            # 強度でソートして上位n個を選択
            valid_peaks.sort(key=lambda x: x[1], reverse=True)
            
            # 周波数でソート（低い順）してフォルマントとする
            if valid_peaks:
                formant_freqs = sorted([f for f, _ in valid_peaks[:n_formants*2]])[:n_formants]
            else:
                formant_freqs = []
            
            # 必要な数に満たない場合は0で埋める
            formant_freqs = formant_freqs + [0] * (n_formants - len(formant_freqs))
            
        except Exception:
            # エラーが発生した場合はデフォルト値
            formant_freqs = [0] * n_formants
        
        formants_list.append(formant_freqs)
    
    # 全フレームのフォルマントの中央値を計算
    formants_array = np.array(formants_list)
    # 0以外の値のみで中央値を計算
    median_formants = []
    for i in range(n_formants):
        valid_values = formants_array[:, i][formants_array[:, i] > 0]
        if len(valid_values) > 0:
            median_formants.append(np.median(valid_values))
        else:
            median_formants.append(0)
    
    return np.array(median_formants)

def solve_lpc(autocorr, order):
    """Levinson-Durbin再帰によるLPC係数の計算"""
    # 初期化
    error = autocorr[0]
    lpc = np.zeros(order + 1)
    lpc[0] = 1.0
    
    for i in range(1, order + 1):
        # 反射係数の計算
        k = -np.sum(lpc[:i] * autocorr[i:0:-1]) / error
        
        # LPC係数の更新
        lpc_temp = lpc.copy()
        lpc[i] = k
        for j in range(1, i):
            lpc[j] = lpc_temp[j] + k * lpc_temp[i - j]
        
        # 予測誤差の更新
        error *= (1 - k * k)
        
        if error <= 0:
            break
    
    return lpc


# === 音声ファイルからMFCC特徴量とフォルマントを抽出 ===
def extract_features():
    X_mfcc, X_formants, y = [], [], []
    all_formants = {}  # 母音ごとの個別サンプルのフォルマントを保存
    
    # 日本語母音の処理
    for vowel in JAPANESE_VOWELS:
        all_formants[vowel] = []  # この母音のサンプルフォルマントリスト
        
        file_prefix = vowel
        
        samples_found = 0
        for i in range(1, SAMPLES_PER_VOWEL + 1):
            filepath = os.path.join(RECORDINGS_DIR, f"{file_prefix}_{i}.wav")
            if os.path.exists(filepath):
                # 音声読み込み
                y_data, sr = librosa.load(filepath, sr=RATE)
                
                # 無音区間除去
                y_data, _ = librosa.effects.trim(y_data, top_db=20)
                
                # MFCC抽出
                mfcc = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=N_MFCC)
                mfcc_mean = np.mean(mfcc, axis=1)
                # MFCC4とMFCC5（インデックス3と4）のみを使用
                X_mfcc.append(mfcc_mean[3:5])  # 4番目と5番目の係数のみ
                
                # フォルマント抽出
                formants = extract_formants(y_data, sr)
                X_formants.append(formants)
                
                # 個別サンプルのフォルマントを保存
                all_formants[vowel].append(formants)
                
                y.append(vowel)
                samples_found += 1
                
                print(f"✅ {vowel}: サンプル {i} を読み込み")
            else:
                print(f"⚠️ 日本語母音 {vowel} のファイルが見つかりません: {filepath}")
        
        if samples_found > 0:
            print(f"📊 {vowel}: {samples_found}個のサンプルを登録")
    
    # 英語母音の処理（実際の音声ファイルから読み込み）
    print("\n🇬🇧 英語母音サンプルを検索中...")
    for vowel in ENGLISH_VOWELS:
        all_formants[vowel] = []
        
        # 様々なファイル名パターンを試す
        patterns = [
            f"*_women_{vowel}.wav",  # FUsAoaI8QFg_women_æ.wav
            f"*_{vowel}.wav",  # 他のパターン
        ]
        
        samples_found = 0
        for pattern in patterns:
            import glob
            files = glob.glob(os.path.join(RECORDINGS_DIR, pattern))
            for filepath in files[:SAMPLES_PER_VOWEL]:  # 最大SAMPLES_PER_VOWEL個まで
                if os.path.exists(filepath):
                    try:
                        # 音声読み込み
                        y_data, sr = librosa.load(filepath, sr=RATE)
                        
                        # 無音区間除去
                        y_data, _ = librosa.effects.trim(y_data, top_db=20)
                        
                        # MFCC抽出
                        mfcc = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=N_MFCC)
                        mfcc_mean = np.mean(mfcc, axis=1)
                        X_mfcc.append(mfcc_mean[3:5])
                        
                        # フォルマント抽出
                        formants = extract_formants(y_data, sr)
                        X_formants.append(formants)
                        
                        # 個別サンプルのフォルマントを保存
                        all_formants[vowel].append(formants)
                        
                        y.append(vowel)
                        samples_found += 1
                        
                        print(f"✅ {vowel}: {os.path.basename(filepath)} を読み込み")
                        break  # 次のファイルへ
                    except Exception as e:
                        print(f"⚠️ {vowel}: {os.path.basename(filepath)} の読み込み失敗: {e}")
                
                if samples_found >= 1:  # 1つでも見つかったら次の母音へ
                    break
        
        if samples_found > 0:
            print(f"📊 {vowel}: {samples_found}個のサンプルを登録")
        else:
            print(f"❌ {vowel}: サンプルファイルが見つかりません")
    
    return np.array(X_mfcc), np.array(X_formants), np.array(y), all_formants

# === 英語母音の理論的なフォルマント値（Hz）と表記情報 ===
ENGLISH_FORMANT_DATA = {
    'æ': [660, 1720],  # cat
    'ɪ': [400, 2000],  # sit
    'ʊ': [440, 1020],  # put
    'ɛ': [530, 1840],  # get
    'ɔ': [570, 840],   # caught
    'ʌ': [640, 1190],  # but
    'ɑ': [730, 1090]   # father
}

# === 英語母音の表示ラベル（フォント対応） ===
ENGLISH_LABELS = {
    'æ': 'ae',  # cat
    'ɪ': 'I',   # sit
    'ʊ': 'U',   # put  
    'ɛ': 'E',   # get
    'ɔ': 'O',   # caught
    'ʌ': 'V',   # but
    'ɑ': 'A'    # father
}

# === 英語母音の説明 ===
ENGLISH_DESCRIPTIONS = {
    'æ': 'cat',
    'ɪ': 'sit',
    'ʊ': 'put',
    'ɛ': 'get', 
    'ɔ': 'caught',
    'ʌ': 'but',
    'ɑ': 'father'
}

# === フォルマント値からMFCC4,5の近似値を計算する関数 ===
def formant_to_mfcc45(f1, f2, jp_templates=None):
    """フォルマント値からMFCC4,5の近似値を計算（日本語母音の実測値ベース）"""
    # 日本語母音の実測値から動的に変換式を生成
    if jp_templates is not None:
        # 日本語母音のF1,F2の範囲を取得
        jp_f1_min, jp_f1_max = 300, 800  # デフォルト値
        jp_f2_min, jp_f2_max = 800, 2500
        jp_mfcc4_min, jp_mfcc4_max = -30, 70
        jp_mfcc5_min, jp_mfcc5_max = -25, 25
        
        # 実際の日本語母音データから範囲を推定
        if 'a' in jp_templates and 'i' in jp_templates:
            # aは通常F1が高く、iはF2が高い
            jp_mfcc4_range = jp_templates['a'][0] - jp_templates['i'][0]
            jp_mfcc5_range = jp_templates['i'][1] - jp_templates['a'][1]
            
            # より正確な変換係数を計算
            f1_to_mfcc4_scale = jp_mfcc4_range / (jp_f1_max - jp_f1_min) if jp_mfcc4_range != 0 else -0.1
            f2_to_mfcc5_scale = jp_mfcc5_range / (jp_f2_max - jp_f2_min) if jp_mfcc5_range != 0 else -0.02
            
            # 中心位置を計算
            jp_center_mfcc4 = np.mean([jp_templates[v][0] for v in JAPANESE_VOWELS if v in jp_templates])
            jp_center_mfcc5 = np.mean([jp_templates[v][1] for v in JAPANESE_VOWELS if v in jp_templates])
            
            # 変換式を適用
            mfcc4 = jp_center_mfcc4 - (f1 - 550) * f1_to_mfcc4_scale
            mfcc5 = jp_center_mfcc5 - (f2 - 1650) * f2_to_mfcc5_scale
        else:
            # フォールバック：従来の変換式
            mfcc4 = 80 - (f1 / 10)
            mfcc5 = 35 - (f2 / 80)
    else:
        # フォールバック：従来の変換式
        mfcc4 = 80 - (f1 / 10)
        mfcc5 = 35 - (f2 / 80)
    
    return np.array([mfcc4, mfcc5])

# === 母音ごとにMFCCとフォルマントの平均テンプレートを作成 ===
def build_templates(X_mfcc, X_formants, y):
    templates = {}
    formant_templates = {}
    
    print("\n📝 テンプレート作成中...")
    
    # 日本語母音のテンプレート作成（音声サンプルから）
    samples_used = {}
    for vowel in JAPANESE_VOWELS:
        indices = y == vowel
        if np.any(indices):
            templates[vowel] = np.mean(X_mfcc[indices], axis=0)
            formant_templates[vowel] = np.mean(X_formants[indices], axis=0)
            samples_used[vowel] = np.sum(indices)
            print(f"✅ 日本語 {vowel}: {samples_used[vowel]}個のサンプルからテンプレート作成")
    
    # 英語母音のテンプレート作成（実際の音声サンプルまたは理論値から）
    for eng_vowel in ENGLISH_VOWELS:
        indices = y == eng_vowel
        if np.any(indices):  # 実際のサンプルがある場合
            templates[eng_vowel] = np.mean(X_mfcc[indices], axis=0)
            formant_templates[eng_vowel] = np.mean(X_formants[indices], axis=0)
            samples_used[eng_vowel] = np.sum(indices)
            print(f"✅ 英語 {eng_vowel}: {samples_used[eng_vowel]}個のサンプルからテンプレート作成")
        elif eng_vowel in ENGLISH_FORMANT_DATA:  # サンプルがない場合は理論値を使用
            print(f"🔄 {eng_vowel}: サンプルがないため理論値から推定...")
            formants = ENGLISH_FORMANT_DATA[eng_vowel]
            formant_templates[eng_vowel] = np.array(formants + [0])  # F3は0で初期化
            
            # 日本語母音の実測値を使ってフォルマント値からMFCC4,5を推定
            mfcc_pos = formant_to_mfcc45(formants[0], formants[1], templates)
            templates[eng_vowel] = mfcc_pos
            
            print(f"🔤 {eng_vowel}: F1={formants[0]}Hz, F2={formants[1]}Hz → MFCC4,5=({mfcc_pos[0]:.1f}, {mfcc_pos[1]:.1f})")
    
    print(f"\n✅ 総テンプレート数: {len(templates)}")
    jp_count = len([v for v in templates if v in JAPANESE_VOWELS])
    en_count = len([v for v in templates if v in ENGLISH_VOWELS])
    en_sample_count = len([v for v in samples_used if v in ENGLISH_VOWELS])
    en_theory_count = en_count - en_sample_count
    
    print(f"  ● 日本語: {jp_count}/{len(JAPANESE_VOWELS)} (全てサンプルから)")
    print(f"  ▲ 英語: {en_count}/{len(ENGLISH_VOWELS)} (実サンプル: {en_sample_count}, 理論値: {en_theory_count})")
    
    # 曖昧母音（シュワー）の理論値を計算
    if jp_count >= len(JAPANESE_VOWELS):
        # 日本語5母音のフォルマント値から中心値を計算
        jp_formants = [formant_templates[v] for v in JAPANESE_VOWELS if v in formant_templates]
        if jp_formants:
            jp_formants_array = np.array(jp_formants)
            schwa_formants = np.mean(jp_formants_array, axis=0)
            
            # 曖昧母音のMFCCは日本語母音の平均
            jp_mfcc = [templates[v] for v in JAPANESE_VOWELS if v in templates]
            if jp_mfcc:
                jp_mfcc_array = np.array(jp_mfcc)
                schwa_mfcc = np.mean(jp_mfcc_array, axis=0)
                
                # テンプレートに追加
                templates['ə'] = schwa_mfcc
                formant_templates['ə'] = schwa_formants
                
                print(f"🔲 シュワー音: F1={schwa_formants[0]:.0f}Hz, F2={schwa_formants[1]:.0f}Hz (日本語5母音の平均)")
    
    return templates, formant_templates


# === 録音された音声から特徴を抽出 ===
def extract_user_features(filepath):
    # 音声読み込み
    y_data, sr = librosa.load(filepath, sr=RATE)
    
    # 無音区間除去
    y_data, _ = librosa.effects.trim(y_data, top_db=20)
    
    # MFCC抽出
    mfcc = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    # MFCC4とMFCC5（インデックス3と4）のみを使用
    mfcc_features = mfcc_mean[3:5]  # 4番目と5番目の係数のみ
    
    # フォルマント抽出
    formant_features = extract_formants(y_data, sr)
    
    return mfcc_features, formant_features

# === ユーザーのMFCCとフォルマントをテンプレートと比較して分類 ===
def classify(user_mfcc, user_formants, templates, formant_templates, all_formants=None):
    # MFCC距離の計算
    mfcc_distances = {vowel: np.linalg.norm(user_mfcc - vec) for vowel, vec in templates.items()}
    
    # サンプルの個別フォルマントとの比較（類似度計算）
    sample_similarities = {}
    very_similar_vowel = None
    max_formant = 4000  # 正規化のための最大フォルマント周波数
    
    if all_formants is not None:
        # 日本語と英語両方の母音をチェック
        for vowel in VOWELS:
            vowel_samples = all_formants.get(vowel, [])
            if vowel_samples:
                # 各サンプルとの類似度を計算
                similarities = []
                for sample_formant in vowel_samples:
                    if len(sample_formant) >= 2 and len(user_formants) >= 2:
                        # F1とF2の距離を計算
                        f1_dist = abs(user_formants[0] - sample_formant[0]) / max_formant
                        f2_dist = abs(user_formants[1] - sample_formant[1]) / max_formant
                        
                        # 距離から類似度を計算
                        dist = 0.5 * f1_dist + 0.5 * f2_dist
                        
                        sim = 1.0 / (1.0 + 10 * dist)
                        similarities.append(sim)
                
                # 最も類似度の高いサンプルを選択
                if similarities:
                    max_sim = max(similarities)
                    
                    # 類似度はそのまま使用
                    
                    sample_similarities[vowel] = max_sim
                    
                    # 非常に高い類似度の場合、その母音を記録
                    if max_sim > VERY_SIMILAR_THRESHOLD and (very_similar_vowel is None or max_sim > sample_similarities.get(very_similar_vowel, 0)):
                        very_similar_vowel = vowel
    
    # MFCC距離のみを使用（フォルマントは類似度判定のみ）
    combined_distances = mfcc_distances.copy()
    
    # 距離が近い順にソート
    sorted_distances = sorted(combined_distances.items(), key=lambda x: x[1])
    
    # サンプルとの類似度が非常に高い場合、その母音を優先
    if very_similar_vowel is not None:
        print(f"\n⭐ サンプルとの高い類似度検出: 「{very_similar_vowel}」 (類似度: {sample_similarities[very_similar_vowel]:.3f})")
        # 該当の母音を先頭に持ってくる
        sorted_distances = [(very_similar_vowel, combined_distances[very_similar_vowel])] + \
                          [d for d in sorted_distances if d[0] != very_similar_vowel]
    
    # 距離が近い順にソート済み
    
    # 判別結果とともに各種情報を返す
    return sorted_distances, mfcc_distances, sample_similarities if 'sample_similarities' in locals() else {}


# === 2次元プロットの初期化（MFCC4,5のみ使用） ===
def init_2d_plot(X, y, templates):
    # 直接MFCC4,5をプロット
    X_plot = X  # すでにMFCC4,5の2次元
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # 背景色を設定
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    # 日本語母音クラスタをプロット（実際のサンプル位置）
    for vowel in JAPANESE_VOWELS:
        cluster = X_plot[y == vowel]
        if len(cluster) > 0:
            color = COLOR_MAP.get(vowel, 'gray')
            # 実際のサンプルポイントを小さく表示（参考用）
            ax.scatter(cluster[:, 1], cluster[:, 0],  # X軸とY軸を入れ替え
                      s=60, color=color, alpha=0.5, 
                      edgecolor='gray', linewidth=1, zorder=3)
    
    # 日本語母音のテンプレート位置をプロット（理想的な位置）
    for vowel in JAPANESE_VOWELS:
        if vowel in templates:
            template_point = templates[vowel]
            color = COLOR_MAP.get(vowel, 'gray')
            
            # テンプレート位置に大きな丸を表示
            ax.scatter(template_point[1], template_point[0],  # X軸とY軸を入れ替え
                      label=f'{vowel} (日本語)', s=300, color=color, alpha=0.9, 
                      edgecolor='white', linewidth=3, zorder=8)
            
            # テンプレート位置の中心に母音ラベルを表示
            ax.text(template_point[1], template_point[0], vowel,  # X軸とY軸を入れ替え
                    fontsize=18, weight='bold', color='white', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.95, 
                             edgecolor='white', linewidth=2), zorder=9)
            
            # テンプレート位置を中心とした円を表示
            circle = plt.Circle((template_point[1], template_point[0]), radius=5, 
                              fill=False, edgecolor=color, linewidth=2, 
                              alpha=0.6, linestyle=':', zorder=7)
            ax.add_patch(circle)
            
            # 母音名を外側に表示（より目立つように）
            ax.text(template_point[1] + 3, template_point[0] + 3, f'/{vowel}/', 
                    fontsize=12, weight='bold', color=color, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, 
                             edgecolor=color, linewidth=1), zorder=10)
            
            print(f"  ✅ {vowel} テンプレート位置: ({template_point[0]:.1f}, {template_point[1]:.1f})")
    
    # 英語母音の理論位置を強制的にプロット
    eng_count = len([v for v in templates.keys() if v in ENGLISH_VOWELS])
    print(f"\n🔧 英語母音テンプレートをプロット中... (テンプレート数: {eng_count}/{len(ENGLISH_VOWELS)})")
    
    if eng_count == 0:
        print("    ⚠️ 英語母音のテンプレートが作成されていません！")
        print(f"    テンプレートキー: {list(templates.keys())}")
    
    for eng_vowel in ENGLISH_VOWELS:
        if eng_vowel in templates:
            point = templates[eng_vowel]
            color = COLOR_MAP.get(eng_vowel, 'gray')
            print(f"  📍 {eng_vowel}: 位置=({point[0]:.1f}, {point[1]:.1f}), 色={color}")
            
            # 図の範囲内かチョック
            if -30 <= point[0] <= 80 and -30 <= point[1] <= 30:
                # より大きく、見やすい三角形マーカーで英語母音テンプレートをプロット
                ax.scatter(point[1], point[0],  # X軸とY軸を入れ替え
                          label=f'{ENGLISH_LABELS.get(eng_vowel, eng_vowel)} ({ENGLISH_DESCRIPTIONS.get(eng_vowel, "EN")})', 
                          s=350, color=color, alpha=0.9, 
                          marker='^', edgecolor='white', linewidth=3, zorder=10)
                
                # より大きく見やすいテキストラベル
                display_label = ENGLISH_LABELS.get(eng_vowel, eng_vowel)
                
                ax.text(point[1], point[0], display_label,  # X軸とY軸を入れ替え
                        fontsize=16, weight='bold', color='white', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.95, 
                                 edgecolor='white', linewidth=2),
                        zorder=11)
                
                # 英語母音テンプレート位置を中心とした円を表示
                circle = plt.Circle((point[1], point[0]), radius=4, 
                                  fill=False, edgecolor=color, linewidth=2, 
                                  alpha=0.6, linestyle='--', zorder=9)
                ax.add_patch(circle)
                
                # 英語母音の説明を外側に表示
                desc = ENGLISH_DESCRIPTIONS.get(eng_vowel, eng_vowel)
                ax.text(point[1] + 3, point[0] - 3, f'/{desc}/', 
                        fontsize=10, weight='bold', color=color, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow', alpha=0.8, 
                                 edgecolor=color, linewidth=1), zorder=12)
                
                print(f"    ✅ {eng_vowel} をプロット完了")
            else:
                print(f"    ⚠️ {eng_vowel} が範囲外: ({point[0]:.1f}, {point[1]:.1f})")
        else:
            print(f"    ❌ {eng_vowel} がテンプレートに見つかりません")
    
    # 曖昧母音の理論位置をプロット
    if 'ə' in templates:
        schwa_point = templates['ə']  # すでに2次元
        color = COLOR_MAP.get('ə', 'gray')
        ax.scatter(schwa_point[1], schwa_point[0],  # X軸とY軸を入れ替え
                  label='ə (シュワー)', s=300, color=color, alpha=0.8, 
                  marker='s', edgecolor='white', linewidth=3, zorder=7)
        ax.text(schwa_point[1], schwa_point[0], 'ə',  # X軸とY軸を入れ替え
                fontsize=16, weight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9,
                         edgecolor='white', linewidth=2), zorder=8)
        
        # シュワー音にも円と説明を追加
        circle = plt.Circle((schwa_point[1], schwa_point[0]), radius=4.5, 
                          fill=False, edgecolor=color, linewidth=2, 
                          alpha=0.4, linestyle='-.', zorder=6)
        ax.add_patch(circle)
        
        ax.text(schwa_point[1] + 3, schwa_point[0], '/schwa/', 
                fontsize=10, weight='bold', color=color, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='lightcyan', alpha=0.8, 
                         edgecolor=color, linewidth=1), zorder=9)
    
    # グラフの設定（軸ラベルも入れ替え）
    ax.set_title("🎯 MFCC4-5 母音認識空間\n" + 
                "🔵 日本語母音　🔺 英語母音　🔲 シュワー音　⭐ あなたの発音", 
                fontsize=18, pad=25, weight='bold')
    ax.set_xlabel("MFCC5（第6係数）", fontsize=16, weight='bold')
    ax.set_ylabel("MFCC4（第5係数）", fontsize=16, weight='bold')
    
    # より見やすい凡例設定
    handles, labels = ax.get_legend_handles_labels()
    # 日本語と英語で分けて表示
    jp_items = [(h, l) for h, l in zip(handles, labels) if '日本語' in l or 'シュワー' in l]
    en_items = [(h, l) for h, l in zip(handles, labels) if '(' in l and '日本語' not in l and 'シュワー' not in l]
    
    if jp_items:
        jp_legend = ax.legend([h for h, l in jp_items], [l for h, l in jp_items],
                             bbox_to_anchor=(1.02, 1), loc='upper left', 
                             title='🔵 日本語母音', title_fontsize=14, fontsize=12,
                             frameon=True, fancybox=True, shadow=True)
        jp_legend.get_title().set_weight('bold')
        plt.gca().add_artist(jp_legend)
    
    if en_items:
        en_legend = ax.legend([h for h, l in en_items], [l for h, l in en_items],
                             bbox_to_anchor=(1.02, 0.55), loc='upper left',
                             title='🔺 英語母音', title_fontsize=14, fontsize=12,
                             frameon=True, fancybox=True, shadow=True)
        en_legend.get_title().set_weight('bold')
    
    # より見やすいグリッド
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='lightgray')
    ax.grid(True, linestyle=':', linewidth=1, alpha=0.5, color='blue', which='major')
    
    # 軸の範囲を手動で設定して全ての点が見えるようにする
    all_points = list(templates.values())
    if all_points:
        all_points = np.array(all_points)
        x_margin = max(5, (all_points[:, 1].max() - all_points[:, 1].min()) * 0.15)
        y_margin = max(5, (all_points[:, 0].max() - all_points[:, 0].min()) * 0.15)
        ax.set_xlim(all_points[:, 1].min() - x_margin, all_points[:, 1].max() + x_margin)
        ax.set_ylim(all_points[:, 0].min() - y_margin, all_points[:, 0].max() + y_margin)

    print(f"\n📊 プロット完了 - 総ポイント数: {len(templates)}")
    
    # 軸の色と太さを調整
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    
    # 軸のラベルの色を調整
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#333333')
    
    # 図を適切なサイズに調整
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)  # 凡例用のスペースを確保
    
    return fig, ax, X_plot

# === ユーザーの母音点をプロットに追加・更新する ===
def update_user_point(ax, user_vec, predicted_label, score, prev_scatter=None, templates=None):
    # 以前の点を削除（常に1つだけ表示）
    if prev_scatter:
        prev_scatter.remove()
    user_point = user_vec
    scatter = ax.scatter(user_point[1], user_point[0], color='red', s=400, marker='*', 
                        edgecolor='yellow', linewidth=3, zorder=20, alpha=0.9)
    
    # 最も近いテンプレートとの距離を計算して表示
    distance_info = ""
    if templates and predicted_label in templates:
        target_point = templates[predicted_label]
        distance = np.linalg.norm(user_point - target_point)
        if distance < 5:
            distance_info = f" | 🎉 テンプレートまで {distance:.1f} - 素晴らしい！"
        elif distance < 15:
            distance_info = f" | 😊 テンプレートまで {distance:.1f} - 良い感じ！"
        elif distance < 30:
            distance_info = f" | 💪 テンプレートまで {distance:.1f} - あと少し！"
        else:
            distance_info = f" | 🎯 テンプレートまで {distance:.1f} - 頭張ろう！"
    
    advice = ADVICE_MAP.get(predicted_label, "練習を続けましょう。")
    ax.set_title(f"推定: 「{predicted_label}」 (距離: {score:.2f}){distance_info}\n💡 {advice}", 
                fontsize=14, pad=10)
    plt.pause(0.01)
    return scatter

# === 発音スコアに応じたアドバイス表示 ===
def show_advice(vowel, score):
    print("\n🧪 発音評価:")
    if score < 15:
        print("✅ 発音は良好です！")
    elif score < 30:
        print("⚠ 少しズレています。もう一度意識して発音してみましょう。")
    else:
        print("❌ 発音がかなりズレています。以下の改善点を参考にしてください。")
        print(f"🗣 「{vowel}」の発音アドバイス: {ADVICE_MAP.get(vowel, '練習を続けましょう。')}")

# === MFCC特徴量の説明 ===
def show_mfcc_info():
    print("\n📊 使用中の特徴量:")
    print("  MFCC4（第4係数）: 中域スペクトル構造")
    print("  MFCC5（第5係数）: 中域スペクトル構造")
    print("\n📖 MFCC4,5の特性:")
    print("  ・フォルマント周波数（特にF1,F2）と高い相関")
    print("  ・母音の音響的特徴を効率的に表現")
    print("  ・スペクトルの中域（約1000-3000Hz）の情報を主に反映")


# === フォルマント情報の表示 ===
def display_formant_info(formant_templates):
    """各母音のフォルマント情報を表示"""
    print("\n🔍 母音のフォルマント情報:")
    for vowel, formants in formant_templates.items():
        if len(formants) >= 2:
            print(f"  「{vowel}」: F1={formants[0]:.0f}Hz, F2={formants[1]:.0f}Hz")
            
            # 特にiとeの違いを詳細表示
            if vowel in ['i', 'e']:
                print(f"    👉 「{vowel}」の特徴: {'高い第2フォルマント' if vowel == 'i' else '中程度の第2フォルマント'}")





# === メイン処理 ===
def main():
    print("📦 テンプレート読み込み中...")
    X_mfcc, X_formants, y, all_formants = extract_features()
    
    if len(X_mfcc) == 0:
        print("❌ テンプレートがありません。recordings フォルダを確認してください。")
        return

    # テンプレート作成
    templates, formant_templates = build_templates(X_mfcc, X_formants, y)
    
    # フォルマント情報の表示
    display_formant_info(formant_templates)
    
    
    
    # MFCC情報表示
    show_mfcc_info()

    plt.ion()  # 対話モードON
    _, ax, _ = init_2d_plot(X_mfcc, y, templates)

    print(f"\n💫 個別サンプルとの類似度比較: 有効 (閾値: {VERY_SIMILAR_THRESHOLD})")
    print("\n📊 テンプレート位置情報:")
    print("  🔵 日本語母音テンプレート (大きな丸・理想的な位置):")
    for vowel, mfcc_pos in templates.items():
        if vowel in JAPANESE_VOWELS:
            print(f"    ● {vowel}: ({mfcc_pos[0]:.1f}, {mfcc_pos[1]:.1f}) - この位置を目標に発音練習")
    
    print("  🔺 英語母音テンプレート (三角形・理想的な位置):")
    for vowel, mfcc_pos in templates.items():
        if vowel in ENGLISH_VOWELS:
            desc = ENGLISH_DESCRIPTIONS.get(vowel, vowel)
            print(f"    ▲ {vowel} ({desc}): ({mfcc_pos[0]:.1f}, {mfcc_pos[1]:.1f}) - この位置を目標に発音練習")
    
    for vowel, mfcc_pos in templates.items():
        if vowel == 'ə':
            print(f"  🔲 {vowel} (シュワー): ({mfcc_pos[0]:.1f}, {mfcc_pos[1]:.1f}) - 中立的な音の位置")
    
    print("\n📌 MFCC4,5ベースの母音発音練習システム")
    print("  → MFCC4,5は中域のスペクトル構造を表し、フォルマント情報と相関")
    print("  → 大きなマーカー = テンプレート位置（理想的な目標）")
    print("  → 小さなマーカー = 実際の音声サンプル（参考用）")
    print("  → 破線の円 = 発音練習の目標範囲")
    print("🎯 目標: 大きなマーカーの位置に近づけるように発音してください！")
    print("🟢 リアルタイム母音認識を開始します（Ctrl+Cで停止）")
    prev_scatter = None
    last_predicted = None
    last_dist = None
    
    RATE = 16000
    FRAME_SIZE = int(RATE * 0.05)  # 0.05秒フレーム
    MIN_VOICE_FRAMES = int(0.15 / 0.05)  # 0.15秒以上の音声のみ判定
    MAX_SILENCE_FRAMES = int(0.2 / 0.05)  # 0.2秒以上無音で音声区間終了
    q = queue.Queue()

    def audio_callback(indata, *args):
        q.put(indata.copy())

    is_voice = False
    silence_count = 0
    voice_frames = []

    with sd.InputStream(samplerate=RATE, channels=1, callback=audio_callback, blocksize=FRAME_SIZE):
        try:
            while True:
                # バッファにデータを追加
                while not q.empty():
                    frame = q.get().flatten()
                    # 音声区間判定
                    max_amplitude = np.max(np.abs(frame))
                    rms_energy = np.sqrt(np.mean(frame**2))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(frame, frame_length=len(frame), hop_length=len(frame)//2))
                    # さらにしきい値を和らげる
                    is_valid = (max_amplitude > 0.03 and rms_energy > 0.005 and zcr < 0.30)
                    if is_valid:
                        voice_frames.append(frame)
                        is_voice = True
                        silence_count = 0
                    else:
                        if is_voice:
                            silence_count += 1
                            if silence_count >= MAX_SILENCE_FRAMES:
                                # 音声区間終了
                                if len(voice_frames) >= MIN_VOICE_FRAMES:
                                    voice_audio = np.concatenate(voice_frames)
                                    tmp_path = "_tmp_user_input.wav"
                                    sf.write(tmp_path, voice_audio, RATE)
                                    user_mfcc, user_formants = extract_user_features(tmp_path)
                                    os.remove(tmp_path)
                                    if user_mfcc is not None and user_formants is not None:
                                        results, _, sample_similarities = classify(
                                            user_mfcc, user_formants, templates, formant_templates, all_formants)
                                        predicted, dist = results[0]
                                        print("\n=== 判定結果 ===")
                                        print(f"🗣 推定: 「{predicted}」 / 距離スコア: {dist:.2f}")
                                        f1, f2 = user_formants[0], user_formants[1]
                                        print(f"📊 あなたのフォルマント: F1={f1:.0f}Hz, F2={f2:.0f}Hz, F3={user_formants[2]:.0f}Hz")
                                        if sample_similarities:
                                            print("📊 サンプルとの類似度:")
                                            for vowel, sim in sorted(sample_similarities.items(), key=lambda x: x[1], reverse=True)[:3]:
                                                print(f"  「{vowel}」: {sim:.3f}")
                                        print("📊 類似度ランキング:")
                                        for i, (v, d) in enumerate(results):
                                            print(f"  {i+1}. {v}（距離: {d:.2f}）")
                                        show_advice(predicted, dist)
                                        prev_scatter = update_user_point(ax, user_mfcc, predicted, dist, prev_scatter, templates)
                                        last_predicted = predicted
                                        last_dist = dist
                                        plt.pause(0.01)
                                # バッファリセット
                                voice_frames = []
                                is_voice = False
                                silence_count = 0
                        else:
                            # 無音時もグラフ・点・タイトルはそのまま
                            if last_predicted is not None and last_dist is not None:
                                advice = ADVICE_MAP.get(last_predicted, "練習を続けましょう。")
                                ax.set_title(f"推定された母音: 「{last_predicted}」 (距離: {last_dist:.2f})\n💡 {advice}\n🔇 無音が検出されました", fontsize=14, color='red')
                            else:
                                ax.set_title("🔇 無音が検出されました", fontsize=16, color='red')
                            for text in ax.texts:
                                text.remove()
                            ax.text(0, 0, "🔇\n無音", fontsize=20, color='red', ha='center', va='center', alpha=0.8, weight='bold')
                            plt.pause(0.01)
                sleep(0.01)
        except KeyboardInterrupt:
            print("\n🛑 終了しました。")
            plt.ioff()
            plt.close('all')

# === 録音機能 ===
def record_vowels():
    """日本語母音のサンプルを録音する"""
    vowels = ['a', 'i', 'u', 'e', 'o']
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    
    print("🎙️ 日本語母音サンプルの録音を開始します")
    print("各母音を3回ずつ録音してください")
    print("録音時間: 1秒間")
    
    for vowel in vowels:
        print(f"\n=== 「{vowel}」の録音 ===")
        for i in range(SAMPLES_PER_VOWEL):
            print(f"→「{vowel}」を発音してください（{i+1}/{SAMPLES_PER_VOWEL}）")
            print("3秒後に録音開始...")
            
            # カウントダウン
            for j in range(3, 0, -1):
                print(f"{j}...")
                sleep(1)
            
            print("🔴 録音中...")
            audio = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=1)
            sd.wait()
            
            # ファイル保存
            filename = f"{RECORDINGS_DIR}/{vowel}_{i+1}.wav"
            sf.write(filename, audio, RATE)
            print(f"✅ 録音完了: {filename}")
    
    print("\n🎉 すべての録音が完了しました！")
    print("システムを再起動してテンプレートを作成してください。")

def check_missing_samples():
    """不足している日本語母音サンプルをチェック"""
    missing_vowels = []
    
    for vowel in JAPANESE_VOWELS:
        found_samples = 0
        for i in range(1, SAMPLES_PER_VOWEL + 1):
            filepath = os.path.join(RECORDINGS_DIR, f"{vowel}_{i}.wav")
            if os.path.exists(filepath):
                found_samples += 1
        
        if found_samples == 0:
            missing_vowels.append(vowel)
        elif found_samples < SAMPLES_PER_VOWEL:
            print(f"⚠️ {vowel}: {found_samples}/{SAMPLES_PER_VOWEL}個のサンプルのみ存在")
    
    return missing_vowels

# === エントリーポイント ===
if __name__ == "__main__":
    # recordingsディレクトリの存在確認
    if not os.path.exists(RECORDINGS_DIR):
        print(f"📁 {RECORDINGS_DIR}ディレクトリを作成します...")
        os.makedirs(RECORDINGS_DIR)
    
    # 不足サンプルのチェック
    missing_vowels = check_missing_samples()
    
    if missing_vowels:
        print(f"❌ 以下の日本語母音のサンプルがありません: {missing_vowels}")
        
        # 録音するかユーザーに確認
        response = input("\n録音しますか？ (y/n): ").lower().strip()
        if response in ['y', 'yes', 'はい']:
            record_vowels()
        else:
            print("⚠️ サンプルファイルを用意してから再実行してください")
    else:
        print("✅ 日本語母音サンプルが揃っています")
        main()