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
from matplotlib.widgets import Button

# === フォント設定（Mac対応・IPA記号対応） ===
# IPA記号が表示できるフォントを設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Lucida Grande', 'AppleGothic']

# === 音声処理設定 ===
RATE = 16000  # サンプリングレート（16kHz）
DURATION = 1.0  # 録音時間（秒）
N_MFCC = 13  # MFCCの次元数
RECORDINGS_DIR = "recordings_formant"  # 録音ファイルの保存先
JAPANESE_VOWELS = ['a', 'i', 'u', 'e', 'o']  # 日本語母音
ENGLISH_VOWELS = ['æ', 'ɪ', 'ʊ', 'ɛ', 'ɔ', 'ʌ', 'ɑ', 'ə']  # 英語母音（cat, sit, put, get, caught, but, father, schwa）
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


# === 英語母音の表示ラベル（フォント対応） ===
ENGLISH_LABELS = {
    'æ': 'ae',  # cat
    'ɪ': 'I',   # sit
    'ʊ': 'U',   # put  
    'ɛ': 'E',   # get
    'ɔ': 'O',   # caught
    'ʌ': 'V',   # but
    'ɑ': 'A',   # father
    'ə': 'S'    # schwa
}

# === 英語母音の説明 ===
ENGLISH_DESCRIPTIONS = {
    'æ': 'cat',
    'ɪ': 'sit',
    'ʊ': 'put',
    'ɛ': 'get', 
    'ɔ': 'caught',
    'ʌ': 'but',
    'ɑ': 'father',
    'ə': 'schwa'
}


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
    
    # 英語母音のテンプレート作成（実際の音声サンプルから）
    for eng_vowel in ENGLISH_VOWELS:
        indices = y == eng_vowel
        if np.any(indices):  # 実際のサンプルがある場合
            templates[eng_vowel] = np.mean(X_mfcc[indices], axis=0)
            formant_templates[eng_vowel] = np.mean(X_formants[indices], axis=0)
            samples_used[eng_vowel] = np.sum(indices)
            print(f"✅ 英語 {eng_vowel}: {samples_used[eng_vowel]}個のサンプルからテンプレート作成")
        else:
            print(f"❌ {eng_vowel}: サンプルがありません。スキップします")
    
    print(f"\n✅ 総テンプレート数: {len(templates)}")
    jp_count = len([v for v in templates if v in JAPANESE_VOWELS])
    en_count = len([v for v in templates if v in ENGLISH_VOWELS])
    en_sample_count = len([v for v in samples_used if v in ENGLISH_VOWELS])
    
    print(f"  ● 日本語: {jp_count}/{len(JAPANESE_VOWELS)} (全てサンプルから)")
    print(f"  ▲ 英語: {en_count}/{len(ENGLISH_VOWELS)} (実サンプルのみ)")
    
    
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

# === ユーザーのMFCCをテンプレートと比較して分類 ===
def classify(user_mfcc, templates):
    # MFCC距離の計算
    mfcc_distances = {vowel: np.linalg.norm(user_mfcc - vec) for vowel, vec in templates.items()}
    
    # 距離が近い順にソート
    sorted_distances = sorted(mfcc_distances.items(), key=lambda x: x[1])
    
    # 判別結果を返す
    return sorted_distances


# === 2次元プロットの初期化（MFCC4,5のみ使用） ===
def init_2d_plot(X, y, templates):
    # 直接MFCC4,5をプロット
    X_plot = X  # すでにMFCC4,5の2次元
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # 背景色を設定
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    # 日本語母音の個別サンプルポイントは非表示（テンプレートのみ表示）
    
    # 日本語母音のテンプレート位置をプロット（理想的な位置）
    for vowel in JAPANESE_VOWELS:
        if vowel in templates:
            template_point = templates[vowel]
            color = COLOR_MAP.get(vowel, 'gray')
            
            # テンプレート位置に大きな丸を表示
            ax.scatter(template_point[1], template_point[0],  # X軸とY軸を入れ替え
                      s=300, color=color, alpha=0.9, 
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
                
                
                print(f"    ✅ {eng_vowel} をプロット完了")
            else:
                print(f"    ⚠️ {eng_vowel} が範囲外: ({point[0]:.1f}, {point[1]:.1f})")
        else:
            print(f"    ❌ {eng_vowel} がテンプレートに見つかりません")
    
    
    # グラフの設定（軸ラベルも入れ替え）
    ax.set_title("🎯 MFCC4-5 母音認識空間\n" + 
                "🔵 日本語母音　🔺 英語母音　🔲 シュワー音　⭐ あなたの発音", 
                fontsize=18, pad=25, weight='bold')
    ax.set_xlabel("MFCC5（第6係数）", fontsize=16, weight='bold')
    ax.set_ylabel("MFCC4（第5係数）", fontsize=16, weight='bold')
    
    # 英語母音のラベル対応表をグラフ外側右に表示
    fig.text(0.85, 0.8, 
            "🔺 英語母音ラベル対応表\n\n"
            "ae → æ (cat)\n"
            "I  → ɪ (sit)\n" 
            "U  → ʊ (put)\n"
            "E  → ɛ (get)\n"
            "O  → ɔ (caught)\n"
            "V  → ʌ (but)\n"
            "A  → ɑ (father)\n"
            "S  → ə (schwa)",
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='lightyellow', 
                     alpha=0.9,
                     edgecolor='orange',
                     linewidth=1))
    
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
    
    # 図を適切なサイズに調整（右側にスペースを確保）
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # 右側に対応表用のスペースを確保
    
    return fig, ax, X_plot

# === ユーザーの母音点をプロットに追加・更新する ===
def update_user_point(ax, user_vec, predicted_label, prev_scatter=None):
    # 以前の点を削除（常に1つだけ表示）
    if prev_scatter:
        prev_scatter.remove()
    user_point = user_vec
    scatter = ax.scatter(user_point[1], user_point[0], color='red', s=400, marker='*', 
                        edgecolor='yellow', linewidth=3, zorder=20, alpha=0.9)
    
    advice = ADVICE_MAP.get(predicted_label, "練習を続けましょう。")
    ax.set_title(f"推定: 「{predicted_label}」\n💡 {advice}", 
                fontsize=14, pad=10)
    plt.pause(0.01)
    return scatter


# === MFCC特徴量の説明 ===
def show_mfcc_info():
    print("\n📊 使用中の特徴量:")
    print("  MFCC4（第4係数）: 中域スペクトル構造")
    print("  MFCC5（第5係数）: 中域スペクトル構造")
    print("\n📖 MFCC4,5の特性:")
    print("  ・フォルマント周波数（特にF1,F2）と高い相関")
    print("  ・母音の音響的特徴を効率的に表現")
    print("  ・スペクトルの中域（約1000-3000Hz）の情報を主に反映")

# === メイン処理 ===
def main():
    print("📦 テンプレート読み込み中...")
    X_mfcc, X_formants, y, _ = extract_features()
    
    if len(X_mfcc) == 0:
        print("❌ テンプレートがありません。recordings フォルダを確認してください。")
        return

    # テンプレート作成
    templates, _ = build_templates(X_mfcc, X_formants, y)
    
    
    
    
    # MFCC情報表示
    show_mfcc_info()

    plt.ion()  # 対話モードON
    fig, ax, _ = init_2d_plot(X_mfcc, y, templates)
    
    # 終了フラグを共有するためのdict
    stop_flag = {'stop': False}
    
    # 終了ボタンを追加
    ax_button = plt.axes([0.85, 0.01, 0.13, 0.05])  # 右下にボタン配置
    button = Button(ax_button, '終了', color='lightcoral', hovercolor='red')
    
    def on_button_click(event):
        stop_flag['stop'] = True
        print("\n🔴 終了ボタンが押されました")
    
    button.on_clicked(on_button_click)

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
    
    
    print("\n📌 MFCC4,5ベースの母音発音練習システム")
    print("  → MFCC4,5は中域のスペクトル構造を表し、フォルマント情報と相関")
    print("  → 大きなマーカー = テンプレート位置（理想的な目標）")
    print("  → 小さなマーカー = 実際の音声サンプル（参考用）")
    print("  → 破線の円 = 発音練習の目標範囲")
    print("🎯 目標: 大きなマーカーの位置に近づけるように発音してください！")
    print("🟢 リアルタイム母音認識を開始します（終了ボタンまたはCtrl+Cで停止）")
    prev_scatter = None
    last_predicted = None
    
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
            while not stop_flag['stop']:
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
                                    user_mfcc, _ = extract_user_features(tmp_path)
                                    os.remove(tmp_path)
                                    if user_mfcc is not None:
                                        results = classify(user_mfcc, templates)
                                        predicted, _ = results[0]
                                        print("\n=== 判定結果 ===")
                                        print(f"🗣 推定: 「{predicted}」")
                                        print("📊 類似度ランキング:")
                                        for i, (v, _) in enumerate(results[:5]):
                                            print(f"  {i+1}. {v}")
                                        prev_scatter = update_user_point(ax, user_mfcc, predicted, prev_scatter)
                                        last_predicted = predicted
                                        plt.pause(0.01)
                                # バッファリセット
                                voice_frames = []
                                is_voice = False
                                silence_count = 0
                        else:
                            # 無音時はタイトルのみに表示
                            if last_predicted is not None:
                                advice = ADVICE_MAP.get(last_predicted, "練習を続けましょう。")
                                ax.set_title(f"推定された母音: 「{last_predicted}」\n💡 {advice}\n🔇 無音", fontsize=14, color='gray')
                            else:
                                ax.set_title("🔇 無音", fontsize=16, color='gray')
                            plt.pause(0.01)
                sleep(0.01)
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+Cで終了しました。")
        
        # 終了処理
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