"""
issue: cc/ conj対応
issue: stanzaで構造解析を行う際に物語内の文章全て入力して同時に解析するのと一行ずつ分けて解析する場合とどちらが速度が速いか
"""
# 本コードの場合$CORENLP_HOMEの設定が必要ない
import sys
import re
import os
import csv

import stanza

# 一旦大規模ファイルは気にしない(メモリに乗りきるサイズのみを想定)
def file_load(filepath):
  if os.path.isfile(filepath):
    print("load file:" + filepath)
    return open(filepath, 'r')
  else:
    print(filepath + " is not a file")

def file_close(loaded_file):
  loaded_file.close()

class DependencyAnalysis:
  # filepath: 読み込みファイルのパス
  # config_gpu: 依存関係解析モデルをGPU上で動かすか否か
  def __init__(self, config_gpu=False):
    self.all_result = []
    # gpuマシンで動かす場合、Falseを変更
    self.nlp = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma, depparse', use_gpu=config_gpu)

  """
  主節動詞の抽出(非主節の場合は動詞はそもそも抽出されているのでこの関数を実行しない)
  dep: 現在の文章における解析された依存関係
  dependent_root_id: 解析したい依存関係における、より根に近い単語のID
  """
  def extract_verbs(self, dep, dependent_root_id):
    for i in range(2):
      for d in dep:
        if (d[1] == "root") or (d[0].id == dependent_root_id):
          if 'VB' in d[2].xpos:
            return d[2].id, d[2].text
          elif dependent_root_id == 0:
            dependent_root_id = d[2].id

  """
  入力された動詞idに対する動作主体の抽出。取る可能性のある依存関係はcsubj:~,nsubj:~のみ。
  issue:cc/conjへの対応追加
  引数はextract_verbsに同じ
  """
  def extract_subject(self, dep, dependent_root_id):
    for i in dep:
      if ("subj" in i[1]) and (i[0].id == dependent_root_id):
        if "csubj" in i[1]:
          return i[2].id, [i[2].text]
        elif "nsubj" in i[1]:
          return i[2].id, i[2].text
    return -1, "<none>"

  """
  引数として渡された動詞idに対する動作対象の抽出。取る可能性があるのはobj,obl,ccom系統。
  issue:cc/conjへの対応追加
  引数はextract_verbsに同じ
  """
  def extract_object(self, dep, dependent_root_id):
    for i in dep:
      if (("obj" in i[1]) or ("obl" in i[1])) and (i[0].id == dependent_root_id):
        return i[2].id, i[2].text
      if (("ccomp" in i[1]) or ("xcomp" in i[1])) and (i[0].id == dependent_root_id):
        return i[2].id, [i[2].text]
    return -1, "<none>"

  """
  節ごとの依存関係抽出(途中で目的節、主語節が見つかった場合、先に内部を抽出しに行く)
  dep: 現在の行の依存関係(linearn_dependencies)
  mcv_id: main clausal verbs id(主節動詞のindex)
  linear_result: 現在の行の抽出結果 [verbs, subject, object]
  """
  def extract_clausal_dependencies(self, dep, mcv_id, linear_result):
    current_subject_id, current_subject_text = self.extract_subject(dep, mcv_id)
    if type(current_subject_text) == type([]):
      linear_result.append(self.extract_clausal_dependencies(dep, current_subject_id, current_subject_text))
    else:
      linear_result.append(current_subject_text)

    current_object_id, current_object_text = self.extract_object(dep, mcv_id)
    if type(current_object_text) == type([]):
      linear_result.append(self.extract_clausal_dependencies(dep, current_object_id, current_object_text))
    else:
      linear_result.append(current_object_text)
    return linear_result

  """
  依存関係の抽出
  current_depends
  """
  def extract_dependencies(self, line_dependencies):
    print(line_dependencies)
    # 探索した依存関係の根側単語id(初期値はrootのため0)
    dependent_root_id = 0
    line_result = []

    # 主節動詞抽出
    main_clausal_verbs_id, main_clausal_verbs_text = self.extract_verbs(line_dependencies, 0)
    line_result.append(main_clausal_verbs_text)

    # 主節動詞主体/対象抽出
    line_result = self.extract_clausal_dependencies(line_dependencies, main_clausal_verbs_id, line_result)
    # サンプル用に1文章のみ実行して終了している
    return line_result

  """
  stanzaを用いた文章の解析。解析結果を1行ずつextract_dependsに渡し、依存関係を抽出する
  """
  def sentence_analysis(self, current_lines):
    current_depends = self.nlp(current_lines)
    for analysis_results in current_depends.sentences:
      line_result = self.extract_dependencies(analysis_results.dependencies)
    return line_result

# 実行関数
def run(filepath):
  # ファイルの読み込み
  loaded_file = file_load(filepath)
  all_result = []

  # 構文解析用クラスの設定
  da = DependencyAnalysis(config_gpu=False)
  # ファイルが大規模になった場合に引っかかりそう。そして処理も重そう。そもそもloadで引っかかる可能性もある。
  # それでも、できればreadlineで抜き出して正規表現で""を弾いてあげたい。
  for row in csv.reader(loaded_file):
    current_dependencies = []
    # 手が空いたらやる
    # fixed_lines = re.sub('("[^"]*),([^"]*")','###comma###',i)
    # ファイル内1行をカンマ区切りで分割されたリストのループ
    for idx, sentence in enumerate(row):
      # 現状 "タイトル,文章,文章..." で成り立っているファイルを想定しているため、最初の一区切りはタイトルとして語彙のみデータとして持って次のループへ行く
      if idx == 0:
        current_dependencies.append(sentence)
        continue

      # 各行の依存関係の抽出
      result = da.sentence_analysis(sentence)
      #print("current line:"+sentence)
      #print("extract result: " + str(result))
      current_dependencies.append(result)
    print("line ended")
    all_result.append(current_dependencies)
  file_close(loaded_file)
  exit()
  return 0


# 現状ROCStoriesのデータセットを想定して作成
# 最初のfileloadとcsv解析あたりをいじればどのようなファイルでも対応可能
if __name__ == "__main__":
  # 既に実行端末でダウンロード済みであれば以下一行は必要なし
  # stanza.download('en')

  filepath = "./sample"
  fixed_data = run(filepath)
