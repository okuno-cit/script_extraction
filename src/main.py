# 本コードの場合$CORENLP_HOMEの設定が必要ない
import sys
import re
import os
import csv

import stanza

class DependencyAnalysis:
  # filepath: 読み込みファイルのパス
  # config_gpu: 依存関係解析モデルをGPU上で動かすか否か
  def __init__(self, filepath, config_gpu=False):
    self.all_result = []
    self.filepath = filepath
    # gpuマシンで動かす場合、Falseを変更
    self.nlp = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma, depparse', use_gpu=config_gpu)

  # 一旦大規模ファイルは気にしない(メモリに乗りきるサイズのみを想定)
  def file_load(self):
    if os.path.isfile(self.filepath):
      print("load file:" + self.filepath)
      self.load_file = open(self.filepath, 'r')

  def file_close(self):
    if os.path.isfile(self.filepath):
      self.load_file.close()

  """
  主節動詞の抽出(非主節の場合は動詞はそもそも抽出されているのでこの関数を実行しない)
  dep: 現在の文章における解析された依存関係
  dependent_root_id: 解析したい依存関係における、より根に近い単語のID
  """
  def extract_verbs(self, dep, dependent_root_id):
    for i in dep:
      if (i[1] == "root") or (i[0].id == dependent_root_id):
        if 'VB' in i[2].xpos:
          return i[2].id, i[2].text
    return self.extract_verbs(dep, i[2].id)

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
  def extract_dependencies(self, current_depends):
    # 探索した依存関係の根側単語id(初期値はrootのため0)
    dependent_root_id = 0
    linear_result = []
    linear_dependencies = current_depends.sentences[0].dependencies

    # 主節動詞抽出
    main_clausal_verbs_id, main_clausal_verbs_text = self.extract_verbs(linear_dependencies, 0)
    linear_result.append(main_clausal_verbs_text)

    # 主節動詞主体/対象抽出
    linear_result = self.extract_clausal_dependencies(linear_dependencies, main_clausal_verbs_id, linear_result)
    print(linear_result)
    # サンプル用に1文章のみ実行して終了している
    exit()
    return linear_result

  # 実行関数
  def run(self):
    # ファイルの読み込み
    self.file_load()
    # ファイルが大規模になった場合に引っかかりそう。そして処理も重そう。そもそもloadで引っかかる可能性もある。
    # それでも、できればreadlineで抜き出して正規表現で""を弾いてあげたい。
    for row in csv.reader(self.load_file):
      current_dependencies = []
      for idx, sentence in enumerate(row):
        if idx == 0:
          current_dependencies.append([sentence])
          continue
        # 手が空いたらやる
        # fixed_lines = re.sub('("[^"]*),([^"]*")','###comma###',i)

        # 各行の依存関係の抽出
        result = self.nlp(sentence)
        print("current line:"+result.text)
        current_dependencies.append(self.extract_dependencies(result))


# 現状ROCStoriesのデータセットを想定して作成
# 最初のfileloadとcsv解析あたりをいじればどのようなファイルでも対応可能
if __name__ == "__main__":
  # 既に実行端末でダウンロード済みであれば以下一行は必要なし
  # stanza.download('en')

  da = DependencyAnalysis("./sample", config_gpu=False)
  fixed_data = da.run()
