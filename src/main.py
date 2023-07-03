"""
issue: stanzaで構造解析を行う際に物語内の文章全て入力して同時に解析するのと一行ずつ分けて解析する場合とどちらが速度が速いか
"""
import sys
import re
import os
import csv
import gc

import stanza


def file_write(filepath, results):
  if os.path.isfile(filepath):
    print("edit file:" + filepath)
    with open(filepath, mode='a') as f:
        f.writelines(results)
  else:
    print("create file:" + filepath)
    with open(filepath, mode='w') as f:
        f.writelines(results)

class DependencyAnalysis:
  # filepath: 読み込みファイルのパス
  # config_gpu: 依存関係解析モデルをGPU上で動かすか否か
  def __init__(self, config_gpu=False):
    self.all_result = []
    # gpuマシンで動かす場合、Falseを変更
    self.nlp = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma, depparse', use_gpu=config_gpu)
    #最終的に書く文章から平均幾つの語彙、単語、節が抽出されているか調べたいので三つの値を持つ
    #文章からスクリプト形式に抽出した全ての語彙数(John and Maryでも1)
    self.word_count = 0
    #文章からスクリプト形式に抽出した全ての単語数(John and Maryで3)
    self.word_sequence = 0
    #節数
    self.clausal_count = 0

  """
  主節動詞の抽出(非主節の場合は動詞はそもそも抽出されているのでこの関数を実行しない)
  dep: 現在の文章における解析された依存関係
  dependent_root_id: 解析したい依存関係における、より根に近い単語のID
  """
  def extract_verbs(self, dep, dependent_root_id):
    for i in range(len(dep)):
      for d in dep:
        if (d[1] == 'root') or (d[0].id == dependent_root_id):
          if 'VB' in d[2].xpos:
            return d[2].id, '<v> '+d[2].text
          elif dependent_root_id == 0:
            dependent_root_id = d[2].id
    return -1, '<v> <none>'

  """
  入力された動詞idに対する動作主体の抽出。取る可能性のある依存関係はcsubj:~,nsubj:~のみ。
  issue:cc/conjへの対応追加
  引数はextract_verbsに同じ
  """
  def extract_subject(self, dep, dependent_root_id):
    for i in dep:
      if ('subj' in i[1]) and (i[0].id == dependent_root_id):
        if 'csubj' in i[1]:
          return i[2].id, ['<cs> '+i[2].text]
        elif 'nsubj' in i[1]:
          cc_result = self.extract_conjunction(dep, i[2].id)
          if cc_result:
            return i[2].id, '<s> '+i[2].text+cc_result
          return i[2].id, '<s> '+i[2].text
    return -1, '<s> <none>'

  """
  引数として渡された動詞idに対する動作対象の抽出。取る可能性があるのはobj,obl,ccom系統。
  issue:cc/conjへの対応追加
  引数はextract_verbsに同じ
  """
  def extract_object(self, dep, dependent_root_id):
    for i in dep:
      if ('obj' in i[1]) and (i[0].id == dependent_root_id):
        cc_result = self.extract_conjunction(dep, i[2].id)
        if cc_result:
          return i[2].id, '<o> '+i[2].text+cc_result
        return i[2].id, '<o> '+i[2].text
      if (('ccomp' in i[1]) or ('xcomp' in i[1])) and (i[0].id == dependent_root_id):
        return i[2].id, ['<co> '+i[2].text]
    return -1, '<o> <none>'

  """
  対象/主体に対する接続詞抽出(conj)
  dep: 現在の文章における解析された依存関係
  dependent_root_id: 接続詞を探索したい単語のid
  """
  def extract_conjunction(self, dep, dependent_root_id):
    for d in dep:
      if ('conj' in d[1]) and (d[0].id == dependent_root_id):
        return self.extract_cc(dep, d[2].id)+d[2].text
    return False

  """
  conjに対するcc(coordination conjunction等位接続詞自身)の抽出
  dep: 現在の文章における解析された依存関係
  dependent_root_id: 接続詞を探索したい単語のid
  """
  def extract_cc(self, dep, dependent_root_id):
    for d in dep:
      if ('cc' == d[1]) and (d[0].id == dependent_root_id):
        return d[2].text + ' '
    return ''

  """
  節ごとの依存関係抽出(途中で目的節、主語節が見つかった場合、先に内部を抽出しに行く)
  dep: 現在の行の依存関係(line_dependencies)
  mcv_id: main clausal verbs id(主節動詞のindex)
  line_result: 現在の行の抽出結果 [verbs, subject, object]
  """
  def extract_clausal_dependencies(self, dep, mcv_id, line_result):
    current_subject_id, current_subject_text = self.extract_subject(dep, mcv_id)
    if type(current_subject_text) == type([]):
      line_result=line_result+' '+self.extract_clausal_dependencies(dep, current_subject_id, current_subject_text[0])
    else:
      line_result=line_result+' '+current_subject_text

    current_object_id, current_object_text = self.extract_object(dep, mcv_id)
    if type(current_object_text) == type([]):
      line_result=line_result+' '+self.extract_clausal_dependencies(dep, current_object_id, current_object_text[0])
    else:
      line_result=line_result+' '+current_object_text
    return line_result

  """
  依存関係の抽出
  """
  def extract_dependencies(self, line_dependencies):
    # 探索した依存関係の根側単語id(初期値はrootのため0)
    dependent_root_id = 0
    line_result = ''

    # 主節動詞抽出
    main_clausal_verbs_id, main_clausal_verbs_text = self.extract_verbs(line_dependencies, 0)
    line_result = line_result + main_clausal_verbs_text

    # 主節動詞主体/対象抽出
    line_result = self.extract_clausal_dependencies(line_dependencies, main_clausal_verbs_id, line_result)
    return line_result

  """
  stanzaを用いた文章の解析。解析結果を1行ずつextract_dependsに渡し、依存関係を抽出する
  """
  def sentence_analysis(self, current_lines):
    line_result = ''
    current_depends = self.nlp(current_lines)
    # 念のため一行ずつの処理(複数文章nlpに渡すとsentencesが長さ1以上のリストになる)
    for analysis_results in current_depends.sentences:
      line_result = line_result + self.extract_dependencies(analysis_results.dependencies)
    return line_result

# 実行関数
def run(filepath):
  load_file = open(filepath, 'r')
  output_file = filepath+'_extracted'
  all_result = []
  da = DependencyAnalysis(config_gpu=False)
  for line_num, row in enumerate(csv.reader(load_file)):
    if line_num % 1000 == 0:
      print('execute: ' + str(line_num))
      file_write(output_file, all_result)
      all_result = []
    current_dependencies = ''
    for idx, sentence in enumerate(row):
      # デフォルトは"タイトル,文章,文章..." で成り立っているファイルを想定しているため、最初の一区切りはタイトルとして語彙のみデータとして持って次のループへ行く
      if idx == 0:
        current_dependencies = current_dependencies + sentence + ' <EOT> '
        continue

      # 各行の依存関係の抽出
      result = da.sentence_analysis(sentence)
      #current_dependencies.append(result)
      current_dependencies = current_dependencies + result + ' # '
    all_result.append(current_dependencies+' <EOSC>\n')
  file_write(output_file, all_result)
  load_file.close()

  with open(output_file, mode='w') as f:
      f.writelines(all_result)

# 現状ROCStoriesのデータセットを想定して作成
# もし大規模になった場合は最初のfileloadとcsv解析あたりをいじればどのようなファイルでも対応可能
if __name__ == '__main__':
  # 既に実行端末でダウンロード済みであれば以下一行は必要なし
  # stanza.download('en')

  filepath = './rocs_all.csv'
  #filepath = './sample'
  run(filepath)
