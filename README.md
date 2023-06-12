# script_extraction
データ抽出用(仮)  
文章に対して、現状以下のようなリスト型でデータを抽出する(Nは文章数)  
[Title, [verbs1, subj1, obj1] ... [verbsN, subjN, objN]]
list to sentenceは難しくないためとりあえず後回し、まずはfeaturesについて調べておく  

## 現状対応データ  
title\_words,sentence,sentence,sentence...  
title\_words,sentence,sentence,sentence...  

のようなファイルに対応している。(カンマ区切りでタイトルと文章が分けられているファイル、かつ、一行毎に文章が入っているファイル)  
データセットとしてはRODCStoriesDatasetを想定。  
他のデータセットに対応したい場合は序盤のデータ処理の部分を変更、それかクラス中でファイルを開くのを変更してしまってクラス自体を解析用クラスとしてしまえばよいため、修正予定としておく。  

