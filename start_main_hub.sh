#!/bin/bash

# スクリプトが置かれているディレクトリに移動
# dirname "${BASH_SOURCE[0]}" で、このスクリプト自体のディレクトリパスを取得
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"

# Python 3 インタープリタを使って main_hub.py を実行
# /usr/bin/env python3 は環境に依存せず python3 を見つけ実行する最も一般的な方法
/usr/bin/env python3 main_hub.py