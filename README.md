# Hybrid quantum-classical methods

用于学习和研究混合量子经典动力学，目前实现共享谐振环境中的
多态、多模式 Ehrenfest 轨迹平均。

## 当前模型

CPC60 MSH 模型，包含三个参与电子传播的激发态和
600 个质量加权核坐标。基态 G 用于定义核初始热 Wigner 分布。

## 安装

python -m pip install -r requirements.txt

## 参数检查

python cpc60_msh/run_cpc60_msh.py --conf 3 --prepare-only

## 短时间测试

python cpc60_msh/run_cpc60_msh.py --conf 3 --ntraj 4 --t-end-fs 10 --no-show

## 文件说明

- cpc60_msh/multistate_ehrenfest.py：通用 Ehrenfest 传播器。
- cpc60_msh/run_cpc60_msh.py：CPC60 参数读取和运行入口。
- cpc60_msh/data/：原始模型参数。
- cpc60_msh/tight_check/：已有的短时间积分检查结果。
- cpc60_msh/study.ipynb：学习与分析 notebook。

详细说明见 cpc60_msh/README_ZH.md。

## 当前验证范围

已保存短时间、少量轨迹的实现和积分容差检查。
长时间积分与轨迹数收敛仍需进一步验证。
