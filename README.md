## 结合GraphQSAT与DeepGate2代码进行测试

## 1.1 文件说明

- data文件夹：存储GQSAT初始的CNF数据
- aigdata文件夹：存储进行AIG转换后的aig相关数据
- deepgate文件夹：保存当前的deepgate2 model的相关API
- deepgatesat文件夹：对gqsat相关内容重新整合
- dqn_modified.py：对当前的框架进行重新整合

## 1.2 运行命令

（1）后台运行命令

nohup python -u dqn_modified.py > log.txt 2>&1 &

当前问题：

每个episode执行即为完成一个cnf或者circuit的赋值

GraphQSAT网络更新可以使得cnf在每个episode的执行步数逐渐变小，即cnf消去越来越快，因此，后续的episode需要进行网络更新的次数也会逐渐减小

在learner.step_ctr一定（50000）的情况下，cnf的episode数量会比较多

DeepGate预训练网络+MLP更新目前没有使得circuit的执行步数逐渐变小，因此，后续的episode需要进行网络更新的次数和之前差不多，所以cnf的episode数量也比较少
