# Debiasing
天池2020KDD-Debiasing
学习paddlepaddle推荐系统后，参加天池比赛练练手。
先使用skip-gram制作item-embedding，再融合其他数据，进入GRU模型进行训练和预测。

使用官方的evaluate函数，得到的结果：

TrackA:
current_time: 1589558400
date_time: 2020-05-16 00:00:00
current_phase: 6
|  ndcg_50_full   | ndcg_50_half  | hitrate_50_full | hitrate_50_half |
|  ----  | ----  | ----  | ----  |
| 0.14619681 | 0.14513455 | 0.33132893 | 0.31195652 |
| 0.31854808 | 0.32222784 | 0.71777153 | 0.67700076 |
| 0.47687775 | 0.48702055 | 1.0964698 | 1.0552514 |
| 0.6313792 | 0.6583645 | 1.464828 | 1.4363703 |
| 0.793236 | 0.8445185 | 1.8518304 | 1.8506722 |
| 1.0163158 | 1.1028891 | 2.290095 | 2.3028173 |
| 1.1120732 | 1.198773 | 2.4855921 | 2.4698386 |



TrackB:
current_time: 1591315201
date_time: 2020-06-05 08:00:01
current_phase: 9

|  ndcg_50_full   | ndcg_50_half  | hitrate_50_full | hitrate_50_half |
|  ----  | ----  | ----  | ----  |
| 7.1581383 | 7.1730366 | 7.374513 | 7.3862143 |
| 7.343614 | 7.3943596 | 7.7694526 | 7.8206673 |
| 7.419655 | 7.469683 | 7.938973 | 7.9618974 |