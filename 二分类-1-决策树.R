rm(list = ls())
source("tidyfuncs4cls2.R")
library(readxl)
library(tidyverse)
library(tidymodels)
library(mice)
library(dplyr)
library(tableone)
library(writexl) 
library(Boruta)
# 加载必要的包
library(kernlab)
tidymodels_prefer()#设置首选项，使其更倾向于使用 tidymodels 包中的函数和方法
# install.packages("tidymodels")
# 读取数据
# 读取训练集和验证集
load("data.RData")
# 将目标变量转换为因子类型
traindata$Early_recurrence <- as.factor(traindata$Early_recurrence)
testdata$Early_recurrence <- as.factor(testdata$Early_recurrence)
# 设定阳性类别和阴性类别
yourpositivelevel <- "1"  # 根据原始数据集中的实际值设定
yournegativelevel <- "0"   # 根据原始数据集中的实际值设定

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata, v = 5, strata = Early_recurrence)
folds

# 数据预处理配方
datarecipe_dt <- recipe(Early_recurrence ~ ., traindata)
datarecipe_dt

# 设定模型
model_dt <- decision_tree(
  mode = "classification",
  engine = "rpart",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_args(model=TRUE)
model_dt

# workflow
wk_dt <- 
  workflow() %>%
  add_recipe(datarecipe_dt) %>%
  add_model(model_dt)
wk_dt

##############################################################

############################  超参数寻优2选1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_dt <- parameters(
  tree_depth(range = c(3, 7)),
  min_n(range = c(5, 10)),
  cost_complexity(range = c(-6, -1))
) %>%
  # grid_regular(levels = c(3, 2, 4)) # 常规网格
  grid_random(size = 20) # 随机网格
  # grid_latin_hypercube(size = 10) # 拉丁方网格
  # grid_max_entropy(size = 10) # 最大熵网格
hpgrid_dt
log10(hpgrid_dt$cost_complexity)
# 网格也可以自己手动生成expand.grid()
# hpgrid_dt <- expand.grid(
#   tree_depth = c(2:5),
#   min_n = c(5, 11),
#   cost_complexity = 10^(-5:-1)
# )

# 交叉验证网格搜索过程
set.seed(42)
tune_dt <- wk_dt %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_dt,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_dt <- wk_dt %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_dt <- tune_dt %>%
  collect_metrics()
eval_tune_dt

# 图示
autoplot(tune_dt)
eval_tune_dt %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost_complexity', values = ~cost_complexity),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "DT HPO Guided by AUCROC")

# 经过交叉验证得到的最优超参数
hpbest_dt <- tune_dt %>%
  select_by_one_std_err(metric = "roc_auc", desc(cost_complexity))
hpbest_dt

# 采用最优超参数组合训练最终模型
set.seed(42)
final_dt <- wk_dt %>%
  finalize_workflow(hpbest_dt) %>%
  fit(traindata)
final_dt


##################################################################

# 训练集预测评估
predtrain_dt <- eval4cls2(
  model = final_dt, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "DT", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_dt$prediction
predtrain_dt$predprobplot
predtrain_dt$predprobplot_fill
predtrain_dt$rocresult
predtrain_dt$rocplot
predtrain_dt$prresult
predtrain_dt$prplot
predtrain_dt$cmresult
predtrain_dt$cmplot
predtrain_dt$metrics
predtrain_dt$diycutoff
predtrain_dt$ksplot
predtrain_dt$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_dt$proc)
pROC::ci.auc(predtrain_dt$proc)

# 预测评估测试集预测评估
predtest_dt <- eval4cls2(
  model = final_dt, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "DT", 
  datasetname = "testdata",
  cutoff = predtrain_dt$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)#DT8测试集的校准曲线
predtest_dt$prediction
predtest_dt$predprobplot
predtest_dt$predprobplot_fill
predtest_dt$rocresult
predtest_dt$rocplot
predtest_dt$prresult
predtest_dt$prplot
predtest_dt$cmresult
predtest_dt$cmplot
predtest_dt$metrics
predtest_dt$diycutoff
predtest_dt$ksplot
predtest_dt$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_dt$proc)
pROC::ci.auc(predtest_dt$proc)

# ROC比较检验
pROC::roc.test(predtrain_dt$proc, predtest_dt$proc)

# 合并训练集和测试集上ROC曲线
predtrain_dt$rocresult %>%
  bind_rows(predtest_dt$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
# 合并训练集和测试集上PR曲线
predtrain_dt$prresult %>%
  bind_rows(predtest_dt$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_dt$metrics %>%
  bind_rows(predtest_dt$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

library(readr)

# 合并训练集和测试集上的性能指标
combined_metrics <- predtrain_dt$metrics %>%
  bind_rows(predtest_dt$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# 最优超参数交叉验证的结果
evalcv_dt <- bestcv4cls2(
  wkflow = wk_dt,
  tuneresult = tune_dt,
  hpbest = hpbest_dt,
  yname = "Early_recurrence",
  modelname = "DT",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_dt$plotcv
evalcv_dt$evalcv
model_dt
# 保存评估结果
save(datarecipe_dt,
     model_dt,
     wk_dt,
     # hpgrid_dt, 
     tune_dt,
     predtrain_dt,
     predtest_dt,
     evalcv_dt,
     file = ".\\cls2\\evalresult_dt.RData")

