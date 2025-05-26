library(bonsai)             # 加载 bonsai
# 数据预处理配方
datarecipe_lightgbm <- recipe(Early_recurrence ~ ., traindata)
datarecipe_lightgbm

# 设定模型
model_lightgbm <- boost_tree(
  mode = "classification",
  engine = "lightgbm",
  tree_depth = tune(),
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune()
)
model_lightgbm

# workflow
wk_lightgbm <- 
  workflow() %>%
  add_recipe(datarecipe_lightgbm) %>%
  add_model(model_lightgbm)
wk_lightgbm

##############################################################

############################  超参数寻优1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_lightgbm <- parameters(
  tree_depth(range = c(1, 3)),
  trees(range = c(100, 500)),
  learn_rate(range = c(-3, -1)),
  mtry(range = c(2, 8)),
  min_n(range = c(5, 10)),
  loss_reduction(range = c(-3, 0))
) %>%
  # grid_regular(levels = c(3, 2, 2, 3, 2, 2)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_lightgbm
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_lightgbm,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2-贝叶斯优化
library(bonsai)
# 更新超参数范围
param_lightgbm <- model_lightgbm %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_lightgbm,
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
eval_tune_lightgbm <- tune_lightgbm %>%
  collect_metrics()
eval_tune_lightgbm

# 图示
# autoplot(tune_lightgbm)
eval_tune_lightgbm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction)
    )
  ) %>%
  plotly::layout(title = "lightgbm HPO Guided by AUCROC")

# 经过交叉验证得到的最优超参数
hpbest_lightgbm <- tune_lightgbm %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_lightgbm

# 采用最优超参数组合训练最终模型
set.seed(42)
final_lightgbm <- wk_lightgbm %>%
  finalize_workflow(hpbest_lightgbm) %>%
  fit(traindata)
final_lightgbm
# 保存模型为RDS文件
saveRDS(final_lightgbm, "final_lightgbm_model.rds")
##################################################################

# 训练集预测评估
predtrain_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "Lightgbm", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_lightgbm$prediction
predtrain_lightgbm$predprobplot
predtrain_lightgbm$predprobplot_fill
predtrain_lightgbm$rocresult
predtrain_lightgbm$rocplot
predtrain_lightgbm$prresult
predtrain_lightgbm$prplot
predtrain_lightgbm$cmresult
predtrain_lightgbm$cmplot
predtrain_lightgbm$metrics
predtrain_lightgbm$diycutoff
predtrain_lightgbm$ksplot
predtrain_lightgbm$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_lightgbm$proc)
pROC::ci.auc(predtrain_lightgbm$proc)

# 预测评估测试集预测评估
predtest_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "Lightgbm", 
  datasetname = "testdata",
  cutoff = predtrain_lightgbm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_lightgbm$prediction
predtest_lightgbm$predprobplot
predtest_lightgbm$predprobplot_fill
predtest_lightgbm$rocresult
predtest_lightgbm$rocplot
predtest_lightgbm$prresult
predtest_lightgbm$prplot
predtest_lightgbm$cmresult
predtest_lightgbm$cmplot
predtest_lightgbm$metrics
predtest_lightgbm$diycutoff
predtest_lightgbm$ksplot
predtest_lightgbm$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_lightgbm$proc)
pROC::ci.auc(predtest_lightgbm$proc)

# ROC比较检验
pROC::roc.test(predtrain_lightgbm$proc, predtest_lightgbm$proc)

# 合并训练集和测试集上ROC曲线
predtrain_lightgbm$rocresult %>%
  bind_rows(predtest_lightgbm$rocresult) %>%
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
predtrain_lightgbm$prresult %>%
  bind_rows(predtest_lightgbm$prresult) %>%
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
predtrain_lightgbm$metrics %>%
  bind_rows(predtest_lightgbm$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_lightgbm <- bestcv4cls2(
  wkflow = wk_lightgbm,
  tuneresult = tune_lightgbm,
  hpbest = hpbest_lightgbm,
  yname = "Early_recurrence",
  modelname = "Lightgbm",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_lightgbm$plotcv
evalcv_lightgbm$evalcv

# 保存评估结果
save(datarecipe_lightgbm,
     model_lightgbm,
     wk_lightgbm,
     # hpgrid_lightgbm,
     tune_lightgbm,
     predtrain_lightgbm,
     predtest_lightgbm,
     evalcv_lightgbm,
     file = ".\\cls2\\evalresult_lightgbm.RData")