
# 数据预处理配方
datarecipe_xgboost <- recipe(Early_recurrence ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names)

# 设定模型
model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) %>%
  set_args(validation = 0.2,
           event_level = "second")
model_xgboost

# workflow
wk_xgboost <- 
  workflow() %>%
  add_recipe(datarecipe_xgboost) %>%
  add_model(model_xgboost)
wk_xgboost

##############################################################

############################  超参数寻优1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_xgboost <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
) %>%
  # grid_regular(levels = c(3, 2, 2, 3, 2, 2)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_xgboost


# 交叉验证网格搜索过程
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_xgboost,
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

# 更新超参数范围
param_xgboost <- model_xgboost %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_xgboost,
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
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost

# 图示
autoplot(tune_xgboost)
eval_tune_xgboost %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction),
      list(label = 'sample_size', values = ~sample_size)
    )
  ) %>%
  plotly::layout(title = "xgboost HPO Guided by AUCROC")

# 经过交叉验证得到的最优超参数
hpbest_xgboost <- tune_xgboost %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_xgboost

# 采用最优超参数组合训练最终模型
set.seed(42)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(traindata)

# 训练集预测评估
predtrain_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "Xgboost", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_xgboost$prediction
predtrain_xgboost$predprobplot
predtrain_xgboost$predprobplot_fill
predtrain_xgboost$rocresult
predtrain_xgboost$rocplot
predtrain_xgboost$prresult
predtrain_xgboost$prplot
predtrain_xgboost$cmresult
predtrain_xgboost$cmplot
predtrain_xgboost$metrics
predtrain_xgboost$diycutoff
predtrain_xgboost$ksplot
predtrain_xgboost$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_xgboost$proc)
pROC::ci.auc(predtrain_xgboost$proc)

# 预测评估测试集预测评估
predtest_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "Xgboost", 
  datasetname = "testdata",
  cutoff = predtrain_xgboost$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_xgboost$prediction
predtest_xgboost$predprobplot
predtest_xgboost$predprobplot_fill
predtest_xgboost$rocresult
predtest_xgboost$rocplot
predtest_xgboost$prresult
predtest_xgboost$prplot
predtest_xgboost$cmresult
predtest_xgboost$cmplot
predtest_xgboost$metrics
predtest_xgboost$diycutoff
predtest_xgboost$ksplot
predtest_xgboost$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_xgboost$proc)
pROC::ci.auc(predtest_xgboost$proc)

# ROC比较检验
pROC::roc.test(predtrain_xgboost$proc, predtest_xgboost$proc)


# 合并训练集和测试集上ROC曲线
predtrain_xgboost$rocresult %>%
  bind_rows(predtest_xgboost$rocresult) %>%
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
predtrain_xgboost$prresult %>%
  bind_rows(predtest_xgboost$prresult) %>%
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
predtrain_xgboost$metrics %>%
  bind_rows(predtest_xgboost$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
# 合并训练集和测试集上性能指标
performance_metrics <- predtrain_xgboost$metrics %>%
  bind_rows(predtest_xgboost$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# 最优超参数交叉验证的结果
evalcv_xgboost <- bestcv4cls2(
  wkflow = wk_xgboost,
  tuneresult = tune_xgboost,
  hpbest = hpbest_xgboost,
  yname = "Early_recurrence",
  modelname = "Xgboost",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_xgboost$plotcv
evalcv_xgboost$evalcv

# 保存评估结果
save(datarecipe_xgboost,
     model_xgboost,
     wk_xgboost,
     param_xgboost,  # 如果采用贝叶斯优化则替换为 param_xgboost
     tune_xgboost,
     predtrain_xgboost,
     predtest_xgboost,
     evalcv_xgboost,
     file = ".\\cls2\\evalresult_xgboost.RData")